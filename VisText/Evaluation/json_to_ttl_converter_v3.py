#!/usr/bin/env python3
"""
JSON → TTL (Turtle) converter V2 (datatable + caption_L1 only)

This converter produces TTL that matches the schema used by the LLM predictions:
- Uses chart: namespace with #-based URIs (http://example.org/chart#)
- chart:category and chart:value for data points
- Compatible with the evaluation extractor using pred.endswith('#value')

Only these JSON fields are used:
- datatable
- caption_L1
(and img_id just to build a stable namespace)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


def escape_ttl_string(text: str) -> str:
    """Escape special characters for TTL string literals."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return text


def truncate_end(text: str, max_length: int = 35) -> str:
    """Truncate at the end and add '...'."""
    return text if len(text) <= max_length else text[:max_length] + "..."


def truncate_middle(text: str, max_length: int = 120) -> str:
    """Truncate in the middle and add '...' (keeps start + end)."""
    if len(text) <= max_length:
        return text
    if max_length <= 3:
        return text[:max_length]
    front = (max_length - 3) // 2 + (max_length - 3) % 2
    back = (max_length - 3) // 2
    return text[:front].rstrip() + "..." + text[-back:].lstrip()


def _label_words(label: str) -> List[str]:
    """Tokenize a label into alphanumeric words (for URI local-names)."""
    return re.findall(r"[A-Za-z0-9]+", label)


def _norm_token(token: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", token.lower())


def _find_subsequence(haystack: List[str], needle: List[str]) -> Optional[int]:
    """Return index of first occurrence of needle in haystack, else None."""
    if not needle:
        return None
    n = len(needle)
    for i in range(0, len(haystack) - n + 1):
        if haystack[i : i + n] == needle:
            return i
    return None


def parse_numeric_token(token: str) -> Tuple[str, bool]:
    """
    Parse numeric token and return (canonical_string, is_numeric).
    Accepts commas and percent signs (e.g. "1,234", "23%").
    """
    t = token.strip()
    t = t.strip("()[]{}")
    t = t.rstrip(",;")
    t = t.replace(",", "")
    t_no_pct = t.replace("%", "")

    try:
        float(t_no_pct)
        return t_no_pct, True
    except ValueError:
        return token, False


@dataclass
class CaptionInfo:
    chart_type: str = ""
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    y_min: str = ""
    y_max: str = ""
    
@dataclass
class PropertiesInfo:
    chart_type: str = ""
    title: str = ""
    x_label: str = ""
    y_label: str = ""

def parse_L1_properties(l1_properties: list) -> PropertiesInfo:
    info = PropertiesInfo()
    if not l1_properties:
        return info

    info.chart_type = l1_properties[0].strip().lower()
    info.title = l1_properties[1].strip()
    info.x_label = l1_properties[2].strip()
    info.y_label = l1_properties[3].strip()
    
    return info

def parse_caption_l1(caption_l1: str) -> CaptionInfo:
    """
    Extract (best-effort) chart type, title, axis labels, and y-range from caption_L1.
    """
    info = CaptionInfo()
    if not caption_l1:
        return info

    cap = " ".join(caption_l1.strip().split())
    low = cap.lower()

    # chart type
    m = re.search(r"\b(bar|line|pie|scatter|area)\s+(?:chart|diagram|graph|plot)\b", low)
    if m:
        info.chart_type = m.group(1)
    else:
        # Also check for "is a bar plot" pattern
        m2 = re.search(r"\bis\s+(?:a|an)\s+(bar|line|pie|scatter|area)\s+(?:chart|diagram|graph|plot)\b", low)
        if m2:
            info.chart_type = m2.group(1)
        else:
            for t in ("bar", "line", "pie", "scatter", "area"):
                if t in low:
                    info.chart_type = t
                    break

    # title - multiple patterns
    # Pattern 1: "called/titled/named X"
    m = re.search(
        r"\b(?:called|titled|named)\s+(.+?)(?:\s*\.\s*(?:The|On|A)|$)",
        cap,
        flags=re.IGNORECASE,
    )
    if m:
        info.title = m.group(1).strip().strip('"').rstrip('.')
    
    # Pattern 2: "is a bar plot named X"
    if not info.title:
        m = re.search(
            r"(?:bar|line|pie|scatter|area)\s+(?:chart|diagram|graph|plot)\s+(?:called|titled|named)\s+(.+?)(?:\s*\.\s*(?:The|On|A)|$)",
            cap,
            flags=re.IGNORECASE,
        )
        if m:
            info.title = m.group(1).strip().strip('"').rstrip('.')

    # x label - "x-axis shows/measures/displays X"
    m = re.search(
        r"\bx-?axis\s+(?:shows?|measures?|displays?|plots?)\s+([^.]+?)(?:\s+(?:as|along|on|using)\s+|\.|$)",
        cap,
        flags=re.IGNORECASE,
    )
    if m:
        info.x_label = m.group(1).strip().strip('"')
    
    # Fallback: "while the x-axis shows X"
    if not info.x_label:
        m = re.search(
            r"while\s+the\s+x-?axis\s+(?:shows?|measures?|displays?)\s+([^.]+?)(?:\s+(?:as|along)\s+|\.|$)",
            cap,
            flags=re.IGNORECASE,
        )
        if m:
            info.x_label = m.group(1).strip().strip('"')

    # y label - "y-axis measures/shows/labeled Y"
    m = re.search(
        r"\by-?axis\s+(?:measures?|shows?|displays?|labeled|labelled)\s+([^.]+?)(?:\s+(?:as|along|on|using|with)\s+|\.|$)",
        cap,
        flags=re.IGNORECASE,
    )
    if m:
        info.y_label = m.group(1).strip().strip('"')

    # y range - "minimum of X and maximum of Y" or "range X to Y"
    m = re.search(r"\bminimum\s+(?:of\s+)?([-\d\.,]+)\s+and\s+(?:a\s+)?maximum\s+(?:of\s+)?([-\d\.,]+)\b", cap, flags=re.IGNORECASE)
    if m:
        info.y_min = m.group(1).replace(",", "")
        info.y_max = m.group(2).replace(",", "")
    else:
        m = re.search(r"\brange\s+(?:of\s+)?([-\d\.,]+)\s*(?:to|–|-)\s*([-\d\.,]+)\b", cap, flags=re.IGNORECASE)
        if m:
            info.y_min = m.group(1).replace(",", "")
            info.y_max = m.group(2).replace(",", "")

    return info


def chart_type_string(chart_type: str) -> str:
    """Return chart type string for TTL."""
    t = (chart_type or "").lower()
    if "line" in t:
        return "line"
    if "pie" in t:
        return "pie"
    if "scatter" in t:
        return "scatter"
    if "area" in t:
        return "area"
    return "bar"


def parse_datatable(
    datatable: str,
    x_label_hint: str = "",
    y_label_hint: str = "",
) -> Tuple[str, str, str, List[Tuple[str, str]]]:
    """
    Parse datatable into:
      (title, x_label, y_label, [(category, value), ...])
    """
    if not datatable:
        return "", "", "", []

    if "<s>" in datatable:
        title_part, data_part = datatable.split("<s>", 1)
        title = " ".join(title_part.strip().split())
        data_section = data_part.strip()
    else:
        title = ""
        data_section = datatable.strip()

    # Multi-line: first line header, remaining rows.
    lines = [ln.strip() for ln in data_section.splitlines() if ln.strip()]
    if len(lines) > 1:
        header = lines[0]
        x_label = x_label_hint.strip()
        y_label = y_label_hint.strip()

        if not (x_label and y_label):
            header_tokens = header.split()
            if not x_label and header_tokens:
                x_label = header_tokens[0]
            if not y_label and len(header_tokens) > 1:
                y_label = " ".join(header_tokens[1:])

        points: List[Tuple[str, str]] = []
        for ln in lines[1:]:
            toks = ln.split()
            if len(toks) < 2:
                continue
            y_val, is_num = parse_numeric_token(toks[-1])
            x_val = " ".join(toks[:-1])
            points.append((x_val, y_val if is_num else toks[-1]))
        return title, x_label, y_label, points

    # One-line datatable
    data_section = " ".join(data_section.split())
    tokens = data_section.split()
    if not tokens:
        return title, x_label_hint.strip(), y_label_hint.strip(), []

    norm_tokens = [_norm_token(t) for t in tokens]

    x_label = x_label_hint.strip()
    y_label = y_label_hint.strip()
    data_start = 0

    # Prefer: find y_label_hint sequence inside tokens -> start data right after it.
    if y_label:
        y_seq = [_norm_token(t) for t in y_label.split()]
        idx = _find_subsequence(norm_tokens, y_seq)
        if idx is not None:
            data_start = idx + len(y_seq)
            if not x_label and idx > 0:
                x_label = " ".join(tokens[:idx])

    # Fallback: guess header length that maximizes extracted points.
    if data_start == 0:
        best_points: List[Tuple[str, str]] = []
        best_header_len = 0

        for header_len in range(1, min(10, len(tokens))):
            pts: List[Tuple[str, str]] = []
            x_parts: List[str] = []

            for tok in tokens[header_len:]:
                val, is_num = parse_numeric_token(tok)
                if is_num:
                    if x_parts:
                        pts.append((" ".join(x_parts), val))
                        x_parts = []
                else:
                    x_parts.append(tok)

            if len(pts) > len(best_points):
                best_points = pts
                best_header_len = header_len

        if best_points:
            data_start = best_header_len
            if not x_label:
                x_label = tokens[0]
            if not y_label and data_start > 1:
                y_label = " ".join(tokens[1:data_start])
            return title, x_label, y_label, best_points

    # Parse rows: (words...) + (number) repeating
    points: List[Tuple[str, str]] = []
    x_parts: List[str] = []
    for tok in tokens[data_start:]:
        val, is_num = parse_numeric_token(tok)
        if is_num:
            if x_parts:
                points.append((" ".join(x_parts), val))
                x_parts = []
        else:
            x_parts.append(tok)

    return title, x_label, y_label, points


class JSONToTTLConverterV2:
    """
    Converts JSON chart data to TTL using ONLY datatable + caption_L1.
    
    Output uses the same schema as LLM predictions:
    - chart: namespace with #-based URIs
    - chart:category and chart:value for data points
    """

    def __init__(
        self,
        base_uri: str = "http://example.org/",
        truncate_point_label_len: int = 50,
        truncate_title_len: int = 150,
        sort_points: bool = False,  # Don't sort to preserve original order
    ):
        self.base_uri_template = base_uri
        self.truncate_point_label_len = truncate_point_label_len
        self.truncate_title_len = truncate_title_len
        self.sort_points = sort_points

    def convert_json_to_ttl(self, json_data: Dict[str, Any]) -> str:
        # Handle empty lists or non-dict data
        if not isinstance(json_data, dict):
            raise ValueError(f"Expected dict, got {type(json_data).__name__}")
        
        img_id = str(json_data.get("img_id", "unknown"))

        datatable = json_data.get("datatable", "") or ""
        # caption_l1 = json_data.get("caption_L1", "") or ""
        l1_properties = json_data.get("L1_properties", "") or ""
        
        # cap = parse_caption_l1(caption_l1)
        properties = parse_L1_properties(l1_properties)
        dt_title, dt_x, dt_y, points = parse_datatable(
            datatable,
            x_label_hint=properties.x_label,
            y_label_hint=properties.y_label,
        )

        title = properties.title or dt_title or ""
        x_label = properties.x_label or dt_x or ""
        y_label = properties.y_label or dt_y or ""
        chart_type = chart_type_string(properties.chart_type)
        
        # print("Title:", title)
        # print("X Label:", x_label)
        # print("Y Label:", y_label)
        # print("Chart Type:", chart_type)

        # Build TTL matching the prediction schema
        ttl_lines: List[str] = [
            f"@prefix ex:    <http://example.org/> .",
            f"@prefix chart: <http://example.org/chart#> .",
            f"@prefix xsd:   <http://www.w3.org/2001/XMLSchema#> .",
            "",
            f"ex:chart-{img_id} a chart:Chart ;",
        ]
        
        if chart_type:
            ttl_lines.append(f'    chart:chartType "{chart_type}" ;')
        
        if title:
            ttl_lines.append(f'    chart:title "{escape_ttl_string(truncate_middle(title, self.truncate_title_len))}" ;')
        
        ttl_lines.append(f"    chart:xAxis ex:x-{img_id} ;")
        ttl_lines.append(f"    chart:yAxis ex:y-{img_id} .")
        ttl_lines.append("")
        
        # X-axis
        ttl_lines.append(f"ex:x-{img_id} a chart:Axis ;")
        if x_label:
            ttl_lines.append(f'    chart:label "{escape_ttl_string(x_label)}" .')
        else:
            ttl_lines.append(f'    chart:label "" .')
        ttl_lines.append("")
        
        # Y-axis
        ttl_lines.append(f"ex:y-{img_id} a chart:Axis ;")
        if y_label:
            ttl_lines.append(f'    chart:label "{escape_ttl_string(y_label)}" .')
        else:
            ttl_lines.append(f'    chart:label "" .')
        ttl_lines.append("")

        # Sort if requested
        if self.sort_points:
            points = sorted(points, key=lambda p: p[0].lower())

        # Data points - using chart:category and chart:value (with #-based URIs)
        for i, (category, value_raw) in enumerate(points, start=1):
            category_clean = truncate_end(category, self.truncate_point_label_len)
            value, is_num = parse_numeric_token(value_raw)

            ttl_lines.append(f"ex:DataPoint{i} a chart:DataPoint ;")
            ttl_lines.append(f'    chart:independentVariable "{escape_ttl_string(category_clean)}" ;')
            
            if is_num:
                if '.' in value:
                    ttl_lines.append(f'    chart:value "{value}"^^xsd:decimal .')
                else:
                    ttl_lines.append(f'    chart:value "{value}"^^xsd:integer .')
            else:
                ttl_lines.append(f'    chart:value "{escape_ttl_string(str(value_raw))}" .')

            ttl_lines.append("")

        return "\n".join(ttl_lines).rstrip() + "\n"

    def convert_file(self, json_file_path: str, output_file_path: Optional[str] = None) -> str:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        ttl_content = self.convert_json_to_ttl(json_data)

        if output_file_path is None:
            base_name = os.path.splitext(os.path.basename(json_file_path))[0]
            output_file_path = f"{base_name}.ttl"

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(ttl_content)

        return output_file_path

    def convert_directory(self, input_dir: str, output_dir: Optional[str] = None) -> None:
        if output_dir is None:
            output_dir = input_dir

        os.makedirs(output_dir, exist_ok=True)

        json_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".json")]
        json_files.sort()

        converted = 0
        errors = 0
        for json_file in json_files:
            input_path = os.path.join(input_dir, json_file)
            output_filename = os.path.splitext(json_file)[0] + ".ttl"
            output_path = os.path.join(output_dir, output_filename)

            try:
                self.convert_file(input_path, output_path)
                converted += 1
            except Exception as e:
                print(f"✗ Error converting {json_file}: {e}")
                errors += 1
        
        print(f"Converted {converted} files, {errors} errors")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert JSON chart data to TTL V2 (datatable + caption_L1 only, matching LLM prediction schema)"
    )
    parser.add_argument("input", help="Input JSON file or a directory containing JSON files")
    parser.add_argument("-o", "--output", help="Output TTL file (if input is a file) OR output directory (if input is a directory)")
    parser.add_argument("--sort", action="store_true", help="Alphabetically sort data points by category")
    parser.add_argument("--truncate-label", type=int, default=50, help="Max chars for each category label")
    parser.add_argument("--truncate-title", type=int, default=150, help="Max chars for title")

    args = parser.parse_args()

    converter = JSONToTTLConverterV2(
        truncate_point_label_len=args.truncate_label,
        truncate_title_len=args.truncate_title,
        sort_points=args.sort,
    )

    if os.path.isfile(args.input):
        out_path = converter.convert_file(args.input, args.output)
        print(f"✓ Successfully converted to: {out_path}")
        return 0

    if os.path.isdir(args.input):
        converter.convert_directory(args.input, args.output)
        print("\n✓ Conversion complete!")
        return 0

    print(f"Error: {args.input} is not a valid file or directory")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
