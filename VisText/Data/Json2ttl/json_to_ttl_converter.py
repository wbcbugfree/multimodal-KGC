#!/usr/bin/env python3
"""
JSON → TTL (Turtle) converter (datatable + caption_L1 only)

Only these JSON fields are used:
- datatable
- caption_L1
(and img_id just to build a stable namespace)

Output TTL structure (like your example):
- :Chart1 a schema:BarChart ;
- :DataPoint1 a :Response ; rdfs:label "..."; :shareOfRespondents "0.22"^^xsd:decimal ; :belongsToChart :Chart1 .
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


def pascal_case(label: str, default: str = "Thing") -> str:
    """Convert label → PascalCase local name (class names)."""
    parts = _label_words(label)
    if not parts:
        return default
    return "".join(p[:1].upper() + p[1:] for p in parts)


def lower_camel_case(label: str, default: str = "value") -> str:
    """Convert label → lowerCamelCase local name (predicate names)."""
    parts = _label_words(label)
    if not parts:
        return default
    first = parts[0].lower()
    rest = "".join(p[:1].upper() + p[1:] for p in parts[1:])
    return first + rest


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
    m = re.search(r"\b(bar|line|pie|scatter)\s+(?:chart|diagram|graph|plot)\b", low)
    if m:
        info.chart_type = m.group(1)
    else:
        for t in ("bar", "line", "pie", "scatter"):
            if t in low:
                info.chart_type = t
                break

    # title (stop before axis sentence if present)
    m = re.search(
        r"\b(?:called|titled|named)\s+(.*?)(?:\s*\.\s*(?:On|The)\s+the\s+x-?axis|\s*\.\s*(?:On|The)\s+x-?axis|\s*\.\s*A\s+linear|\s*\.\s*$)",
        cap,
        flags=re.IGNORECASE,
    )
    if m:
        info.title = m.group(1).strip().strip('"')

    # x label
    m = re.search(
        r"\bOn\s+the\s+x-?axis,?\s*(.*?)(?:\s+is\s+(?:defined|shown|given|labeled|labelled)|\.)",
        cap,
        flags=re.IGNORECASE,
    )
    if m:
        info.x_label = m.group(1).strip().strip('"')

    # y label
    m = re.search(r"\by-?axis,?\s*(?:labeled|labelled)\s+(.*?)(?:\.|$)", cap, flags=re.IGNORECASE)
    if m:
        info.y_label = m.group(1).strip().strip('"')

    # y range
    m = re.search(r"\brange\s+([-\d\.,]+)\s*(?:to|–|-)\s*([-\d\.,]+)\b", cap, flags=re.IGNORECASE)
    if m:
        info.y_min = m.group(1).replace(",", "")
        info.y_max = m.group(2).replace(",", "")

    return info


def chart_class_from_type(chart_type: str) -> str:
    t = (chart_type or "").lower()
    if "line" in t:
        return "schema:LineChart"
    if "pie" in t:
        return "schema:PieChart"
    if "scatter" in t:
        return "schema:ScatterPlot"
    return "schema:BarChart"


def parse_datatable(
    datatable: str,
    x_label_hint: str = "",
    y_label_hint: str = "",
) -> Tuple[str, str, str, List[Tuple[str, str]]]:
    """
    Parse datatable into:
      (title, x_label, y_label, [(x_val, y_val), ...])

    Key fix:
    Uses y_label_hint (from caption_L1) to locate where the header ends,
    so the FIRST ROW is not swallowed in one-line datatables.
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


class JSONToTTLConverter:
    """Converts JSON chart data to TTL using ONLY datatable + caption_L1."""

    def __init__(
        self,
        base_uri: str = "http://example.org/chart-data/{img_id}/",
        truncate_point_label_len: int = 35,
        truncate_title_len: int = 120,
        sort_points: bool = True,
    ):
        self.base_uri_template = base_uri
        self.truncate_point_label_len = truncate_point_label_len
        self.truncate_title_len = truncate_title_len
        self.sort_points = sort_points

    def convert_json_to_ttl(self, json_data: Dict[str, Any]) -> str:
        img_id = str(json_data.get("img_id", "unknown"))

        datatable = json_data.get("datatable", "") or ""
        caption_l1 = json_data.get("caption_L1", "") or ""

        cap = parse_caption_l1(caption_l1)
        dt_title, dt_x, dt_y, points = parse_datatable(
            datatable,
            x_label_hint=cap.x_label,
            y_label_hint=cap.y_label,
        )

        title = dt_title or cap.title or f"Chart {img_id}"
        x_label = cap.x_label or dt_x or "x"
        y_label = cap.y_label or dt_y or "y"
        chart_class = chart_class_from_type(cap.chart_type)

        # Dynamic names like your example: :Response, :shareOfRespondents
        point_class = pascal_case(x_label, default="Category")
        value_prop = lower_camel_case(y_label, default="value")

        base_uri = self.base_uri_template.format(img_id=img_id)
        if not base_uri.endswith(("/", "#")):
            base_uri = base_uri.rstrip("/") + "/"

        ttl_lines: List[str] = [
            f"@prefix : <{base_uri}> .",
            "@prefix xsd: <http://www.w3.org/2001/XMLSchema#> .",
            "@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .",
            "@prefix schema: <https://schema.org/> .",
            "",
            f":Chart1 a {chart_class} ;",
            f'    schema:title "{escape_ttl_string(truncate_middle(title, self.truncate_title_len))}" ;',
            f'    :hasXAxisLabel "{escape_ttl_string(x_label)}" ;',
            f'    :hasYAxisLabel "{escape_ttl_string(y_label)}" .',
            "",
        ]

        # Sort to match your sample style (alphabetical by category)
        if self.sort_points:
            points = sorted(points, key=lambda p: p[0].lower())

        for i, (x_val, y_val_raw) in enumerate(points, start=1):
            label = truncate_end(x_val, self.truncate_point_label_len)
            y_val, is_num = parse_numeric_token(y_val_raw)

            ttl_lines.append(f":DataPoint{i} a :{point_class} ;")
            ttl_lines.append(f'    rdfs:label "{escape_ttl_string(label)}" ;')

            if is_num:
                ttl_lines.append(f'    :{value_prop} "{y_val}"^^xsd:decimal ;')
            else:
                ttl_lines.append(f'    :{value_prop} "{escape_ttl_string(str(y_val_raw))}"^^xsd:string ;')

            ttl_lines.append("    :belongsToChart :Chart1 .")
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

        for json_file in json_files:
            input_path = os.path.join(input_dir, json_file)
            output_filename = os.path.splitext(json_file)[0] + ".ttl"
            output_path = os.path.join(output_dir, output_filename)

            try:
                self.convert_file(input_path, output_path)
                print(f"✓ Converted: {json_file} -> {output_filename}")
            except Exception as e:
                print(f"✗ Error converting {json_file}: {e}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Convert JSON chart data to TTL (datatable + caption_L1 only)")
    parser.add_argument("input", help="Input JSON file or a directory containing JSON files")
    parser.add_argument("-o", "--output", help="Output TTL file (if input is a file) OR output directory (if input is a directory)")
    parser.add_argument(
        "--base-uri",
        default="http://example.org/chart-data/{img_id}/",
        help="Base URI for the ':' prefix. You can use {img_id} as a placeholder.",
    )
    parser.add_argument("--no-sort", action="store_true", help="Do not alphabetically sort data points")
    parser.add_argument("--truncate-label", type=int, default=35, help="Max chars for each rdfs:label before '...'")
    parser.add_argument("--truncate-title", type=int, default=120, help="Max chars for schema:title before middle '...'")

    args = parser.parse_args()

    converter = JSONToTTLConverter(
        base_uri=args.base_uri,
        truncate_point_label_len=args.truncate_label,
        truncate_title_len=args.truncate_title,
        sort_points=not args.no_sort,
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
