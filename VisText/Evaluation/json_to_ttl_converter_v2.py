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

# TODO: Handle FY Years ex. 4985.

def escape_ttl_string(text: str) -> str:
    """Escape special characters for TTL string literals."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return text


def single_token_string(date_str: str) -> str:
    """Replace spaces in date strings with @ to keep them as single tokens 
    (e.g. 'Dec 31, 2020' → 'Dec@31,@2020', 'FY 2017' → 'FY@2017', 'Q1 2020' → 'Q1@2020')."""
    
    months = r'(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December|Septembre)'
    
    patterns = [
        r'[A-Z][a-z]{2} \d{1,2}, \d{4}',     # e.g. Dec 31, 2020
        r'FY \d{4}',                         # e.g. FY 2017
        r'Q[1-4] \d{4}\*?',                  # e.g. Q1 2020 or Q2 2023*
        r'\d{4} / \d{4}\*?',                 # e.g. 2019 / 2020 or 2019 / 2020*
        r'\d+ to \d+ years',                 # e.g. 1 to 5 years
        r'\d+ years and more',                # e.g. 75 years and more
        r'\d+ years or older',
        r'\d+ years or younger',
        r'Younger than \d+ years',              # e.g. Younger than 18 years
        r'Younger than \d+',              # e.g. Younger than 18
        r'\d+ to \d+ people',
        r'Greater than \d+ people',
        r'Less than \d+ people',
        r'{months} \d{{4}}'.format(months=months),  # e.g. January 2020
        # r'Mar \d{2,4}',                         # e.g. Mar 2020
        r'H\d{1} \d{4}',                            # e.g. H1 2020
        r'{months} \d{{2,4}} - {months} \d{{2,4}}'.format(months=months),  # e.g. January 2020 - March 2020
        r'{months} \d{{2,4}}-{months} \d{{2,4}}'.format(months=months),
        r'\d{4} \*', # Eg. 2013 *,
        r'\d{2} \*'
        
    ]
    
    combined_pattern = '|'.join(f'(?:{p})' for p in patterns)
    return re.sub(combined_pattern, lambda m: m.group().replace(' ', '@'), date_str)


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
    t = t.replace("−", "-")  # Handle unicode minus sign
    t_no_pct = t.replace("%", "")

    try:
        float(t_no_pct)
        return t_no_pct, True
    except ValueError:
        return token, False

def is_year(t: str) -> bool:
    if re.fullmatch(r"\d{4}", t):
        t_int = int(t)
        return 1900 <= t_int <= 2100
    
    if re.fullmatch(r"\d{4}\/\d{4}", t):
        parts = t.split("/")
        if len(parts) == 2 and all(re.fullmatch(r"\d{4}", p) for p in parts) and all(1900 <= int(p) <= 2100 for p in parts):
            return True
    return False

def check_year_value_pattern(tokens: List[str]) -> bool:
    """Check if tokens 0 and 2 looks like a 4-digit year, removing all commas and *, and tokens 1 and 3 look like numeric values, which would suggest a pattern of Year Value Year Value that we can parse."""
    if len(tokens) < 4:
        return False
    
    year1 = re.sub(r"[,*]", "", tokens[0])
    year2 = re.sub(r"[,*]", "", tokens[2])
    
    if is_year(year1) and is_year(year2):
        val1, is_num1 = parse_numeric_token(tokens[1])
        val2, is_num2 = parse_numeric_token(tokens[3])
        
        # Also check that the values are not years themselves, to avoid confusion with patterns like Year Year Year Year
        if is_num1 and is_num2:
            val1_no_pct = val1.replace("*", "")
            val2_no_pct = val2.replace("*", "")
            if not (is_year(val1_no_pct) or is_year(val2_no_pct)):
                return True
    return False 

def check_value_year_pattern(tokens: List[str]) -> bool:
    """Check if tokens 1 and 3 looks like a 4-digit year, removing all commas and *, and tokens 0 and 2 look like numeric values, which would suggest a pattern of Value Year Value Year that we can parse."""
    if len(tokens) < 4:
        return False
    
    year1 = re.sub(r"[,*]", "", tokens[1])
    year2 = re.sub(r"[,*]", "", tokens[3])
    
    if is_year(year1) and is_year(year2):
        val1, is_num1 = parse_numeric_token(tokens[0])
        val2, is_num2 = parse_numeric_token(tokens[2])
        # Also check that the values are not years themselves, to avoid confusion with patterns like Year Year Year Year
        if is_num1 and is_num2:
                val1_no_pct = val1.replace("*", "")
                val2_no_pct = val2.replace("*", "")
                
                if not (is_year(val1_no_pct) or is_year(val2_no_pct)):
                    return True
                else:
                    print("Failed last if")
    return False

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

def chart_type_string(chart_type: str) -> str:
    """Return chart type string for TTL."""
    t = (chart_type or "").lower()
    if "line" in t:
        return "LineChart"
    if "pie" in t:
        return "PieChart"
    if "scatter" in t:
        return "ScatterPlot"
    if "area" in t:
        return "AreaChart"
    return "BarChart"


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
        
    # Treat dates as single tokens by replacing spaces with @ (e.g. "Dec 31, 2020" → "Dec@31,@2020")
    data_section = single_token_string(data_section)
    
    # If the xlabel or ylabel are "year", they could get interpreted as independent variables instead of labels.
    # To prevent this, we add a @ after anything that looks like a 4-digit year


    # Multi-line: first line header, remaining rows. Join them with spaces to parse as one line.
    lines = [ln.strip() for ln in data_section.splitlines() if ln.strip()]
    if len(lines) > 1:
        data_section = " ".join(lines[1:])

    # One-line datatable
    data_section = " ".join(data_section.split())
    tokens = data_section.split()
    if not tokens:
        return title, x_label_hint.strip(), y_label_hint.strip(), []

    norm_tokens = [_norm_token(t) for t in tokens]

    x_label = x_label_hint.strip()
    y_label = y_label_hint.strip()
    data_start = 0

    #Find y_label_hint sequence inside tokens -> start data right after it.
    y_seq = [_norm_token(t) for t in y_label.split()]
    idx = _find_subsequence(norm_tokens, y_seq)
    if idx is not None:
        data_start = idx + len(y_seq)

    # Check if the x and y axes are both numeric, then it won't be possible to automatically parse.
    label_types = [parse_numeric_token(t)[1] for t in tokens[data_start:]]
    # % that are numeric, if more than 50, then raise error since it's likely that the labels are numeric and we can't parse the datatable.
    if label_types.count(True) / len(label_types) > 0.5:
        # Labels are mostly numeric in datatable, trying alternative parsing strategies...
        year_value_pattern = check_year_value_pattern(tokens[data_start:])
        value_year_pattern = check_value_year_pattern(tokens[data_start:])

        if year_value_pattern or value_year_pattern:
            # Caught Year Value / Value Year pattern...
            points: List[Tuple[str, str]] = []
            for i in range(data_start, len(tokens)-1, 2):
                x = tokens[i]
                y = tokens[i+1]
                points.append((x.replace("@", " "), y.replace("@", " ")))
            
            return title, x_label, y_label, points

        raise ValueError(f"Labels are mostly numeric in datatable: {tokens[data_start:data_start+10]}, cannot reliably parse data points, and no known patterns detected")

    
    first_token_after_y_label = tokens[data_start]    
    _, is_num = parse_numeric_token(first_token_after_y_label)
    if is_num:
        labels_first = False
    else:
        labels_first = True

    if labels_first:
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

        points = [(x.replace("@", " "), y.replace("@", " ")) for x, y in points]
        return title, x_label, y_label, points
    
    else:
        # Parse rows: (number) + (words...) repeating
        points: List[Tuple[str, str]] = []
        y_parts: List[str] = []
        
        current_value = None
        for tok in tokens[data_start:]:
            val, is_num = parse_numeric_token(tok)
            if is_num:
                if y_parts:
                    points.append((current_value, " ".join(y_parts)))
                    y_parts = []

                current_value = val
            else:
                y_parts.append(tok)
        if current_value is not None and y_parts:
            points.append((current_value, " ".join(y_parts)))

        points = [(x.replace("@", " "), y.replace("@", " ")) for x, y in points]
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
        l1_properties = json_data.get("L1_properties", "") or ""
        
        # If datatable or L1properties are empty, we can't parse the chart, so we raise an error to fix the data before conversion.
        if not datatable or not l1_properties or len(l1_properties) < 5:
            raise ValueError(f"Missing datatable or L1_properties in img_id {img_id}, correct before rerunning")
        
        properties = parse_L1_properties(l1_properties)
        
        # Raise error and continue to the next file if any of the properties is missing, since we rely on them to parse the datatable and get the data points.
        missing = [k for k, v in {
            "chart_type": properties.chart_type,
            "title": properties.title,
            "x_label": properties.x_label,
            "y_label": properties.y_label,
        }.items() if not v]

        if missing:
            raise ValueError(f"Missing fields {missing} in img_id {img_id}: {properties}, correct before rerunning")
        
        dt_title, dt_x, dt_y, points = parse_datatable(
            datatable,
            x_label_hint=properties.x_label,
            y_label_hint=properties.y_label,
        )

        title = properties.title or dt_title or ""
        x_label = properties.x_label or dt_x or ""
        y_label = properties.y_label or dt_y or ""
        chart_type = chart_type_string(properties.chart_type)

        # Build TTL matching the prediction schema
        ttl_lines: List[str] = [
            f"@prefix : <http://example.org/> .",
            "",
            f":Chart a :{chart_type} ;",
        ]
        
        if title:
            ttl_lines.append(f'    :title "{escape_ttl_string(truncate_middle(title, self.truncate_title_len))}" ;')
        
        ttl_lines.append(f"    :xAxis :XAxis ;")
        ttl_lines.append(f"    :yAxis :YAxis .")
        ttl_lines.append("")
        
        # X-axis
        ttl_lines.append(f":XAxis a :Axis ;")
        if x_label:
            ttl_lines.append(f'    :title "{escape_ttl_string(x_label)}" .')
        else:
            ttl_lines.append(f'    :title "" .')
        ttl_lines.append("")
        
        # Y-axis
        ttl_lines.append(f":YAxis a :Axis ;")
        if y_label:
            ttl_lines.append(f'    :title "{escape_ttl_string(y_label)}" .')
        else:
            ttl_lines.append(f'    :title "" .')
        ttl_lines.append("")

        # Sort if requested
        if self.sort_points:
            points = sorted(points, key=lambda p: p[0].lower())

        # Data points - using chart:category and chart:value (with #-based URIs)
        for i, (x, y) in enumerate(points, start=1):
            x_clean = truncate_end(str(x), self.truncate_point_label_len)
            y_clean = truncate_end(str(y), self.truncate_point_label_len)

            ttl_lines.append(f":DataPoint{i} a :DataPoint ;")
            
            value, is_num = parse_numeric_token(x_clean)
            if is_num:
                ttl_lines.append(f'    :xValue "{value}" ;')
            else:
                ttl_lines.append(f'    :xValue "{escape_ttl_string(x_clean)}" ;')
            
            value, is_num = parse_numeric_token(y_clean)
            if is_num:
                ttl_lines.append(f'    :yValue "{value}" ;')
            else:
                ttl_lines.append(f'    :yValue "{escape_ttl_string(y_clean)}" ;')
            
            ttl_lines.append(f'    :belongsTo :Chart .')
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
