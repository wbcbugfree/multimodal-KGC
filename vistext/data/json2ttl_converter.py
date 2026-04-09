#!/usr/bin/env python3
"""
JSON -> TTL (Turtle) converter for VisText ground-truth generation.

This remains a datatable-first converter:
- `L1_properties` provide chart metadata
- `datatable` remains the canonical source for data values
- `scenegraph` is used conservatively to protect categorical labels from
  token splitting, repair clearly damaged bar labels, and provide a
  suspicious fallback when bar-chart datatable parsing fails

The converter intentionally avoids scenegraph-driven interpolation except for
previously missing bar-chart outputs that cannot be parsed from `datatable`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from difflib import SequenceMatcher
from typing import Any, Dict, List, Optional, Tuple

FLOAT_PATTERN = r"-?\d+(?:\.\d+)?"

def escape_ttl_string(text: str) -> str:
    """Escape special characters for TTL string literals."""
    text = text.replace("\\", "\\\\")
    text = text.replace('"', '\\"')
    text = text.replace("\n", "\\n")
    text = text.replace("\r", "\\r")
    text = text.replace("\t", "\\t")
    return text


def single_token_string(text: str) -> str:
    """Protect common multi-token labels by replacing spaces with @."""

    months = (
        r"(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|"
        r"April|June|July|August|September|October|November|December|Septembre)"
    )

    patterns = [
        r"[A-Z][a-z]{2} \d{1,2}, \d{4}",
        r"FY \d{4}",
        r"Q[1-4] \d{4}\*?",
        r"\d{4} / \d{4}\*?",
        r"\d+ to \d+ years",
        r"\d+ years and more",
        r"\d+ years and older",
        r'\d+ years or older',
        r'\d+ years or younger',
        r'Younger than \d+ years',
        r'Younger than \d+',
        r'\d+ to \d+ people',
        r'Greater than \d+ people',
        r'Less than \d+ people',
        r'{months} \d{{4}}'.format(months=months),
        r'H\d{1} \d{4}',
        r'{months} \d{{2,4}} - {months} \d{{2,4}}'.format(months=months),
        r'{months} \d{{2,4}}-{months} \d{{2,4}}'.format(months=months),
        r'\d{4} \*',
        r'\d{2} \*',
    ]

    combined_pattern = '|'.join(f'(?:{p})' for p in patterns)
    return re.sub(combined_pattern, lambda match: match.group().replace(' ', '@'), text)


def protect_label_phrases(text: str, phrases: List[str]) -> str:
    """Protect exact multi-token categorical labels from whitespace tokenization."""
    protected = text
    cleaned_phrases = {
        " ".join(str(phrase).split()).strip()
        for phrase in phrases
        if phrase and "none val" not in str(phrase).lower()
    }
    for phrase in sorted(cleaned_phrases, key=len, reverse=True):
        if " " not in phrase:
            continue
        replacement = phrase.replace(" ", "@")
        protected = re.sub(rf"(?<!\S){re.escape(phrase)}(?!\S)", replacement, protected)
    return protected


def strip_datatable_header_prefix(text: str, x_label_hint: str, y_label_hint: str) -> str:
    """Remove an exact leading header prefix such as `x_label y_label`."""
    compact = " ".join(text.split())
    candidates = [
        " ".join(filter(None, [x_label_hint.strip(), y_label_hint.strip()])),
        " ".join(filter(None, [y_label_hint.strip(), x_label_hint.strip()])),
    ]
    for candidate in sorted({candidate for candidate in candidates if candidate}, key=len, reverse=True):
        if compact == candidate:
            return ""
        if compact.startswith(candidate + " "):
            return compact[len(candidate) + 1 :]
    return compact


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


@dataclass
class Tick:
    coord: float
    label: str
    numeric_value: Optional[float]


@dataclass
class Mark:
    x: float
    y: float
    width: Optional[float] = None
    height: Optional[float] = None


@dataclass
class ScenegraphData:
    mark_type: str
    x_ticks: List[Tick]
    y_ticks: List[Tick]
    marks: List[Mark]


@dataclass
class ConversionResult:
    ttl: str
    status: str
    reasons: List[str] = field(default_factory=list)
    repair_actions: List[str] = field(default_factory=list)

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


def _parse_numeric_label(label: str) -> Optional[float]:
    value, is_numeric = parse_numeric_token(label)
    if not is_numeric:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_tick_section(section: str, coord_name: str) -> List[Tick]:
    pattern = re.compile(
        rf"{coord_name}\s+({FLOAT_PATTERN})\s+val\s+(.*?)(?=(?:\s+{coord_name}\s+{FLOAT_PATTERN}\s+val\s+)|$)",
        re.S,
    )
    ticks: List[Tick] = []
    for coord_text, label_text in pattern.findall(section):
        label = " ".join(label_text.split()).strip()
        if not label or "none val" in label.lower():
            continue
        ticks.append(
            Tick(
                coord=float(coord_text),
                label=label,
                numeric_value=_parse_numeric_label(label),
            )
        )
    return ticks


def _parse_marks(section: str) -> Tuple[str, List[Mark]]:
    match = re.match(rf"\s*(bar|line|area)\b(.*)$", section, re.S)
    if not match:
        raise ValueError("Missing or unsupported scenegraph mark type")

    mark_type = match.group(1)
    rest = match.group(2)
    marks: List[Mark] = []

    if mark_type == "bar":
        pattern = re.compile(
            rf"XY\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+width\s+({FLOAT_PATTERN})\s+H\s+({FLOAT_PATTERN})\s+desc"
        )
        for x_text, y_text, width_text, height_text in pattern.findall(rest):
            marks.append(
                Mark(
                    x=float(x_text),
                    y=float(y_text),
                    width=float(width_text),
                    height=float(height_text),
                )
            )
    elif mark_type == "line":
        pattern = re.compile(rf"XY\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+desc")
        for x_text, y_text in pattern.findall(rest):
            marks.append(Mark(x=float(x_text), y=float(y_text)))
    else:
        pattern = re.compile(rf"XY\s+({FLOAT_PATTERN})\s+({FLOAT_PATTERN})\s+H\s+({FLOAT_PATTERN})\s+desc")
        for x_text, y_text, height_text in pattern.findall(rest):
            marks.append(Mark(x=float(x_text), y=float(y_text), height=float(height_text)))

    if not marks:
        raise ValueError(f"No marks parsed for {mark_type} scenegraph")
    return mark_type, marks


def parse_scenegraph(scenegraph: str) -> ScenegraphData:
    if not scenegraph:
        raise ValueError("Missing scenegraph")

    parts = re.split(r"\b(xtick|ytick|marks)\b", scenegraph)
    if len(parts) < 7:
        raise ValueError("Scenegraph missing xtick/ytick/marks sections")

    x_ticks = _parse_tick_section(parts[2], "x")
    y_ticks = _parse_tick_section(parts[4], "y")
    mark_type, marks = _parse_marks(parts[6])
    if not x_ticks or not y_ticks:
        raise ValueError("Scenegraph tick parsing failed")
    return ScenegraphData(mark_type=mark_type, x_ticks=x_ticks, y_ticks=y_ticks, marks=marks)


def _axis_kind(ticks: List[Tick]) -> str:
    if not ticks:
        return "unknown"
    numeric = sum(tick.numeric_value is not None for tick in ticks)
    return "numeric" if numeric / len(ticks) >= 0.7 else "categorical"


def _normalize_label(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text).lower())


def _scenegraph_categorical_labels(scenegraph: ScenegraphData) -> List[str]:
    labels: List[str] = []
    seen: set[str] = set()
    if _axis_kind(scenegraph.x_ticks) == "categorical":
        for tick in scenegraph.x_ticks:
            normalized = _normalize_label(tick.label)
            if normalized and normalized not in seen:
                seen.add(normalized)
                labels.append(tick.label)
    if _axis_kind(scenegraph.y_ticks) == "categorical":
        for tick in scenegraph.y_ticks:
            normalized = _normalize_label(tick.label)
            if normalized and normalized not in seen:
                seen.add(normalized)
                labels.append(tick.label)
    return labels


def _numeric_ticks(ticks: List[Tick]) -> List[Tick]:
    return [tick for tick in ticks if tick.numeric_value is not None]


def _closest_to_zero_coord(ticks: List[Tick]) -> Optional[float]:
    numeric_ticks = _numeric_ticks(ticks)
    if not numeric_ticks:
        return None
    zero_tick = min(numeric_ticks, key=lambda tick: abs(float(tick.numeric_value or 0.0)))
    return zero_tick.coord


def _interpolate_numeric(ticks: List[Tick], coord: float) -> float:
    numeric_ticks = _numeric_ticks(ticks)
    if len(numeric_ticks) < 2:
        raise ValueError("Need at least two numeric ticks for interpolation")

    first = numeric_ticks[0]
    last = numeric_ticks[-1]
    if first.coord == last.coord:
        return float(first.numeric_value or 0.0)
    return float(first.numeric_value or 0.0) + (coord - first.coord) * (
        (float(last.numeric_value or 0.0) - float(first.numeric_value or 0.0))
        / (last.coord - first.coord)
    )


def _format_numeric(value: float) -> str:
    quantized = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    text = format(quantized, "f").rstrip("0").rstrip(".")
    return text if text else "0"


def _nearest_categorical_label(ticks: List[Tick], coord: float) -> str:
    return min(ticks, key=lambda tick: abs(tick.coord - coord)).label


def _bar_orientation(marks: List[Mark]) -> str:
    avg_width = sum(abs(mark.width or 0.0) for mark in marks) / len(marks)
    avg_height = sum(abs(mark.height or 0.0) for mark in marks) / len(marks)
    return "horizontal" if avg_width >= avg_height else "vertical"


def _scenegraph_axis_value(ticks: List[Tick], axis_kind: str, coord: float) -> str:
    if axis_kind == "numeric":
        return _format_numeric(_interpolate_numeric(ticks, coord))
    return _nearest_categorical_label(ticks, coord)


def _scenegraph_bar_pairs(scenegraph: ScenegraphData) -> List[Tuple[str, str]]:
    zero_x = _closest_to_zero_coord(scenegraph.x_ticks)
    zero_y = _closest_to_zero_coord(scenegraph.y_ticks)
    orientation = _bar_orientation(scenegraph.marks)
    x_axis_kind = _axis_kind(scenegraph.x_ticks)
    y_axis_kind = _axis_kind(scenegraph.y_ticks)

    point_pairs: List[Tuple[str, str]] = []
    for mark in scenegraph.marks:
        if orientation == "vertical":
            x_coord = mark.x + (mark.width or 0.0) / 2.0
            x_value = _scenegraph_axis_value(scenegraph.x_ticks, x_axis_kind, x_coord)

            top_edge = mark.y
            bottom_edge = mark.y + (mark.height or 0.0)
            baseline = zero_y if zero_y is not None else max(top_edge, bottom_edge)
            value_coord = top_edge if abs(top_edge - baseline) >= abs(bottom_edge - baseline) else bottom_edge
            y_coord = mark.y + (mark.height or 0.0) / 2.0 if y_axis_kind != "numeric" else value_coord
            y_value = _scenegraph_axis_value(scenegraph.y_ticks, y_axis_kind, y_coord)
        else:
            left_edge = mark.x
            right_edge = mark.x + (mark.width or 0.0)
            baseline = zero_x if zero_x is not None else min(left_edge, right_edge)
            value_coord = left_edge if abs(left_edge - baseline) >= abs(right_edge - baseline) else right_edge
            x_coord = mark.x + (mark.width or 0.0) / 2.0 if x_axis_kind != "numeric" else value_coord
            x_value = _scenegraph_axis_value(scenegraph.x_ticks, x_axis_kind, x_coord)

            y_coord = mark.y + (mark.height or 0.0) / 2.0
            y_value = _scenegraph_axis_value(scenegraph.y_ticks, y_axis_kind, y_coord)

        point_pairs.append((str(x_value), str(y_value)))
    return point_pairs


def parse_datatable(
    datatable: str,
    x_label_hint: str = "",
    y_label_hint: str = "",
    protected_label_phrases: Optional[List[str]] = None,
) -> Tuple[str, str, str, List[Tuple[str, str]]]:
    """Parse datatable into (title, x_label, y_label, points)."""
    if not datatable:
        return "", "", "", []

    if "<s>" in datatable:
        title_part, data_part = datatable.split("<s>", 1)
        title = " ".join(title_part.strip().split())
        data_section = data_part.strip()
    else:
        title = ""
        data_section = datatable.strip()

    lines = [ln.strip() for ln in data_section.splitlines() if ln.strip()]
    if len(lines) > 1:
        data_section = " ".join(lines[1:])

    data_section = strip_datatable_header_prefix(data_section, x_label_hint, y_label_hint)

    if protected_label_phrases:
        data_section = protect_label_phrases(data_section, protected_label_phrases)
    data_section = single_token_string(data_section)

    data_section = " ".join(data_section.split())
    tokens = data_section.split()
    if not tokens:
        return title, x_label_hint.strip(), y_label_hint.strip(), []

    norm_tokens = [_norm_token(t) for t in tokens]

    x_label = x_label_hint.strip()
    y_label = y_label_hint.strip()
    data_start = 0

    y_seq = [_norm_token(t) for t in y_label.split()]
    idx = _find_subsequence(norm_tokens, y_seq)
    if idx is not None:
        prefix_tokens = tokens[:idx]
        prefix_has_numeric = any(parse_numeric_token(token)[1] for token in prefix_tokens)
        if not prefix_has_numeric:
            data_start = idx + len(y_seq)

    if data_start >= len(tokens):
        return title, x_label, y_label, []

    label_types = [parse_numeric_token(t)[1] for t in tokens[data_start:]]
    if label_types.count(True) / len(label_types) > 0.5:
        year_value_pattern = check_year_value_pattern(tokens[data_start:])
        value_year_pattern = check_value_year_pattern(tokens[data_start:])

        if year_value_pattern or value_year_pattern:
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
    
    points = []
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


def _point_pairs_equal(left: List[Tuple[str, str]], right: List[Tuple[str, str]]) -> bool:
    return [(str(x), str(y)) for x, y in left] == [(str(x), str(y)) for x, y in right]


def _categorical_point_index(scenegraph: ScenegraphData) -> Optional[int]:
    x_axis_kind = _axis_kind(scenegraph.x_ticks)
    y_axis_kind = _axis_kind(scenegraph.y_ticks)
    if x_axis_kind == "categorical" and y_axis_kind == "numeric":
        return 0
    if y_axis_kind == "categorical" and x_axis_kind == "numeric":
        return 1
    if x_axis_kind == "categorical":
        return 0
    if y_axis_kind == "categorical":
        return 1
    return None


def _categorical_target_labels(scenegraph: ScenegraphData) -> List[str]:
    point_index = _categorical_point_index(scenegraph)
    if point_index == 0:
        ticks = scenegraph.x_ticks
    elif point_index == 1:
        ticks = scenegraph.y_ticks
    else:
        return []

    labels: List[str] = []
    seen: set[str] = set()
    for tick in ticks:
        normalized = _normalize_label(tick.label)
        if normalized and normalized not in seen:
            seen.add(normalized)
            labels.append(tick.label)
    return labels


def _label_similarity(left: str, right: str) -> float:
    norm_left = _normalize_label(left)
    norm_right = _normalize_label(right)
    if not norm_left or not norm_right:
        return 0.0
    if norm_left == norm_right:
        return 1.0
    if norm_right.endswith(norm_left) or norm_left.endswith(norm_right):
        return 0.995
    if norm_left in norm_right or norm_right in norm_left:
        return min(len(norm_left), len(norm_right)) / max(len(norm_left), len(norm_right))
    return SequenceMatcher(None, norm_left, norm_right).ratio()


def _strip_axis_label_prefix(label: str, axis_label: str) -> str:
    if not axis_label or ":" not in label:
        return label
    prefix, remainder = label.split(":", 1)
    if _normalize_label(prefix) == _normalize_label(axis_label):
        return remainder.strip()
    return label


def _strip_axis_prefix_from_points(
    points: List[Tuple[str, str]],
    scenegraph: ScenegraphData,
    x_label: str,
    y_label: str,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    point_index = _categorical_point_index(scenegraph)
    if point_index is None:
        return points, []

    axis_label = x_label if point_index == 0 else y_label
    changed = False
    cleaned_points: List[Tuple[str, str]] = []
    for x_value, y_value in points:
        if point_index == 0:
            cleaned_label = _strip_axis_label_prefix(str(x_value), axis_label)
            changed = changed or cleaned_label != str(x_value)
            cleaned_points.append((cleaned_label, y_value))
        else:
            cleaned_label = _strip_axis_label_prefix(str(y_value), axis_label)
            changed = changed or cleaned_label != str(y_value)
            cleaned_points.append((x_value, cleaned_label))
    return cleaned_points, (["categorical_axis_prefix_stripped"] if changed else [])


def _match_labels_bijectively(source_labels: List[str], target_labels: List[str]) -> Optional[List[str]]:
    if len(source_labels) != len(target_labels):
        return None

    target_norms = [_normalize_label(label) for label in target_labels]
    if len(set(target_norms)) != len(target_norms):
        return None

    assignments: List[str] = []
    used_targets: set[int] = set()
    for source in source_labels:
        scored = sorted(
            (
                (_label_similarity(source, target), index, target)
                for index, target in enumerate(target_labels)
                if index not in used_targets
            ),
            reverse=True,
        )
        if not scored:
            return None
        best_score, best_index, best_target = scored[0]
        second_score = scored[1][0] if len(scored) > 1 else 0.0
        if best_score < 0.88:
            return None
        if best_score < 1.0 and (best_score - second_score) < 0.05:
            return None
        used_targets.add(best_index)
        assignments.append(best_target)
    return assignments if len(assignments) == len(source_labels) else None


def _repair_categorical_labels_from_scenegraph(
    points: List[Tuple[str, str]],
    scenegraph: ScenegraphData,
) -> Tuple[List[Tuple[str, str]], List[str]]:
    point_index = _categorical_point_index(scenegraph)
    if point_index is None or not points:
        return points, []

    target_labels = _categorical_target_labels(scenegraph)
    if len(target_labels) != len(points):
        return points, []

    source_labels = [str(point[point_index]) for point in points]
    if [_normalize_label(label) for label in source_labels] == [_normalize_label(label) for label in target_labels]:
        return points, []

    matched_labels = _match_labels_bijectively(source_labels, target_labels)
    if matched_labels is None:
        return points, []

    repaired_points: List[Tuple[str, str]] = []
    for point, repaired_label in zip(points, matched_labels):
        if point_index == 0:
            repaired_points.append((repaired_label, point[1]))
        else:
            repaired_points.append((point[0], repaired_label))
    return repaired_points, ["scenegraph_fuzzy_categorical_label_repair"]


def _bar_suspicion_reasons(points: List[Tuple[str, str]], scenegraph: Optional[ScenegraphData]) -> List[str]:
    if scenegraph is None or scenegraph.mark_type != "bar":
        return []

    reasons: List[str] = []
    if len(points) != len(scenegraph.marks):
        reasons.append(
            f"bar_point_count_mismatch: datatable points={len(points)} scenegraph bars={len(scenegraph.marks)}"
        )

    point_index = _categorical_point_index(scenegraph)
    target_labels = _categorical_target_labels(scenegraph)
    if point_index is not None and target_labels and len(target_labels) == len(points):
        source_labels = [str(point[point_index]) for point in points]
        source_norms = {_normalize_label(label) for label in source_labels if _normalize_label(label)}
        target_norms = {_normalize_label(label) for label in target_labels if _normalize_label(label)}
        overlap = len(source_norms & target_norms)
        min_size = min(len(source_norms), len(target_norms))
        if min_size and (overlap / min_size) < 0.6:
            reasons.append("bar_categorical_labels_do_not_align_with_scenegraph_inventory")
        if any("none val" in label.lower() for label in source_labels):
            reasons.append("categorical_label_contains_none_val_artifact")
    return reasons


class JSONToTTLConverter:
    """Datatable-first VisText converter with conservative scenegraph-assisted repairs."""

    def __init__(
        self,
        base_uri: str = "http://example.org/vistext#",
        sort_points: bool = False,
    ):
        self.base_uri_template = base_uri
        self.sort_points = sort_points

    def _build_ttl(
        self,
        chart_type: str,
        title: str,
        x_label: str,
        y_label: str,
        points: List[Tuple[str, str]],
    ) -> str:
        ttl_lines: List[str] = [
            "@prefix : <http://example.org/vistext#> .",
            "",
            f":Chart a :{chart_type} ;",
        ]
        if title:
            ttl_lines.append(f'    :title "{escape_ttl_string(title)}" ;')
        ttl_lines.append("    :xAxis :XAxis ;")
        ttl_lines.append("    :yAxis :YAxis .")
        ttl_lines.append("")

        ttl_lines.append(":XAxis a :Axis ;")
        ttl_lines.append(f'    :title "{escape_ttl_string(x_label)}" .' if x_label else '    :title "" .')
        ttl_lines.append("")

        ttl_lines.append(":YAxis a :Axis ;")
        ttl_lines.append(f'    :title "{escape_ttl_string(y_label)}" .' if y_label else '    :title "" .')
        ttl_lines.append("")

        if self.sort_points:
            points = sorted(points, key=lambda point: str(point[0]).lower())

        for i, (x, y) in enumerate(points, start=1):
            ttl_lines.append(f":DataPoint{i} a :DataPoint ;")

            x_value, is_num = parse_numeric_token(str(x))
            if is_num:
                ttl_lines.append(f'    :xValue "{x_value}" ;')
            else:
                ttl_lines.append(f'    :xValue "{escape_ttl_string(str(x))}" ;')

            y_value, is_num = parse_numeric_token(str(y))
            if is_num:
                ttl_lines.append(f'    :yValue "{y_value}" ;')
            else:
                ttl_lines.append(f'    :yValue "{escape_ttl_string(str(y))}" ;')

            ttl_lines.append('    :belongsTo :Chart .')
            ttl_lines.append("")

        return "\n".join(ttl_lines).rstrip() + "\n"

    def convert_json_to_result(self, json_data: Dict[str, Any]) -> ConversionResult:
        if not isinstance(json_data, dict):
            raise ValueError(f"Expected dict, got {type(json_data).__name__}")

        img_id = str(json_data.get("img_id", "unknown"))
        datatable = json_data.get("datatable", "") or ""
        l1_properties = json_data.get("L1_properties") or []
        if not datatable or not l1_properties or len(l1_properties) < 5:
            raise ValueError(f"Missing datatable or L1_properties in img_id {img_id}, correct before rerunning")

        properties = parse_L1_properties(l1_properties)
        missing = [
            key
            for key, value in {
                "chart_type": properties.chart_type,
                "title": properties.title,
                "x_label": properties.x_label,
                "y_label": properties.y_label,
            }.items()
            if not value
        ]
        if missing:
            raise ValueError(f"Missing fields {missing} in img_id {img_id}: {properties}, correct before rerunning")

        chart_type = chart_type_string(properties.chart_type)
        scenegraph = None
        scenegraph_error = None
        scenegraph_label_phrases: List[str] = []
        raw_scenegraph = json_data.get("scenegraph", "") or ""
        if raw_scenegraph:
            try:
                scenegraph = parse_scenegraph(raw_scenegraph)
                scenegraph_label_phrases = _scenegraph_categorical_labels(scenegraph)
            except Exception as exc:
                scenegraph_error = str(exc)

        plain_parse_error = None
        protected_parse_error = None
        plain_points: Optional[List[Tuple[str, str]]] = None
        protected_points: Optional[List[Tuple[str, str]]] = None

        try:
            _, _, _, plain_points = parse_datatable(
                datatable,
                x_label_hint=properties.x_label,
                y_label_hint=properties.y_label,
            )
        except Exception as exc:
            plain_parse_error = str(exc)

        if scenegraph_label_phrases:
            try:
                _, _, _, protected_points = parse_datatable(
                    datatable,
                    x_label_hint=properties.x_label,
                    y_label_hint=properties.y_label,
                    protected_label_phrases=scenegraph_label_phrases,
                )
            except Exception as exc:
                protected_parse_error = str(exc)

        repair_actions: List[str] = []
        reasons: List[str] = []

        if protected_points is not None and (plain_points is None or not _point_pairs_equal(plain_points, protected_points)):
            points = protected_points
            repair_actions.append("scenegraph_exact_label_token_protection")
        elif plain_points is not None:
            points = plain_points
        elif protected_points is not None:
            points = protected_points
            repair_actions.append("scenegraph_exact_label_token_protection")
        elif chart_type == "BarChart" and scenegraph is not None and scenegraph.mark_type == "bar":
            points = _scenegraph_bar_pairs(scenegraph)
            reasons.append("datatable_parse_failed")
            reasons.append(f"datatable_error: {plain_parse_error or protected_parse_error or 'unknown'}")
            reasons.append("used_scenegraph_bar_fallback")
        else:
            if scenegraph_error:
                raise ValueError(f"{plain_parse_error or 'Datatable parse failed'}; scenegraph unavailable: {scenegraph_error}")
            raise ValueError(plain_parse_error or protected_parse_error or f"Could not parse img_id {img_id}")

        if chart_type == "BarChart" and scenegraph is not None and scenegraph.mark_type == "bar":
            stripped_points, strip_repairs = _strip_axis_prefix_from_points(
                points,
                scenegraph,
                properties.x_label,
                properties.y_label,
            )
            if strip_repairs and not _point_pairs_equal(points, stripped_points):
                points = stripped_points
                repair_actions.extend(strip_repairs)

            repaired_points, extra_repairs = _repair_categorical_labels_from_scenegraph(points, scenegraph)
            if extra_repairs and not _point_pairs_equal(points, repaired_points):
                points = repaired_points
                repair_actions.extend(extra_repairs)
            reasons.extend(_bar_suspicion_reasons(points, scenegraph))
        elif scenegraph_error:
            reasons.append(f"scenegraph_unavailable_for_validation: {scenegraph_error}")

        status = "clean"
        if reasons:
            status = "suspicious"
        elif repair_actions:
            status = "repaired"

        ttl = self._build_ttl(
            chart_type=chart_type,
            title=properties.title,
            x_label=properties.x_label,
            y_label=properties.y_label,
            points=[(str(x), str(y)) for x, y in points],
        )
        return ConversionResult(ttl=ttl, status=status, reasons=reasons, repair_actions=repair_actions)

    def convert_json_to_ttl(self, json_data: Dict[str, Any]) -> str:
        return self.convert_json_to_result(json_data).ttl

    def convert_file(self, json_file_path: str, output_file_path: Optional[str] = None) -> str:
        with open(json_file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)

        ttl_content = self.convert_json_to_result(json_data).ttl

        if output_file_path is None:
            base_name = os.path.splitext(os.path.basename(json_file_path))[0]
            output_file_path = f"{base_name}.ttl"

        with open(output_file_path, "w", encoding="utf-8") as f:
            f.write(ttl_content)

        return output_file_path

    def _resolve_report_path(self, output_dir: str, report_path: Optional[str] = None) -> str:
        if report_path is not None:
            return report_path

        normalized_output_dir = os.path.normpath(output_dir)
        if os.path.basename(normalized_output_dir).lower() == "turtle":
            return os.path.join(os.path.dirname(normalized_output_dir), "exceptions_report.json")
        return os.path.join(output_dir, "exceptions_report.json")

    def convert_directory(
        self,
        input_dir: str,
        output_dir: Optional[str] = None,
        report_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if output_dir is None:
            output_dir = input_dir

        os.makedirs(output_dir, exist_ok=True)

        json_files = sorted(f for f in os.listdir(input_dir) if f.lower().endswith(".json"))
        summary = {"clean": 0, "repaired": 0, "suspicious": 0, "error": 0}
        items: List[Dict[str, Any]] = []
        error_details: List[Dict[str, str]] = []
        for json_file in json_files:
            input_path = os.path.join(input_dir, json_file)
            output_filename = os.path.splitext(json_file)[0] + ".ttl"
            output_path = os.path.join(output_dir, output_filename)
            img_id = os.path.splitext(json_file)[0]

            try:
                with open(input_path, "r", encoding="utf-8") as f:
                    json_data = json.load(f)
                result = self.convert_json_to_result(json_data)
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(result.ttl)
                summary[result.status] += 1
                items.append(
                    {
                        "img_id": str(json_data.get("img_id", img_id)),
                        "status": result.status,
                        "ttl_file": output_filename,
                        "reasons": result.reasons,
                        "repair_actions": result.repair_actions,
                    }
                )
            except Exception as e:
                safe_error = str(e).encode("ascii", "backslashreplace").decode()
                print(f"Error converting {json_file}: {safe_error}")
                summary["error"] += 1
                if os.path.exists(output_path):
                    os.remove(output_path)
                error_details.append(
                    {
                        "file": json_file,
                        "error": str(e),
                    }
                )
                items.append(
                    {
                        "img_id": img_id,
                        "status": "error",
                        "ttl_file": None,
                        "reasons": [str(e)],
                        "repair_actions": [],
                    }
                )

        report = {
            "input_dir": input_dir,
            "output_dir": output_dir,
            "summary": summary,
            "items": items,
            "error_details": error_details,
        }
        report_path = self._resolve_report_path(output_dir=output_dir, report_path=report_path)
        os.makedirs(os.path.dirname(report_path), exist_ok=True)
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, ensure_ascii=False, indent=2)

        converted = summary["clean"] + summary["repaired"] + summary["suspicious"]
        print(
            f"Converted {converted} files ({summary['clean']} clean, {summary['repaired']} repaired, "
            f"{summary['suspicious']} suspicious), {summary['error']} errors"
        )
        print(f"Exception report written to: {report_path}")
        return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VisText JSON chart data to Turtle ground truth."
    )
    parser.add_argument("input", help="Input JSON file or a directory containing JSON files")
    parser.add_argument("-o", "--output", help="Output TTL file (if input is a file) OR output directory (if input is a directory)")
    parser.add_argument("--report-path", help="Optional exception report path when converting a directory")
    parser.add_argument("--sort", action="store_true", help="Alphabetically sort data points by category")

    args = parser.parse_args()

    converter = JSONToTTLConverter(
        sort_points=args.sort,
    )

    if os.path.isfile(args.input):
        out_path = converter.convert_file(args.input, args.output)
        print(f"Successfully converted to: {out_path}")
        return 0

    if os.path.isdir(args.input):
        converter.convert_directory(args.input, args.output, args.report_path)
        print("\nConversion complete!")
        return 0

    print(f"Error: {args.input} is not a valid file or directory")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
