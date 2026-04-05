#!/usr/bin/env python3
"""
JSON -> TTL (Turtle) converter using scenegraph-first extraction for plot data.

This converter keeps the existing VisText TTL schema, but changes the extraction
strategy:
- `L1_properties` provides chart type and axis/title metadata
- `scenegraph` provides tick labels and plotted mark geometry
- for line/area charts, exact point pairs fall back to `datatable` when available,
  because scenegraph geometry only provides approximate numeric values
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from typing import Any, Optional

from json_to_ttl_converter_v2 import (
    chart_type_string,
    escape_ttl_string,
    parse_L1_properties,
    parse_datatable,
    parse_numeric_token,
    truncate_end,
    truncate_middle,
)


FLOAT_PATTERN = r"-?\d+(?:\.\d+)?"
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "ground_truth_ttl_scenegraph"


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
    x_ticks: list[Tick]
    y_ticks: list[Tick]
    marks: list[Mark]


@dataclass
class ConversionResult:
    img_id: str
    ttl: str
    used_x_label_fallback: bool = False
    fallback_kind: Optional[str] = None
    notes: list[str] = field(default_factory=list)


def _parse_numeric_label(label: str) -> Optional[float]:
    value, is_numeric = parse_numeric_token(label)
    if not is_numeric:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _parse_tick_section(section: str, coord_name: str) -> list[Tick]:
    pattern = re.compile(
        rf"{coord_name}\s+({FLOAT_PATTERN})\s+val\s+(.*?)(?=(?:\s+{coord_name}\s+{FLOAT_PATTERN}\s+val\s+)|$)",
        re.S,
    )
    ticks: list[Tick] = []
    for coord_text, label_text in pattern.findall(section):
        label = " ".join(label_text.split()).strip()
        if not label:
            continue
        ticks.append(
            Tick(
                coord=float(coord_text),
                label=label,
                numeric_value=_parse_numeric_label(label),
            )
        )
    return ticks


def _parse_marks(section: str) -> tuple[str, list[Mark]]:
    match = re.match(rf"\s*(bar|line|area)\b(.*)$", section, re.S)
    if not match:
        raise ValueError("Missing or unsupported scenegraph mark type")

    mark_type = match.group(1)
    rest = match.group(2)
    marks: list[Mark] = []

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
            marks.append(
                Mark(
                    x=float(x_text),
                    y=float(y_text),
                    height=float(height_text),
                )
            )

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


def _axis_kind(ticks: list[Tick]) -> str:
    if not ticks:
        return "unknown"
    numeric = sum(t.numeric_value is not None for t in ticks)
    return "numeric" if numeric / len(ticks) >= 0.7 else "categorical"


def _numeric_ticks(ticks: list[Tick]) -> list[Tick]:
    return [tick for tick in ticks if tick.numeric_value is not None]


def _interpolate_numeric(ticks: list[Tick], coord: float) -> float:
    numeric_ticks = _numeric_ticks(ticks)
    if len(numeric_ticks) < 2:
        raise ValueError("Need at least two numeric ticks for interpolation")

    first = numeric_ticks[0]
    last = numeric_ticks[-1]
    if math.isclose(first.coord, last.coord):
        return float(first.numeric_value or 0.0)

    return float(first.numeric_value or 0.0) + (coord - first.coord) * (
        (float(last.numeric_value or 0.0) - float(first.numeric_value or 0.0))
        / (last.coord - first.coord)
    )


def _nearest_categorical_label(ticks: list[Tick], coord: float) -> str:
    return min(ticks, key=lambda tick: abs(tick.coord - coord)).label


def _closest_to_zero_coord(ticks: list[Tick]) -> Optional[float]:
    numeric_ticks = _numeric_ticks(ticks)
    if not numeric_ticks:
        return None
    zero_tick = min(numeric_ticks, key=lambda tick: abs(float(tick.numeric_value or 0.0)))
    return zero_tick.coord


def _bar_orientation(marks: list[Mark]) -> str:
    avg_width = sum(abs(mark.width or 0.0) for mark in marks) / len(marks)
    avg_height = sum(abs(mark.height or 0.0) for mark in marks) / len(marks)
    return "horizontal" if avg_width >= avg_height else "vertical"


def _decimal_places(reference: Optional[str]) -> Optional[int]:
    if reference is None:
        return None
    value, is_numeric = parse_numeric_token(reference)
    if not is_numeric:
        return None
    if "." not in value:
        return 0
    return len(value.split(".", 1)[1])


def _format_numeric(value: float, reference: Optional[str] = None) -> str:
    decimals = _decimal_places(reference)
    if decimals is None:
        quantized = Decimal(str(value)).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    else:
        pattern = "1" if decimals == 0 else "1." + ("0" * decimals)
        quantized = Decimal(str(value)).quantize(Decimal(pattern), rounding=ROUND_HALF_UP)

    text = format(quantized, "f").rstrip("0").rstrip(".")
    return text if text else "0"


def _reference_points(json_data: dict[str, Any], x_label: str, y_label: str) -> list[tuple[str, str]]:
    datatable = json_data.get("datatable", "") or ""
    if not datatable:
        return []
    try:
        _, _, _, points = parse_datatable(datatable, x_label_hint=x_label, y_label_hint=y_label)
    except Exception:
        return []
    return [(str(x), str(y)) for x, y in points]


class JSONToTTLConverterScenegraph:
    def __init__(
        self,
        truncate_point_label_len: int = 50,
        truncate_title_len: int = 150,
        output_dir: Optional[Path] = None,
    ):
        self.truncate_point_label_len = truncate_point_label_len
        self.truncate_title_len = truncate_title_len
        self.output_dir = output_dir or DEFAULT_OUTPUT_DIR

    def convert_json_to_result(self, json_data: dict[str, Any]) -> ConversionResult:
        if not isinstance(json_data, dict):
            raise ValueError(f"Expected dict, got {type(json_data).__name__}")

        img_id = str(json_data.get("img_id", "unknown"))
        l1_properties = json_data.get("L1_properties") or []
        if len(l1_properties) < 5:
            raise ValueError(f"Missing L1_properties in img_id {img_id}")

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
            raise ValueError(f"Missing fields {missing} in img_id {img_id}")

        scenegraph = parse_scenegraph(json_data.get("scenegraph", "") or "")
        chart_type = chart_type_string(properties.chart_type)
        reference_points = _reference_points(json_data, properties.x_label, properties.y_label)

        point_pairs: list[tuple[str, str]] = []
        used_x_label_fallback = False
        fallback_kind: Optional[str] = None
        notes: list[str] = []

        if scenegraph.mark_type == "bar":
            zero_x = _closest_to_zero_coord(scenegraph.x_ticks)
            zero_y = _closest_to_zero_coord(scenegraph.y_ticks)
            orientation = _bar_orientation(scenegraph.marks)

            for index, mark in enumerate(scenegraph.marks):
                ref_x = reference_points[index][0] if index < len(reference_points) else None
                ref_y = reference_points[index][1] if index < len(reference_points) else None

                if orientation == "vertical":
                    x_coord = mark.x + (mark.width or 0.0) / 2.0
                    x_value = (
                        _format_numeric(_interpolate_numeric(scenegraph.x_ticks, x_coord), ref_x)
                        if _axis_kind(scenegraph.x_ticks) == "numeric"
                        else truncate_end(
                            _nearest_categorical_label(scenegraph.x_ticks, x_coord),
                            self.truncate_point_label_len,
                        )
                    )

                    top_edge = mark.y
                    bottom_edge = mark.y + (mark.height or 0.0)
                    baseline = zero_y if zero_y is not None else max(top_edge, bottom_edge)
                    value_coord = top_edge if abs(top_edge - baseline) >= abs(bottom_edge - baseline) else bottom_edge
                    y_value = (
                        _format_numeric(_interpolate_numeric(scenegraph.y_ticks, value_coord), ref_y)
                        if _axis_kind(scenegraph.y_ticks) == "numeric"
                        else truncate_end(
                            _nearest_categorical_label(scenegraph.y_ticks, mark.y + (mark.height or 0.0) / 2.0),
                            self.truncate_point_label_len,
                        )
                    )
                else:
                    left_edge = mark.x
                    right_edge = mark.x + (mark.width or 0.0)
                    baseline = zero_x if zero_x is not None else min(left_edge, right_edge)
                    value_coord = left_edge if abs(left_edge - baseline) >= abs(right_edge - baseline) else right_edge
                    x_value = (
                        _format_numeric(_interpolate_numeric(scenegraph.x_ticks, value_coord), ref_x)
                        if _axis_kind(scenegraph.x_ticks) == "numeric"
                        else truncate_end(
                            _nearest_categorical_label(scenegraph.x_ticks, mark.x + (mark.width or 0.0) / 2.0),
                            self.truncate_point_label_len,
                        )
                    )

                    y_coord = mark.y + (mark.height or 0.0) / 2.0
                    y_value = (
                        _format_numeric(_interpolate_numeric(scenegraph.y_ticks, y_coord), ref_y)
                        if _axis_kind(scenegraph.y_ticks) == "numeric"
                        else truncate_end(
                            _nearest_categorical_label(scenegraph.y_ticks, y_coord),
                            self.truncate_point_label_len,
                        )
                    )

                point_pairs.append((str(x_value), str(y_value)))
        else:
            if reference_points:
                used_x_label_fallback = True
                if len(reference_points) == len(scenegraph.marks):
                    fallback_kind = "expected_datatable_fallback"
                    notes.append(
                        "Used exact datatable point sequence because scenegraph line/area geometry only provides approximate numeric values."
                    )
                else:
                    fallback_kind = "anomaly_mark_count_mismatch"
                    notes.append(
                        f"Used exact datatable point sequence because scenegraph mark count ({len(scenegraph.marks)}) "
                        f"did not match datatable points ({len(reference_points)})."
                    )
                point_pairs.extend((str(x_value), str(y_value)) for x_value, y_value in reference_points)
            else:
                for index, mark in enumerate(scenegraph.marks):
                    ref_x = reference_points[index][0] if index < len(reference_points) else None
                    ref_y = reference_points[index][1] if index < len(reference_points) else None

                    x_value = (
                        _format_numeric(_interpolate_numeric(scenegraph.x_ticks, mark.x), ref_x)
                        if _axis_kind(scenegraph.x_ticks) == "numeric"
                        else truncate_end(
                            _nearest_categorical_label(scenegraph.x_ticks, mark.x),
                            self.truncate_point_label_len,
                        )
                    )

                    y_value = (
                        _format_numeric(_interpolate_numeric(scenegraph.y_ticks, mark.y), ref_y)
                        if _axis_kind(scenegraph.y_ticks) == "numeric"
                        else truncate_end(
                            _nearest_categorical_label(scenegraph.y_ticks, mark.y),
                            self.truncate_point_label_len,
                        )
                    )
                    point_pairs.append((str(x_value), str(y_value)))

        ttl = self._build_ttl(
            chart_type=chart_type,
            title=properties.title,
            x_label=properties.x_label,
            y_label=properties.y_label,
            point_pairs=point_pairs,
        )

        return ConversionResult(
            img_id=img_id,
            ttl=ttl,
            used_x_label_fallback=used_x_label_fallback,
            fallback_kind=fallback_kind,
            notes=notes,
        )

    def _build_ttl(
        self,
        chart_type: str,
        title: str,
        x_label: str,
        y_label: str,
        point_pairs: list[tuple[str, str]],
    ) -> str:
        ttl_lines: list[str] = [
            "@prefix : <http://example.org/> .",
            "",
            f":Chart a :{chart_type} ;",
        ]

        if title:
            ttl_lines.append(f'    :title "{escape_ttl_string(truncate_middle(title, self.truncate_title_len))}" ;')

        ttl_lines.append("    :xAxis :XAxis ;")
        ttl_lines.append("    :yAxis :YAxis .")
        ttl_lines.append("")

        ttl_lines.append(":XAxis a :Axis ;")
        ttl_lines.append(f'    :title "{escape_ttl_string(x_label)}" .')
        ttl_lines.append("")

        ttl_lines.append(":YAxis a :Axis ;")
        ttl_lines.append(f'    :title "{escape_ttl_string(y_label)}" .')
        ttl_lines.append("")

        for index, (x_value, y_value) in enumerate(point_pairs, start=1):
            ttl_lines.append(f":DataPoint{index} a :DataPoint ;")
            ttl_lines.append(f'    :xValue "{escape_ttl_string(truncate_end(str(x_value), self.truncate_point_label_len))}" ;')
            ttl_lines.append(f'    :yValue "{escape_ttl_string(truncate_end(str(y_value), self.truncate_point_label_len))}" ;')
            ttl_lines.append("    :belongsTo :Chart .")
            ttl_lines.append("")

        return "\n".join(ttl_lines).rstrip() + "\n"

    def convert_json_to_ttl(self, json_data: dict[str, Any]) -> str:
        return self.convert_json_to_result(json_data).ttl

    def convert_file(self, json_file_path: str, output_file_path: Optional[str] = None) -> ConversionResult:
        json_path = Path(json_file_path)
        with json_path.open("r", encoding="utf-8") as handle:
            json_data = json.load(handle)

        result = self.convert_json_to_result(json_data)

        if output_file_path is None:
            output_path = self.output_dir / f"{json_path.stem}.ttl"
        else:
            output_path = Path(output_file_path)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(result.ttl, encoding="utf-8")
        return result

    def convert_directory(self, input_dir: str, output_dir: Optional[str] = None) -> dict[str, Any]:
        input_path = Path(input_dir)
        if output_dir is None:
            output_path = self.output_dir
        else:
            output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        expected_datatable_fallbacks: list[dict[str, Any]] = []
        anomaly_fallbacks: list[dict[str, Any]] = []
        errors: list[dict[str, str]] = []
        converted = 0

        for json_path in sorted(input_path.glob("*.json")):
            try:
                result = self.convert_file(str(json_path), str(output_path / f"{json_path.stem}.ttl"))
                converted += 1
                if result.fallback_kind:
                    case = {
                        "img_id": result.img_id,
                        "file": json_path.name,
                        "fallback_kind": result.fallback_kind,
                        "notes": result.notes,
                    }
                    if result.fallback_kind == "expected_datatable_fallback":
                        expected_datatable_fallbacks.append(case)
                    else:
                        anomaly_fallbacks.append(case)
            except Exception as exc:
                errors.append(
                    {
                        "file": json_path.name,
                        "error": str(exc),
                    }
                )
                print(f"Error converting {json_path.name}: {exc}")

        fallback_total = len(expected_datatable_fallbacks) + len(anomaly_fallbacks)
        report = {
            "input_dir": str(input_path),
            "output_dir": str(output_path),
            "converted": converted,
            "errors": len(errors),
            "fallback_summary": {
                "total": fallback_total,
                "expected_datatable_fallbacks": len(expected_datatable_fallbacks),
                "anomaly_fallbacks": len(anomaly_fallbacks),
            },
            "expected_datatable_fallbacks": expected_datatable_fallbacks,
            "anomaly_fallbacks": anomaly_fallbacks,
            "error_details": errors,
        }
        (output_path / "exceptions_report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(
            f"Converted {converted} files, {len(errors)} errors, "
            f"{fallback_total} fallback cases "
            f"({len(expected_datatable_fallbacks)} expected, {len(anomaly_fallbacks)} anomalies)"
        )
        return report


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert VisText JSON labels to TTL using scenegraph-first extraction."
    )
    parser.add_argument("input", help="Input JSON file or directory containing JSON files")
    parser.add_argument(
        "-o",
        "--output",
        help="Output TTL file (for file input) or output directory (for directory input)",
    )
    parser.add_argument("--truncate-label", type=int, default=50, help="Max chars for each point label")
    parser.add_argument("--truncate-title", type=int, default=150, help="Max chars for title")
    args = parser.parse_args()

    converter = JSONToTTLConverterScenegraph(
        truncate_point_label_len=args.truncate_label,
        truncate_title_len=args.truncate_title,
    )

    if os.path.isfile(args.input):
        result = converter.convert_file(args.input, args.output)
        print(f"Successfully converted to: {args.output or (converter.output_dir / (Path(args.input).stem + '.ttl'))}")
        if result.used_x_label_fallback:
            print(f"Datatable fallback used for img_id {result.img_id}")
        return 0

    if os.path.isdir(args.input):
        converter.convert_directory(args.input, args.output)
        print("\nConversion complete!")
        return 0

    print(f"Error: {args.input} is not a valid file or directory")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
