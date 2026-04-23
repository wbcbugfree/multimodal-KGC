from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from google import genai
from google.genai import types


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import get_api_key  # noqa: E402


DEFAULT_MODEL = "gemini-3-flash-preview"
ALLOWED_CATEGORIES = ("bar", "area", "line")
CLASSIFICATION_PROMPT = """
You are a chart identification assistant for a visual knowledge extraction dataset.
You will be given one chart image. Classify it as exactly one of:
- bar: a chart that represents values with separate rectangular bars, either vertical or horizontal.
- area: a chart that represents a quantitative series with a filled region under or between plotted lines.
- line: a chart that represents a quantitative series with one or more connected lines and no filled area.

Respond with only the category and no other text. The value must be one of: bar, area, line.
""".strip()


def create_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_api_key("gemini_api_key"))


def normalize_chart_type(value: str) -> str:
    normalized = value.strip().strip('"').lower()
    if normalized not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported chart type: {value}")
    return normalized


def classify_chart_type(
    image_path: Path,
    model: str = DEFAULT_MODEL,
    client: genai.Client | None = None,
) -> str:
    effective_client = client or create_gemini_client()
    response = effective_client.models.generate_content(
        model=model,
        contents=[
            CLASSIFICATION_PROMPT,
            types.Part.from_bytes(
                data=Path(image_path).read_bytes(),
                mime_type=get_mime_type(Path(image_path)),
            ),
        ],
        config=types.GenerateContentConfig(
            responseMimeType="application/json",
            responseSchema=types.Schema(
                type="STRING",
                enum=list(ALLOWED_CATEGORIES),
            ),
        ),
    )
    return normalize_chart_type(extract_response_text(response))


def extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise RuntimeError("Gemini returned no chart classification text")


def get_mime_type(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".jpg", ".jpeg"}:
        return "image/jpeg"
    if suffix == ".png":
        return "image/png"
    if suffix in {".tif", ".tiff"}:
        return "image/tiff"
    if suffix == ".webp":
        return "image/webp"
    if suffix == ".bmp":
        return "image/bmp"
    return "application/octet-stream"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Classify a vistext chart image as bar, area, or line using Gemini.",
    )
    parser.add_argument("image_path", type=Path, help="Path to the chart image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    chart_type = classify_chart_type(args.image_path, model=args.model)
    print(chart_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
