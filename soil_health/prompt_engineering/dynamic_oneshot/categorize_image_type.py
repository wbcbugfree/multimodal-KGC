from __future__ import annotations

import argparse
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
ALLOWED_CATEGORIES = ("figure", "table")
CLASSIFICATION_PROMPT = """
You are an image-type classification assistant for a soil-health knowledge extraction dataset.
You will be given one image. Classify it as exactly one of:
- figure: a diagram, flowchart, conceptual figure, chart, box, or visual schematic.
- table: a tabular matrix or table with rows, columns, cells, and table caption.

Respond with only the category and no other text. The value must be one of: figure, table.
""".strip()


def create_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_api_key("gemini_api_key"))


def normalize_image_type(value: str) -> str:
    normalized = value.strip().strip('"').lower()
    if normalized not in ALLOWED_CATEGORIES:
        raise ValueError(f"Unsupported image type: {value}")
    return normalized


def classify_image_type(
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
    return normalize_image_type(extract_response_text(response))


def extract_response_text(response: object) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    raise RuntimeError("Gemini returned no image-type classification text")


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
        description="Classify a soil-health image as figure or table using Gemini.",
    )
    parser.add_argument("image_path", type=Path, help="Path to the soil-health image.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    image_type = classify_image_type(args.image_path, model=args.model)
    print(image_type)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
