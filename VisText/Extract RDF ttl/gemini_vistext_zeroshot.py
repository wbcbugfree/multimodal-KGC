from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from google import genai
from google.genai import types
from rdflib import Graph


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import get_api_key  # noqa: E402
from common.paths import vistext_dir  # noqa: E402


MODEL = "gemini-3-flash-preview"
REQUEST_DELAY = 2.0
SAMPLE_MODE = "random"
SAMPLE_COUNT = 5
RANDOM_SEED = 42
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
USER_PROMPT = (
    "Generate RDF/Turtle for this chart image. "
    "Follow the system instructions exactly and return only RDF/Turtle."
)

VIS_TEXT_DIR = vistext_dir()
IMAGES_ROOT = VIS_TEXT_DIR / "Data" / "images"
LABELS_ROOT = VIS_TEXT_DIR / "Data" / "labels"
PROMPT_PATH = VIS_TEXT_DIR / "Prompt Text" / "zeroshot.md"
OUTPUT_DIR = VIS_TEXT_DIR / "Extract RDF ttl" / "vistext_Zeroshot_outputs"
MANIFEST_PATH = OUTPUT_DIR / "manifest.json"


@dataclass
class RuntimeConfig:
    model: str
    images_root: Path
    labels_root: Path
    prompt_path: Path
    output_dir: Path
    manifest_path: Path
    request_delay: float
    sample_mode: str
    sample_count: int | None
    seed: int
    ids: list[str] | None
    skip_existing: bool
    dry_run: bool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Call Gemini on VisText chart images and save only valid Turtle outputs.",
    )
    parser.add_argument("--model", default=MODEL, help="Gemini model name.")
    parser.add_argument(
        "--sample-mode",
        choices=("all", "random", "ids"),
        default=SAMPLE_MODE,
        help="Select all images, a deterministic random sample, or explicit ids.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=SAMPLE_COUNT,
        help="Number of images to sample when --sample-mode=random.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for deterministic sampling.",
    )
    parser.add_argument(
        "--ids",
        nargs="+",
        help="Explicit image ids or filenames when --sample-mode=ids.",
    )
    parser.add_argument(
        "--request-delay",
        type=float,
        default=REQUEST_DELAY,
        help="Delay in seconds after each successful API call.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR,
        help="Directory where valid .ttl files are written.",
    )
    parser.add_argument(
        "--manifest-path",
        type=Path,
        default=None,
        help="Manifest file path. Defaults to <output-dir>/manifest.json.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip image ids that already have a .ttl file in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected ids without calling Gemini or writing files.",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.sample_mode == "random" and args.sample_count <= 0:
        parser.error("--sample-count must be greater than 0 when --sample-mode=random")

    if args.sample_mode == "ids" and not args.ids:
        parser.error("--ids is required when --sample-mode=ids")

    if args.request_delay < 0:
        parser.error("--request-delay must be greater than or equal to 0")

    return args


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    output_dir = args.output_dir.resolve()
    manifest_path = (
        args.manifest_path.resolve()
        if args.manifest_path is not None
        else output_dir / "manifest.json"
    )

    return RuntimeConfig(
        model=args.model,
        images_root=IMAGES_ROOT.resolve(),
        labels_root=LABELS_ROOT.resolve(),
        prompt_path=PROMPT_PATH.resolve(),
        output_dir=output_dir,
        manifest_path=manifest_path,
        request_delay=args.request_delay,
        sample_mode=args.sample_mode,
        sample_count=args.sample_count,
        seed=args.seed,
        ids=normalize_ids(args.ids),
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )


def normalize_ids(ids: list[str] | None) -> list[str] | None:
    if ids is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in ids:
        image_id = Path(raw_id).stem.strip()
        if not image_id or image_id in seen:
            continue
        normalized.append(image_id)
        seen.add(image_id)
    return normalized


def load_system_prompt(path: Path) -> str:
    prompt = path.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError(f"System prompt file is empty: {path}")
    return prompt


def list_image_paths(images_root: Path) -> list[Path]:
    return sorted(
        (
            path
            for path in images_root.rglob("*")
            if path.is_file() and path.suffix.lower() in VALID_IMAGE_EXTS
        ),
        key=image_sort_key,
    )


def select_image_paths(
    image_paths: list[Path],
    sample_mode: str,
    sample_count: int | None,
    seed: int,
    ids: list[str] | None,
) -> list[Path]:
    ordered = sorted(image_paths, key=image_sort_key)

    if sample_mode == "all":
        return ordered

    if sample_mode == "random":
        if sample_count is None:
            raise ValueError("sample_count is required when sample_mode='random'")
        if sample_count >= len(ordered):
            return ordered
        selected = random.Random(seed).sample(ordered, sample_count)
        return sorted(selected, key=image_sort_key)

    if sample_mode == "ids":
        if not ids:
            raise ValueError("ids are required when sample_mode='ids'")
        image_by_id = {path.stem: path for path in ordered}
        missing_ids = [image_id for image_id in ids if image_id not in image_by_id]
        if missing_ids:
            joined = ", ".join(missing_ids)
            raise ValueError(f"Unknown image ids: {joined}")
        return [image_by_id[image_id] for image_id in ids]

    raise ValueError(f"Unsupported sample mode: {sample_mode}")


def image_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    return (1, stem, path.name)


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


def build_user_prompt(image_id: str) -> str:
    return f"Chart image id: {image_id}. {USER_PROMPT}"


def create_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_api_key("gemini_api_key"))


def call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    user_prompt: str,
    image_path: Path,
) -> str:
    image_bytes = image_path.read_bytes()
    image_part = types.Part.from_bytes(
        data=image_bytes,
        mime_type=get_mime_type(image_path),
    )
    response = client.models.generate_content(
        model=model,
        contents=[user_prompt, image_part],
        config=types.GenerateContentConfig(
            systemInstruction=system_prompt,
        ),
    )
    return extract_response_text(response)


def extract_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: list[str] = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if content is None:
            continue
        for part in getattr(content, "parts", []) or []:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str) and part_text.strip():
                parts.append(part_text.strip())

    if parts:
        return "\n".join(parts)

    raise RuntimeError("Gemini returned no text content")


def strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def validate_turtle(text: str) -> str | None:
    if not text.strip():
        return "Gemini returned an empty response"

    graph = Graph()
    try:
        graph.parse(data=text, format="turtle")
    except Exception as exc:  # pragma: no cover - exact parser wording varies
        return str(exc)
    return None


def load_existing_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": []}

    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    if not isinstance(manifest, dict):
        raise ValueError(f"Manifest must contain a JSON object: {path}")

    items = manifest.get("items")
    if items is None:
        manifest["items"] = []
        return manifest
    if not isinstance(items, list):
        raise ValueError(f"Manifest 'items' must be a list: {path}")
    return manifest


def upsert_manifest_item(manifest: dict[str, Any], item: dict[str, Any]) -> None:
    items = manifest.setdefault("items", [])
    for index, existing in enumerate(items):
        if existing.get("img_id") == item.get("img_id"):
            items[index] = item
            break
    else:
        items.append(item)

    items.sort(key=manifest_sort_key)


def manifest_sort_key(item: dict[str, Any]) -> tuple[int, int | str]:
    img_id = str(item.get("img_id", ""))
    if img_id.isdigit():
        return (0, int(img_id))
    return (1, img_id)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def build_manifest_item(
    image_path: Path,
    label_path: Path,
    ttl_path: Path,
    model: str,
    status: str,
    error: str | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "img_id": image_path.stem,
        "source_image": str(image_path.resolve()),
        "ttl_file": str(ttl_path.resolve()),
        "model": model,
        "status": status,
    }
    if label_path.exists():
        item["json_label"] = str(label_path.resolve())
    if error:
        item["error"] = error
    return item


def write_ttl(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def process_image(
    client: genai.Client,
    config: RuntimeConfig,
    system_prompt: str,
    image_path: Path,
) -> dict[str, Any]:
    image_id = image_path.stem
    ttl_path = config.output_dir / f"{image_id}.ttl"
    label_path = config.labels_root / f"{image_id}.json"

    if config.skip_existing and ttl_path.exists():
        return build_manifest_item(
            image_path=image_path,
            label_path=label_path,
            ttl_path=ttl_path,
            model=config.model,
            status="skipped_existing",
        )

    try:
        raw_output = call_gemini(
            client=client,
            model=config.model,
            system_prompt=system_prompt,
            user_prompt=build_user_prompt(image_id),
            image_path=image_path,
        )
        turtle_text = strip_code_fences(raw_output)
        parse_error = validate_turtle(turtle_text)
        if parse_error is not None:
            return build_manifest_item(
                image_path=image_path,
                label_path=label_path,
                ttl_path=ttl_path,
                model=config.model,
                status="invalid_ttl",
                error=parse_error,
            )

        write_ttl(ttl_path, turtle_text)
        if config.request_delay > 0:
            time.sleep(config.request_delay)
        return build_manifest_item(
            image_path=image_path,
            label_path=label_path,
            ttl_path=ttl_path,
            model=config.model,
            status="saved",
        )
    except Exception as exc:  # pragma: no cover - network/runtime errors are smoke-tested
        return build_manifest_item(
            image_path=image_path,
            label_path=label_path,
            ttl_path=ttl_path,
            model=config.model,
            status="api_error",
            error=str(exc),
        )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = resolve_runtime_config(args)

    system_prompt = load_system_prompt(config.prompt_path)
    image_paths = list_image_paths(config.images_root)
    selected_images = select_image_paths(
        image_paths=image_paths,
        sample_mode=config.sample_mode,
        sample_count=config.sample_count,
        seed=config.seed,
        ids=config.ids,
    )

    if config.dry_run:
        print(f"[DRY-RUN] Selected {len(selected_images)} image(s)")
        for image_path in selected_images:
            print(image_path.stem)
        return 0

    manifest = load_existing_manifest(config.manifest_path)
    client = create_gemini_client()
    run_items: list[dict[str, Any]] = []

    for index, image_path in enumerate(selected_images, start=1):
        print(f"[{index}/{len(selected_images)}] Processing {image_path.name}")
        item = process_image(
            client=client,
            config=config,
            system_prompt=system_prompt,
            image_path=image_path,
        )
        run_items.append(item)
        upsert_manifest_item(manifest, item)
        write_manifest(config.manifest_path, manifest)

        status = item["status"]
        if status == "saved":
            print(f"  saved -> {Path(item['ttl_file']).name}")
        elif "error" in item:
            print(f"  {status}: {item['error']}")
        else:
            print(f"  {status}")

    counts = Counter(item.get("status", "unknown") for item in run_items)
    print("[SUMMARY]")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"  manifest: {config.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
