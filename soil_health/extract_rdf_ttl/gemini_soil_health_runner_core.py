from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from google import genai
from google.genai import types
from rdflib import Graph


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import get_api_key  # noqa: E402
from common.paths import soil_health_dir  # noqa: E402


DEFAULT_MODEL = "gemini-3-flash-preview"
REQUEST_DELAY = 2.0
PARALLEL_WORKERS = 1
MAX_ATTEMPTS = 3
SAMPLE_MODE = "random"
SAMPLE_COUNT = 5
RANDOM_SEED = 42
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FINAL_USER_PROMPT = (
    "Generate RDF/Turtle for this soil-health image. "
    "Follow the system instructions exactly and return only RDF/Turtle."
)
EXAMPLE_USER_PROMPT = (
    "Example {image_type} image with its correct RDF/Turtle output. "
    "Use this example only as a reference for structure and level of detail."
)

SOIL_HEALTH_DIR = soil_health_dir()
IMAGES_ROOT = SOIL_HEALTH_DIR / "data"
PROMPT_ROOT = SOIL_HEALTH_DIR / "prompt_engineering"
GROUND_TRUTH_ROOT = PROMPT_ROOT / "ground_truth"


@dataclass(frozen=True)
class PromptExample:
    image_type: str
    image_path: Path
    ttl_path: Path


@dataclass(frozen=True)
class PromptPackage:
    system_prompt: str
    examples: list[PromptExample]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBuilderContext:
    model: str
    client: genai.Client | None = None


@dataclass(frozen=True)
class RuntimeConfig:
    model: str
    images_root: Path
    output_dir: Path
    manifest_path: Path
    request_delay: float
    sample_mode: str
    sample_count: int | None
    seed: int
    ids: list[str] | None
    parallel_workers: int
    skip_existing: bool
    dry_run: bool


PromptBuilder = Callable[[Path, PromptBuilderContext], PromptPackage]
_THREAD_LOCAL = threading.local()


def build_parser(
    description: str,
    default_output_dir: Path,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
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
        "--parallel-workers",
        type=int,
        default=PARALLEL_WORKERS,
        help="Number of parallel Gemini worker threads to use.",
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
        help="Delay in seconds after each successful API call or retryable failure.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
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


def parse_args(
    argv: list[str] | None,
    description: str,
    default_output_dir: Path,
) -> argparse.Namespace:
    parser = build_parser(description=description, default_output_dir=default_output_dir)
    args = parser.parse_args(argv)

    if args.sample_mode == "random" and args.sample_count <= 0:
        parser.error("--sample-count must be greater than 0 when --sample-mode=random")

    if args.sample_mode == "ids" and not args.ids:
        parser.error("--ids is required when --sample-mode=ids")

    if args.request_delay < 0:
        parser.error("--request-delay must be greater than or equal to 0")

    if args.parallel_workers <= 0:
        parser.error("--parallel-workers must be greater than 0")

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
        output_dir=output_dir,
        manifest_path=manifest_path,
        request_delay=args.request_delay,
        sample_mode=args.sample_mode,
        sample_count=args.sample_count,
        seed=args.seed,
        ids=normalize_ids(args.ids),
        parallel_workers=args.parallel_workers,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
    )


def normalize_ids(ids: list[str] | None) -> list[str] | None:
    if ids is None:
        return None

    normalized: list[str] = []
    seen: set[str] = set()
    for raw_id in ids:
        raw_path = Path(raw_id.strip())
        image_id = (
            raw_path.stem
            if raw_path.suffix.lower() in VALID_IMAGE_EXTS
            else raw_path.name
        ).strip()
        if not image_id or image_id in seen:
            continue
        normalized.append(image_id)
        seen.add(image_id)
    return normalized


def load_text(path: Path) -> str:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        raise ValueError(f"File is empty: {path}")
    return text


def image_sort_key(path: Path) -> tuple[str, int | str, str]:
    parent_name = path.parent.name
    stem = path.stem
    return (parent_name, natural_stem_key(stem), path.name)


def natural_stem_key(stem: str) -> int | str:
    if stem.isdigit():
        return int(stem)
    return stem


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
            raise ValueError(f"Unknown image ids: {', '.join(missing_ids)}")
        return [image_by_id[image_id] for image_id in ids]

    raise ValueError(f"Unsupported sample mode: {sample_mode}")


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
    return f"Soil-health image id: {image_id}. {FINAL_USER_PROMPT}"


def build_example_user_prompt(image_type: str) -> str:
    return EXAMPLE_USER_PROMPT.format(image_type=image_type)


def build_example_contents(examples: list[PromptExample]) -> list[types.Content]:
    contents: list[types.Content] = []
    for example in examples:
        contents.append(
            types.UserContent(
                parts=[
                    types.Part.from_text(text=build_example_user_prompt(example.image_type)),
                    types.Part.from_bytes(
                        data=example.image_path.read_bytes(),
                        mime_type=get_mime_type(example.image_path),
                    ),
                ]
            )
        )
        contents.append(
            types.ModelContent(parts=[types.Part.from_text(text=load_text(example.ttl_path))])
        )
    return contents


def build_request_contents(
    user_prompt: str,
    image_path: Path,
    examples: list[PromptExample],
) -> list[types.Content]:
    current_turn = types.UserContent(
        parts=[
            types.Part.from_text(text=user_prompt),
            types.Part.from_bytes(
                data=image_path.read_bytes(),
                mime_type=get_mime_type(image_path),
            ),
        ]
    )
    return [*build_example_contents(examples), current_turn]


def create_gemini_client() -> genai.Client:
    return genai.Client(api_key=get_api_key("gemini_api_key"))


def get_worker_client() -> genai.Client:
    client = getattr(_THREAD_LOCAL, "gemini_client", None)
    if client is None:
        client = create_gemini_client()
        _THREAD_LOCAL.gemini_client = client
    return client


def call_gemini(
    client: genai.Client,
    model: str,
    system_prompt: str,
    contents: list[types.Content],
) -> str:
    response = client.models.generate_content(
        model=model,
        contents=contents,
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
    except Exception as exc:  # pragma: no cover
        return str(exc)
    return None


def sleep_if_needed(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def build_attempt_failure(
    attempt: int,
    status: str,
    error: str,
    stage: str,
) -> dict[str, Any]:
    return {
        "attempt": attempt,
        "status": status,
        "stage": stage,
        "error": error,
    }


def finalize_attempt_metadata(
    prompt_package: PromptPackage | None,
    attempt_count: int,
    attempt_failures: list[dict[str, Any]],
) -> dict[str, Any] | None:
    metadata = dict(prompt_package.metadata) if prompt_package is not None else {}
    metadata["attempt_count"] = attempt_count
    if attempt_failures:
        metadata["attempt_failures"] = attempt_failures
    return metadata or None


def exhausted_retry_status(attempt_failures: list[dict[str, Any]]) -> str:
    statuses = {str(failure.get("status", "")) for failure in attempt_failures}
    if statuses == {"invalid_ttl"}:
        return "invalid_ttl"
    if statuses == {"api_error"}:
        return "api_error"
    return "retry_exhausted"


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


def manifest_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    return (0, str(item.get("img_id", "")))


def upsert_manifest_item(manifest: dict[str, Any], item: dict[str, Any]) -> None:
    items = manifest.setdefault("items", [])
    for index, existing in enumerate(items):
        if existing.get("img_id") == item.get("img_id"):
            items[index] = item
            break
    else:
        items.append(item)

    items.sort(key=manifest_sort_key)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)


def build_manifest_item(
    image_path: Path,
    ttl_path: Path,
    model: str,
    status: str,
    error: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "img_id": image_path.stem,
        "image_type": infer_image_type_from_path(image_path),
        "source_image": str(image_path.resolve()),
        "ttl_file": str(ttl_path.resolve()),
        "model": model,
        "status": status,
    }
    if error:
        item["error"] = error
    if extra_metadata:
        item.update(extra_metadata)
    return item


def infer_image_type_from_path(image_path: Path) -> str:
    parent_name = image_path.parent.name.lower()
    if parent_name == "figures" or image_path.stem.lower().startswith(("figure_", "box_")):
        return "figure"
    if parent_name == "tables" or image_path.stem.lower().startswith("table_"):
        return "table"
    return parent_name or "unknown"


def write_ttl(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def process_image(
    client: genai.Client | None,
    config: RuntimeConfig,
    prompt_builder: PromptBuilder,
    image_path: Path,
) -> dict[str, Any]:
    image_id = image_path.stem
    ttl_path = config.output_dir / f"{image_id}.ttl"
    prompt_package: PromptPackage | None = None
    contents: list[types.Content] | None = None
    attempt_failures: list[dict[str, Any]] = []

    if config.skip_existing and ttl_path.exists():
        return build_manifest_item(
            image_path=image_path,
            ttl_path=ttl_path,
            model=config.model,
            status="skipped_existing",
        )

    effective_client = client or create_gemini_client()
    for attempt in range(1, MAX_ATTEMPTS + 1):
        if prompt_package is None or contents is None:
            try:
                prompt_package = prompt_builder(
                    image_path,
                    PromptBuilderContext(model=config.model, client=effective_client),
                )
                contents = build_request_contents(
                    user_prompt=build_user_prompt(image_id),
                    image_path=image_path,
                    examples=prompt_package.examples,
                )
            except Exception as exc:  # pragma: no cover
                attempt_failures.append(
                    build_attempt_failure(
                        attempt=attempt,
                        status="api_error",
                        stage="prompt_builder",
                        error=str(exc),
                    )
                )
                if attempt < MAX_ATTEMPTS:
                    sleep_if_needed(config.request_delay)
                continue

        try:
            raw_output = call_gemini(
                client=effective_client,
                model=config.model,
                system_prompt=prompt_package.system_prompt,
                contents=contents,
            )
        except Exception as exc:  # pragma: no cover
            attempt_failures.append(
                build_attempt_failure(
                    attempt=attempt,
                    status="api_error",
                    stage="generation",
                    error=str(exc),
                )
            )
            if attempt < MAX_ATTEMPTS:
                sleep_if_needed(config.request_delay)
            continue

        turtle_text = strip_code_fences(raw_output)
        parse_error = validate_turtle(turtle_text)
        if parse_error is not None:
            attempt_failures.append(
                build_attempt_failure(
                    attempt=attempt,
                    status="invalid_ttl",
                    stage="validation",
                    error=parse_error,
                )
            )
            if attempt < MAX_ATTEMPTS:
                sleep_if_needed(config.request_delay)
            continue

        write_ttl(ttl_path, turtle_text)
        sleep_if_needed(config.request_delay)
        return build_manifest_item(
            image_path=image_path,
            ttl_path=ttl_path,
            model=config.model,
            status="saved",
            extra_metadata=finalize_attempt_metadata(
                prompt_package=prompt_package,
                attempt_count=attempt,
                attempt_failures=attempt_failures,
            ),
        )

    final_error = attempt_failures[-1]["error"] if attempt_failures else "Unknown error"
    return build_manifest_item(
        image_path=image_path,
        ttl_path=ttl_path,
        model=config.model,
        status=exhausted_retry_status(attempt_failures),
        error=final_error,
        extra_metadata=finalize_attempt_metadata(
            prompt_package=prompt_package,
            attempt_count=MAX_ATTEMPTS,
            attempt_failures=attempt_failures,
        ),
    )


def process_image_in_worker(
    config: RuntimeConfig,
    prompt_builder: PromptBuilder,
    image_path: Path,
) -> dict[str, Any]:
    return process_image(
        client=get_worker_client(),
        config=config,
        prompt_builder=prompt_builder,
        image_path=image_path,
    )


def run_strategy(
    argv: list[str] | None,
    description: str,
    default_output_dir: Path,
    prompt_builder: PromptBuilder,
) -> int:
    args = parse_args(argv, description=description, default_output_dir=default_output_dir)
    config = resolve_runtime_config(args)

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
    run_items: list[dict[str, Any]] = []
    if config.parallel_workers == 1:
        client = create_gemini_client()
        for index, image_path in enumerate(selected_images, start=1):
            print(f"[{index}/{len(selected_images)}] Processing {image_path.name}")
            item = process_image(
                client=client,
                config=config,
                prompt_builder=prompt_builder,
                image_path=image_path,
            )
            run_items.append(item)
            upsert_manifest_item(manifest, item)
            write_manifest(config.manifest_path, manifest)
            print_item_status(item)
    else:
        print(
            f"[PARALLEL] Processing {len(selected_images)} image(s) "
            f"with {config.parallel_workers} worker(s)"
        )
        with ThreadPoolExecutor(
            max_workers=config.parallel_workers,
            thread_name_prefix="gemini-soil-health",
        ) as executor:
            future_to_image = {
                executor.submit(
                    process_image_in_worker,
                    config,
                    prompt_builder,
                    image_path,
                ): image_path
                for image_path in selected_images
            }
            for index, future in enumerate(as_completed(future_to_image), start=1):
                image_path = future_to_image[future]
                item = future.result()
                print(f"[{index}/{len(selected_images)}] Processed {image_path.name}")
                run_items.append(item)
                upsert_manifest_item(manifest, item)
                write_manifest(config.manifest_path, manifest)
                print_item_status(item)

    counts = Counter(item.get("status", "unknown") for item in run_items)
    print("[SUMMARY]")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"  manifest: {config.manifest_path}")
    return 0


def print_item_status(item: dict[str, Any]) -> None:
    status = item["status"]
    if status == "saved":
        print(f"  saved -> {Path(item['ttl_file']).name}")
    elif "error" in item:
        print(f"  {status}: {item['error']}")
    else:
        print(f"  {status}")


def ground_truth_example(image_type: str) -> PromptExample:
    example_paths = {
        "figure": ("figure", "figure_1.1.jpg", "figure_1.1.ttl"),
        "table": ("table", "table_1.1.jpg", "table_1.1.ttl"),
    }
    try:
        folder, image_name, ttl_name = example_paths[image_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported image type: {image_type}") from exc

    base_dir = GROUND_TRUTH_ROOT / folder
    return PromptExample(
        image_type=image_type,
        image_path=base_dir / image_name,
        ttl_path=base_dir / ttl_name,
    )
