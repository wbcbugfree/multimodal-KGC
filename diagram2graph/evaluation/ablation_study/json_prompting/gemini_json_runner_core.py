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


SCRIPT_DIR = Path(__file__).resolve().parent
ABLATION_DIR = SCRIPT_DIR.parent
DIAGRAM2GRAPH_DIR = ABLATION_DIR.parents[1]
REPO_ROOT = DIAGRAM2GRAPH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import get_api_key  # noqa: E402


DEFAULT_MODEL = "gemini-3-flash-preview"
REQUEST_DELAY = 2.0
PARALLEL_WORKERS = 1
MAX_ATTEMPTS = 3
SAMPLE_MODE = "all"
SAMPLE_COUNT = 5
RANDOM_SEED = 42
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
EXAMPLE_IDS = ["3", "4", "6"]
FINAL_USER_PROMPT = (
    "Generate JSON for this diagram image. "
    "Follow the system instructions and JSON schema exactly."
)
EXAMPLE_USER_PROMPT = (
    "Example diagram image with its correct JSON output. "
    "Use this example only as a reference for structure and level of detail."
)

EVAL_IMAGES_ROOT = ABLATION_DIR / "eval_img"
EVAL_GOLD_ROOT = ABLATION_DIR / "eval_gt"
OUTPUTS_ROOT = ABLATION_DIR / "outputs"
PROMPT_ROOT = SCRIPT_DIR
SYSTEM_PROMPT_PATH = PROMPT_ROOT / "zeroshot_system.md"
MAIN_DATA_ROOT = DIAGRAM2GRAPH_DIR / "data"
MAIN_IMAGES_ROOT = MAIN_DATA_ROOT / "images"
MAIN_LABELS_ROOT = MAIN_DATA_ROOT / "labels"

NODE_TYPES = ["start", "process", "decision", "delay", "terminator"]
NODE_SHAPES = ["start_event", "end_event", "task", "gateway", "data_store"]
EDGE_TYPES = ["solid", "dashed"]
RELATIONSHIP_TYPES = ["follows", "branches", "depends_on"]
NODE_KEYS = ["id", "type_of_node", "shape", "label"]
EDGE_KEYS = [
    "source",
    "source_type",
    "source_label",
    "target",
    "target_type",
    "target_label",
    "type_of_edge",
    "relationship_value",
    "relationship_type",
]

DIAGRAM_JSON_RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    required=["nodes", "edges"],
    propertyOrdering=["nodes", "edges"],
    properties={
        "nodes": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=NODE_KEYS,
                propertyOrdering=NODE_KEYS,
                properties={
                    "id": types.Schema(type=types.Type.STRING),
                    "type_of_node": types.Schema(type=types.Type.STRING, enum=NODE_TYPES),
                    "shape": types.Schema(type=types.Type.STRING, enum=NODE_SHAPES),
                    "label": types.Schema(type=types.Type.STRING),
                },
            ),
        ),
        "edges": types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                required=EDGE_KEYS,
                propertyOrdering=EDGE_KEYS,
                properties={
                    "source": types.Schema(type=types.Type.STRING),
                    "source_type": types.Schema(type=types.Type.STRING, enum=NODE_TYPES),
                    "source_label": types.Schema(type=types.Type.STRING),
                    "target": types.Schema(type=types.Type.STRING),
                    "target_type": types.Schema(type=types.Type.STRING, enum=NODE_TYPES),
                    "target_label": types.Schema(type=types.Type.STRING),
                    "type_of_edge": types.Schema(type=types.Type.STRING, enum=EDGE_TYPES),
                    "relationship_value": types.Schema(type=types.Type.STRING),
                    "relationship_type": types.Schema(type=types.Type.STRING, enum=RELATIONSHIP_TYPES),
                },
            ),
        ),
    },
)


@dataclass(frozen=True)
class PromptExample:
    example_id: str
    image_path: Path
    json_path: Path


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
    gold_root: Path
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


def build_parser(description: str, default_output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--sample-mode",
        choices=("all", "random", "ids"),
        default=SAMPLE_MODE,
        help="Select all ablation images, a deterministic random sample, or explicit ids.",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=SAMPLE_COUNT,
        help="Number of images to sample when --sample-mode=random.",
    )
    parser.add_argument("--seed", type=int, default=RANDOM_SEED, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=PARALLEL_WORKERS,
        help="Number of parallel Gemini worker threads to use.",
    )
    parser.add_argument("--ids", nargs="+", help="Explicit image ids or filenames when --sample-mode=ids.")
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
        help="Directory where valid .json files are written.",
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
        help="Skip image ids that already have a .json file in the output directory.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print selected ids without calling Gemini or writing files.",
    )
    return parser


def parse_args(argv: list[str] | None, description: str, default_output_dir: Path) -> argparse.Namespace:
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
    manifest_path = args.manifest_path.resolve() if args.manifest_path is not None else output_dir / "manifest.json"
    return RuntimeConfig(
        model=args.model,
        images_root=EVAL_IMAGES_ROOT.resolve(),
        gold_root=EVAL_GOLD_ROOT.resolve(),
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
        image_id = Path(raw_id).stem.strip()
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


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_diagram_json(data)
    return data


def image_sort_key(path: Path) -> tuple[int, int | str, str]:
    stem = path.stem
    if stem.isdigit():
        return (0, int(stem), path.name)
    return (1, stem, path.name)


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
            raise ValueError(f"Unknown ablation image ids: {', '.join(missing_ids)}")
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
    return f"Diagram image id: {image_id}. {FINAL_USER_PROMPT}"


def build_example_contents(examples: list[PromptExample]) -> list[types.Content]:
    contents: list[types.Content] = []
    for example in examples:
        contents.append(
            types.UserContent(
                parts=[
                    types.Part.from_text(text=EXAMPLE_USER_PROMPT),
                    types.Part.from_bytes(
                        data=example.image_path.read_bytes(),
                        mime_type=get_mime_type(example.image_path),
                    ),
                ]
            )
        )
        contents.append(
            types.ModelContent(
                parts=[
                    types.Part.from_text(
                        text=json.dumps(load_json(example.json_path), indent=2, ensure_ascii=False)
                    )
                ]
            )
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


def call_gemini_json(
    client: genai.Client,
    model: str,
    system_prompt: str,
    contents: list[types.Content],
) -> dict[str, Any]:
    response = generate_content(
        client=client,
        model=model,
        system_prompt=system_prompt,
        contents=contents,
    )
    return extract_response_json(response)


def generate_content(
    client: genai.Client,
    model: str,
    system_prompt: str,
    contents: list[types.Content],
) -> Any:
    return client.models.generate_content(
        model=model,
        contents=contents,
        config=types.GenerateContentConfig(
            systemInstruction=system_prompt,
            responseMimeType="application/json",
            responseSchema=DIAGRAM_JSON_RESPONSE_SCHEMA,
        ),
    )


def extract_response_json(response: Any) -> dict[str, Any]:
    parsed = getattr(response, "parsed", None)
    if isinstance(parsed, dict):
        validate_diagram_json(parsed)
        return parsed

    text = extract_response_text(response)
    try:
        data = json.loads(strip_code_fences(text))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Gemini returned invalid JSON: {exc}") from exc
    validate_diagram_json(data)
    return data


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


def validate_diagram_json(data: Any) -> None:
    if not isinstance(data, dict):
        raise ValueError("Diagram JSON must be an object")
    if set(data) != {"nodes", "edges"}:
        raise ValueError("Diagram JSON must contain exactly the top-level keys: nodes, edges")
    if not isinstance(data["nodes"], list):
        raise ValueError("Diagram JSON 'nodes' must be a list")
    if not isinstance(data["edges"], list):
        raise ValueError("Diagram JSON 'edges' must be a list")

    node_ids: set[str] = set()
    for index, node in enumerate(data["nodes"], start=1):
        validate_object_keys(node, NODE_KEYS, f"nodes[{index}]")
        validate_string_fields(node, NODE_KEYS, f"nodes[{index}]")
        validate_enum(node, "type_of_node", NODE_TYPES, f"nodes[{index}]")
        validate_enum(node, "shape", NODE_SHAPES, f"nodes[{index}]")
        if not node["id"].strip():
            raise ValueError(f"nodes[{index}].id must not be empty")
        node_ids.add(node["id"])

    for index, edge in enumerate(data["edges"], start=1):
        validate_object_keys(edge, EDGE_KEYS, f"edges[{index}]")
        validate_string_fields(edge, EDGE_KEYS, f"edges[{index}]")
        validate_enum(edge, "source_type", NODE_TYPES, f"edges[{index}]")
        validate_enum(edge, "target_type", NODE_TYPES, f"edges[{index}]")
        validate_enum(edge, "type_of_edge", EDGE_TYPES, f"edges[{index}]")
        validate_enum(edge, "relationship_type", RELATIONSHIP_TYPES, f"edges[{index}]")
        if edge["source"] not in node_ids:
            raise ValueError(f"edges[{index}].source references missing node id: {edge['source']}")
        if edge["target"] not in node_ids:
            raise ValueError(f"edges[{index}].target references missing node id: {edge['target']}")


def validate_object_keys(value: Any, expected_keys: list[str], context: str) -> None:
    if not isinstance(value, dict):
        raise ValueError(f"{context} must be an object")
    actual_keys = set(value)
    expected = set(expected_keys)
    if actual_keys != expected:
        missing = sorted(expected - actual_keys)
        extra = sorted(actual_keys - expected)
        parts = []
        if missing:
            parts.append(f"missing={missing}")
        if extra:
            parts.append(f"extra={extra}")
        raise ValueError(f"{context} has invalid keys: {', '.join(parts)}")


def validate_string_fields(value: dict[str, Any], keys: list[str], context: str) -> None:
    for key in keys:
        if not isinstance(value[key], str):
            raise ValueError(f"{context}.{key} must be a string")


def validate_enum(value: dict[str, str], key: str, allowed: list[str], context: str) -> None:
    if value[key] not in allowed:
        raise ValueError(f"{context}.{key} must be one of {allowed}; got {value[key]!r}")


def sleep_if_needed(delay_seconds: float) -> None:
    if delay_seconds > 0:
        time.sleep(delay_seconds)


def build_attempt_failure(attempt: int, status: str, error: str, stage: str) -> dict[str, Any]:
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
    if statuses == {"invalid_json"}:
        return "invalid_json"
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


def manifest_sort_key(item: dict[str, Any]) -> tuple[int, int | str]:
    img_id = str(item.get("img_id", ""))
    if img_id.isdigit():
        return (0, int(img_id))
    return (1, img_id)


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
    gold_path: Path,
    output_path: Path,
    model: str,
    status: str,
    error: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "img_id": image_path.stem,
        "source_image": str(image_path.resolve()),
        "json_file": str(output_path.resolve()),
        "model": model,
        "status": status,
    }
    if gold_path.exists():
        item["json_label"] = str(gold_path.resolve())
    if error:
        item["error"] = error
    if extra_metadata:
        item.update(extra_metadata)
    return item


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    validate_diagram_json(data)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def process_image(
    client: genai.Client | None,
    config: RuntimeConfig,
    prompt_builder: PromptBuilder,
    image_path: Path,
) -> dict[str, Any]:
    image_id = image_path.stem
    output_path = config.output_dir / f"{image_id}.json"
    gold_path = config.gold_root / f"{image_id}.json"
    prompt_package: PromptPackage | None = None
    contents: list[types.Content] | None = None
    attempt_failures: list[dict[str, Any]] = []

    if config.skip_existing and output_path.exists():
        return build_manifest_item(
            image_path=image_path,
            gold_path=gold_path,
            output_path=output_path,
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
                    build_attempt_failure(attempt, "api_error", str(exc), "prompt_builder")
                )
                if attempt < MAX_ATTEMPTS:
                    sleep_if_needed(config.request_delay)
                continue

        try:
            response = generate_content(
                client=effective_client,
                model=config.model,
                system_prompt=prompt_package.system_prompt,
                contents=contents,
            )
        except Exception as exc:  # pragma: no cover
            attempt_failures.append(
                build_attempt_failure(attempt, "api_error", str(exc), "generation")
            )
            if attempt < MAX_ATTEMPTS:
                sleep_if_needed(config.request_delay)
            continue

        try:
            output_json = extract_response_json(response)
        except Exception as exc:
            attempt_failures.append(
                build_attempt_failure(attempt, "invalid_json", str(exc), "validation")
            )
            if attempt < MAX_ATTEMPTS:
                sleep_if_needed(config.request_delay)
            continue

        write_json(output_path, output_json)
        sleep_if_needed(config.request_delay)
        return build_manifest_item(
            image_path=image_path,
            gold_path=gold_path,
            output_path=output_path,
            model=config.model,
            status="saved",
            extra_metadata=finalize_attempt_metadata(prompt_package, attempt, attempt_failures),
        )

    final_error = attempt_failures[-1]["error"] if attempt_failures else "Unknown error"
    return build_manifest_item(
        image_path=image_path,
        gold_path=gold_path,
        output_path=output_path,
        model=config.model,
        status=exhausted_retry_status(attempt_failures),
        error=final_error,
        extra_metadata=finalize_attempt_metadata(prompt_package, MAX_ATTEMPTS, attempt_failures),
    )


def process_image_in_worker(
    config: RuntimeConfig,
    prompt_builder: PromptBuilder,
    image_path: Path,
) -> dict[str, Any]:
    return process_image(get_worker_client(), config, prompt_builder, image_path)


def run_strategy(
    argv: list[str] | None,
    description: str,
    default_output_dir: Path,
    prompt_builder: PromptBuilder,
) -> int:
    args = parse_args(argv, description=description, default_output_dir=default_output_dir)
    config = resolve_runtime_config(args)
    selected_images = select_image_paths(
        image_paths=list_image_paths(config.images_root),
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
            item = process_image(client, config, prompt_builder, image_path)
            run_items.append(item)
            upsert_manifest_item(manifest, item)
            write_manifest(config.manifest_path, manifest)
            print_status(item)
    else:
        print(
            f"[PARALLEL] Processing {len(selected_images)} image(s) "
            f"with {config.parallel_workers} worker(s)"
        )
        with ThreadPoolExecutor(
            max_workers=config.parallel_workers,
            thread_name_prefix="gemini-json-ablation",
        ) as executor:
            future_to_image = {
                executor.submit(process_image_in_worker, config, prompt_builder, image_path): image_path
                for image_path in selected_images
            }
            for index, future in enumerate(as_completed(future_to_image), start=1):
                image_path = future_to_image[future]
                item = future.result()
                print(f"[{index}/{len(selected_images)}] Processed {image_path.name}")
                run_items.append(item)
                upsert_manifest_item(manifest, item)
                write_manifest(config.manifest_path, manifest)
                print_status(item)

    counts = Counter(item.get("status", "unknown") for item in run_items)
    print("[SUMMARY]")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    print(f"  manifest: {config.manifest_path}")
    return 0


def print_status(item: dict[str, Any]) -> None:
    status = item["status"]
    if status == "saved":
        print(f"  saved -> {Path(item['json_file']).name}")
    elif "error" in item:
        print(f"  {status}: {item['error']}")
    else:
        print(f"  {status}")


def ground_truth_example(example_id: str) -> PromptExample:
    image_path = MAIN_IMAGES_ROOT / f"{example_id}.png"
    json_path = MAIN_LABELS_ROOT / f"{example_id}.json"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing example image: {image_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"Missing example JSON: {json_path}")
    return PromptExample(example_id=example_id, image_path=image_path, json_path=json_path)
