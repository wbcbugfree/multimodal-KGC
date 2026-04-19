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
DIAGRAM2GRAPH_DIR = SCRIPT_DIR.parent
REPO_ROOT = DIAGRAM2GRAPH_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from common import get_api_key  # noqa: E402
from common.gemini_batch import (  # noqa: E402
    TERMINAL_BATCH_STATES,
    batch_job_snapshot,
    batch_record_error,
    create_batch_job,
    download_result_file,
    extract_batch_response_text,
    read_json,
    read_jsonl,
    result_file_name,
    state_name,
    upload_jsonl_file,
    utc_now_iso,
    utc_timestamp_slug,
    write_json,
    write_jsonl,
)


DEFAULT_MODEL = "gemini-3-flash-preview"
REQUEST_DELAY = 2.0
PARALLEL_WORKERS = 1
MAX_ATTEMPTS = 3
BATCH_POLL_INTERVAL = 60.0
SAMPLE_MODE = "random"
SAMPLE_COUNT = 5
RANDOM_SEED = 42
VALID_IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
FINAL_USER_PROMPT = (
    "Generate RDF/Turtle for this diagram image. "
    "Follow the system instructions exactly and return only RDF/Turtle."
)
EXAMPLE_USER_PROMPT = (
    "Example diagram image with its correct RDF/Turtle output. "
    "Use this example only as a reference for structure and level of detail."
)

DATA_ROOT = DIAGRAM2GRAPH_DIR / "data"
IMAGES_ROOT = DATA_ROOT / "images"
LABELS_ROOT = DATA_ROOT / "labels"
PROMPT_ROOT = DIAGRAM2GRAPH_DIR / "prompt_engineering"
GROUND_TRUTH_ROOT = PROMPT_ROOT / "ground_truth"


@dataclass(frozen=True)
class PromptExample:
    example_id: str
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
    labels_root: Path
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
    batch_action: str | None
    batch_manifest_path: Path
    batch_display_name: str | None
    batch_poll_interval: float


PromptBuilder = Callable[[Path, PromptBuilderContext], PromptPackage]
_THREAD_LOCAL = threading.local()


def build_parser(description: str, default_output_dir: Path) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Gemini model name.")
    parser.add_argument(
        "--sample-mode",
        choices=("all", "random", "ids"),
        default=None,
        help=(
            "Select all images, a deterministic random sample, or explicit ids. "
            "Defaults to random for interactive runs and all for batch submit."
        ),
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
    parser.add_argument(
        "--batch-action",
        choices=("submit", "status", "collect", "wait"),
        default=None,
        help="Use Gemini Batch API instead of the interactive runner.",
    )
    parser.add_argument(
        "--batch-manifest-path",
        type=Path,
        default=None,
        help="Batch job manifest path. Defaults to <output-dir>/batch_jobs/latest_batch_manifest.json.",
    )
    parser.add_argument("--batch-display-name", default=None, help="Optional Gemini Batch API display name.")
    parser.add_argument(
        "--batch-poll-interval",
        type=float,
        default=BATCH_POLL_INTERVAL,
        help="Polling interval in seconds for --batch-action=wait.",
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
    if args.batch_poll_interval <= 0:
        parser.error("--batch-poll-interval must be greater than 0")
    return args


def resolve_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    output_dir = args.output_dir.resolve()
    manifest_path = args.manifest_path.resolve() if args.manifest_path is not None else output_dir / "manifest.json"
    batch_manifest_path = (
        args.batch_manifest_path.resolve()
        if args.batch_manifest_path is not None
        else output_dir / "batch_jobs" / "latest_batch_manifest.json"
    )
    sample_mode = args.sample_mode or ("all" if args.batch_action == "submit" else SAMPLE_MODE)
    return RuntimeConfig(
        model=args.model,
        images_root=IMAGES_ROOT.resolve(),
        labels_root=LABELS_ROOT.resolve(),
        output_dir=output_dir,
        manifest_path=manifest_path,
        request_delay=args.request_delay,
        sample_mode=sample_mode,
        sample_count=args.sample_count,
        seed=args.seed,
        ids=normalize_ids(args.ids),
        parallel_workers=args.parallel_workers,
        skip_existing=args.skip_existing,
        dry_run=args.dry_run,
        batch_action=args.batch_action,
        batch_manifest_path=batch_manifest_path,
        batch_display_name=args.batch_display_name,
        batch_poll_interval=args.batch_poll_interval,
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
        contents.append(types.ModelContent(parts=[types.Part.from_text(text=load_text(example.ttl_path))]))
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
        config=types.GenerateContentConfig(systemInstruction=system_prompt),
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
    label_path: Path,
    ttl_path: Path,
    model: str,
    status: str,
    error: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
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
    if extra_metadata:
        item.update(extra_metadata)
    return item


def write_ttl(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def file_data_part(file_ref: dict[str, Any]) -> dict[str, Any]:
    return {"file_data": {"mime_type": file_ref["mime_type"], "file_uri": file_ref["file_uri"]}}


def build_batch_example_contents(
    examples: list[PromptExample],
    file_refs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    contents: list[dict[str, Any]] = []
    for example in examples:
        example_ref = file_refs[str(example.image_path.resolve())]
        contents.append(
            {
                "role": "user",
                "parts": [{"text": EXAMPLE_USER_PROMPT}, file_data_part(example_ref)],
            }
        )
        contents.append({"role": "model", "parts": [{"text": load_text(example.ttl_path)}]})
    return contents


def build_batch_request_contents(
    user_prompt: str,
    image_path: Path,
    examples: list[PromptExample],
    file_refs: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    image_ref = file_refs[str(image_path.resolve())]
    return [
        *build_batch_example_contents(examples, file_refs),
        {
            "role": "user",
            "parts": [{"text": user_prompt}, file_data_part(image_ref)],
        },
    ]


def build_batch_request_line(key: str, system_prompt: str, contents: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "key": key,
        "request": {
            "contents": contents,
            "system_instruction": {"parts": [{"text": system_prompt}]},
        },
    }


def upload_batch_file_ref(client: genai.Client, path: Path) -> dict[str, Any]:
    uploaded = client.files.upload(
        file=path,
        config=types.UploadFileConfig(display_name=path.name, mime_type=get_mime_type(path)),
    )
    return {
        "path": str(path.resolve()),
        "file_name": uploaded.name,
        "file_uri": uploaded.uri,
        "mime_type": getattr(uploaded, "mime_type", None) or get_mime_type(path),
    }


def selected_image_paths_for_config(config: RuntimeConfig) -> list[Path]:
    return select_image_paths(
        image_paths=list_image_paths(config.images_root),
        sample_mode=config.sample_mode,
        sample_count=config.sample_count,
        seed=config.seed,
        ids=config.ids,
    )


def run_batch_submit(config: RuntimeConfig, prompt_builder: PromptBuilder, selected_images: list[Path]) -> int:
    if config.dry_run:
        print(f"[BATCH-DRY-RUN] Selected {len(selected_images)} image(s)")
        for image_path in selected_images:
            print(image_path.stem)
        print(f"[BATCH-DRY-RUN] Batch manifest would be written to: {config.batch_manifest_path}")
        return 0

    client = create_gemini_client()
    timestamp = utc_timestamp_slug()
    batch_dir = config.batch_manifest_path.parent / timestamp
    request_jsonl_path = batch_dir / "requests.jsonl"
    result_jsonl_path = batch_dir / "results.jsonl"
    file_refs: dict[str, dict[str, Any]] = {}
    request_records: list[dict[str, Any]] = []
    item_records: list[dict[str, Any]] = []

    def ensure_file_ref(path: Path) -> dict[str, Any]:
        key = str(path.resolve())
        if key not in file_refs:
            print(f"[BATCH] Uploading file: {path}")
            file_refs[key] = upload_batch_file_ref(client, path)
        return file_refs[key]

    print(f"[BATCH] Preparing {len(selected_images)} request(s)")
    for index, image_path in enumerate(selected_images, start=1):
        image_id = image_path.stem
        ttl_path = config.output_dir / f"{image_id}.ttl"
        if config.skip_existing and ttl_path.exists():
            print(f"[{index}/{len(selected_images)}] skipping existing {image_path.name}")
            continue
        print(f"[{index}/{len(selected_images)}] staging {image_path.name}")
        prompt_package = prompt_builder(image_path, PromptBuilderContext(model=config.model, client=client))
        ensure_file_ref(image_path)
        for example in prompt_package.examples:
            ensure_file_ref(example.image_path)
        request_records.append(
            build_batch_request_line(
                key=image_id,
                system_prompt=prompt_package.system_prompt,
                contents=build_batch_request_contents(
                    user_prompt=build_user_prompt(image_id),
                    image_path=image_path,
                    examples=prompt_package.examples,
                    file_refs=file_refs,
                ),
            )
        )
        item_records.append(
            {
                "key": image_id,
                "img_id": image_id,
                "source_image": str(image_path.resolve()),
                "json_label": str((config.labels_root / f"{image_id}.json").resolve()),
                "ttl_file": str(ttl_path.resolve()),
                "model": config.model,
                "prompt_metadata": dict(prompt_package.metadata),
            }
        )

    if not request_records:
        print("[BATCH] No requests to submit")
        return 0

    write_jsonl(request_jsonl_path, request_records)
    uploaded_request_file = upload_jsonl_file(
        client,
        request_jsonl_path,
        display_name=f"{config.batch_display_name or 'diagram2graph-batch'}-requests-{timestamp}",
    )
    display_name = config.batch_display_name or f"diagram2graph-{config.output_dir.name}-{timestamp}"
    batch_job = create_batch_job(
        client,
        model=config.model,
        input_file_name=uploaded_request_file.name,
        display_name=display_name,
    )
    batch_manifest = {
        "created_at_utc": utc_now_iso(),
        "dataset": "diagram2graph",
        "model": config.model,
        "batch_job_name": batch_job.name,
        "batch_job_state": state_name(batch_job),
        "batch_display_name": display_name,
        "request_count": len(request_records),
        "request_jsonl_path": str(request_jsonl_path.resolve()),
        "result_jsonl_path": str(result_jsonl_path.resolve()),
        "output_dir": str(config.output_dir.resolve()),
        "manifest_path": str(config.manifest_path.resolve()),
        "input_file": {"file_name": uploaded_request_file.name, "file_uri": uploaded_request_file.uri},
        "uploaded_files": file_refs,
        "items": item_records,
        "batch_job": batch_job_snapshot(batch_job),
    }
    write_json(config.batch_manifest_path, batch_manifest)
    print(f"[BATCH] Submitted job: {batch_job.name}")
    print(f"[BATCH] State: {state_name(batch_job)}")
    print(f"[BATCH] Manifest: {config.batch_manifest_path}")
    return 0


def refresh_batch_manifest_status(config: RuntimeConfig, client: genai.Client) -> dict[str, Any]:
    batch_manifest = read_json(config.batch_manifest_path)
    batch_job_name = str(batch_manifest.get("batch_job_name") or "")
    if not batch_job_name:
        raise ValueError(f"Missing batch_job_name in {config.batch_manifest_path}")
    batch_job = client.batches.get(name=batch_job_name)
    batch_manifest["checked_at_utc"] = utc_now_iso()
    batch_manifest["batch_job_state"] = state_name(batch_job)
    batch_manifest["batch_job"] = batch_job_snapshot(batch_job)
    result_name = result_file_name(batch_job)
    if result_name:
        batch_manifest["result_file_name"] = result_name
    write_json(config.batch_manifest_path, batch_manifest)
    return batch_manifest


def run_batch_status(config: RuntimeConfig) -> int:
    client = create_gemini_client()
    batch_manifest = refresh_batch_manifest_status(config, client)
    print(f"[BATCH] Job: {batch_manifest['batch_job_name']}")
    print(f"[BATCH] State: {batch_manifest['batch_job_state']}")
    if batch_manifest.get("result_file_name"):
        print(f"[BATCH] Result file: {batch_manifest['result_file_name']}")
    error = batch_manifest.get("batch_job", {}).get("error")
    if error:
        print(f"[BATCH] Error: {error}")
    return 0


def build_batch_collected_manifest_item(
    config: RuntimeConfig,
    item_record: dict[str, Any],
    status: str,
    error: str | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    image_path = Path(item_record["source_image"])
    ttl_path = Path(item_record["ttl_file"])
    label_path = Path(item_record.get("json_label", config.labels_root / f"{image_path.stem}.json"))
    metadata = {
        "batch_job_name": item_record.get("batch_job_name"),
        "batch_key": item_record.get("key"),
        "attempt_count": 1,
    }
    if item_record.get("prompt_metadata"):
        metadata.update(item_record["prompt_metadata"])
    if extra_metadata:
        metadata.update(extra_metadata)
    return build_manifest_item(
        image_path=image_path,
        label_path=label_path,
        ttl_path=ttl_path,
        model=config.model,
        status=status,
        error=error,
        extra_metadata=metadata,
    )


def run_batch_collect(config: RuntimeConfig) -> int:
    client = create_gemini_client()
    batch_manifest = refresh_batch_manifest_status(config, client)
    state = str(batch_manifest.get("batch_job_state"))
    if state != "JOB_STATE_SUCCEEDED":
        print(f"[BATCH] Job is not ready for collection: {state}")
        return 2

    result_name = str(batch_manifest.get("result_file_name") or "")
    if not result_name:
        raise ValueError("Batch job succeeded but no result file was reported")

    result_path = Path(batch_manifest["result_jsonl_path"])
    download_result_file(client, file_name=result_name, output_path=result_path)
    result_records = read_jsonl(result_path)
    items_by_key = {str(item["key"]): item for item in batch_manifest.get("items", [])}
    manifest = load_existing_manifest(config.manifest_path)
    run_items: list[dict[str, Any]] = []
    unknown_keys: list[str] = []

    for record in result_records:
        key = str(record.get("key", ""))
        item_record = dict(items_by_key.get(key) or {})
        if not item_record:
            unknown_keys.append(key)
            continue
        item_record["batch_job_name"] = batch_manifest["batch_job_name"]
        error = batch_record_error(record)
        if error:
            item = build_batch_collected_manifest_item(config, item_record, "api_error", error=error)
        else:
            try:
                turtle_text = strip_code_fences(extract_batch_response_text(record))
                parse_error = validate_turtle(turtle_text)
            except Exception as exc:
                turtle_text = ""
                parse_error = str(exc)
            if parse_error is not None:
                item = build_batch_collected_manifest_item(
                    config,
                    item_record,
                    "invalid_ttl",
                    error=parse_error,
                    extra_metadata={"attempt_failures": [build_attempt_failure(1, "invalid_ttl", parse_error, "validation")]},
                )
            else:
                write_ttl(Path(item_record["ttl_file"]), turtle_text)
                item = build_batch_collected_manifest_item(config, item_record, "saved")
        run_items.append(item)
        upsert_manifest_item(manifest, item)

    write_manifest(config.manifest_path, manifest)
    counts = Counter(item.get("status", "unknown") for item in run_items)
    batch_manifest["collected_at_utc"] = utc_now_iso()
    batch_manifest["collect_summary"] = dict(sorted(counts.items()))
    if unknown_keys:
        batch_manifest["unknown_result_keys"] = unknown_keys
    write_json(config.batch_manifest_path, batch_manifest)
    print("[BATCH] Collection summary")
    for status, count in sorted(counts.items()):
        print(f"  {status}: {count}")
    if unknown_keys:
        print(f"  unknown result keys: {len(unknown_keys)}")
    print(f"  manifest: {config.manifest_path}")
    return 0


def run_batch_wait(config: RuntimeConfig) -> int:
    client = create_gemini_client()
    while True:
        batch_manifest = refresh_batch_manifest_status(config, client)
        state = str(batch_manifest.get("batch_job_state"))
        print(f"[BATCH] State: {state}")
        if state in TERMINAL_BATCH_STATES:
            break
        sleep_if_needed(config.batch_poll_interval)
    if state == "JOB_STATE_SUCCEEDED":
        return run_batch_collect(config)
    return 2


def process_image(
    client: genai.Client | None,
    config: RuntimeConfig,
    prompt_builder: PromptBuilder,
    image_path: Path,
) -> dict[str, Any]:
    image_id = image_path.stem
    ttl_path = config.output_dir / f"{image_id}.ttl"
    label_path = config.labels_root / f"{image_id}.json"
    prompt_package: PromptPackage | None = None
    contents: list[types.Content] | None = None
    attempt_failures: list[dict[str, Any]] = []

    if config.skip_existing and ttl_path.exists():
        return build_manifest_item(
            image_path=image_path,
            label_path=label_path,
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
            label_path=label_path,
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
        label_path=label_path,
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

    if config.batch_action == "status":
        return run_batch_status(config)
    if config.batch_action == "collect":
        return run_batch_collect(config)
    if config.batch_action == "wait":
        return run_batch_wait(config)

    selected_images = selected_image_paths_for_config(config)

    if config.batch_action == "submit":
        return run_batch_submit(config, prompt_builder, selected_images)

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
            print_status(item)
    else:
        print(
            f"[PARALLEL] Processing {len(selected_images)} image(s) "
            f"with {config.parallel_workers} worker(s)"
        )
        with ThreadPoolExecutor(
            max_workers=config.parallel_workers,
            thread_name_prefix="gemini-diagram2graph",
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
        print(f"  saved -> {Path(item['ttl_file']).name}")
    elif "error" in item:
        print(f"  {status}: {item['error']}")
    else:
        print(f"  {status}")


def ground_truth_example(example_id: str) -> PromptExample:
    image_path = GROUND_TRUTH_ROOT / f"{example_id}.png"
    ttl_path = GROUND_TRUTH_ROOT / f"{example_id}.ttl"
    if not image_path.exists():
        raise FileNotFoundError(f"Missing example image: {image_path}")
    if not ttl_path.exists():
        raise FileNotFoundError(f"Missing example Turtle: {ttl_path}")
    return PromptExample(example_id=example_id, image_path=image_path, ttl_path=ttl_path)
