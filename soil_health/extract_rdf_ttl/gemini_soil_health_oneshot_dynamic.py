from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Callable


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gemini_soil_health_runner_core import (  # noqa: E402
    BATCH_POLL_INTERVAL,
    DEFAULT_MODEL,
    PROMPT_ROOT,
    PromptBuilderContext,
    PromptPackage,
    RuntimeConfig,
    create_gemini_client,
    file_data_part,
    ground_truth_example,
    infer_image_type_from_path,
    load_text,
    load_existing_manifest,
    run_strategy,
    upload_batch_file_ref,
    write_manifest,
)


DYNAMIC_PROMPT_DIR = PROMPT_ROOT / "dynamic_oneshot"
if str(DYNAMIC_PROMPT_DIR) not in sys.path:
    sys.path.insert(0, str(DYNAMIC_PROMPT_DIR))

from categorize_image_type import (  # noqa: E402
    ALLOWED_CATEGORIES,
    CLASSIFICATION_PROMPT,
    classify_image_type,
    normalize_image_type,
)
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


OUTPUT_DIR = SCRIPT_DIR / "oneshot_dynamic"
DESCRIPTION = "Call Gemini on soil-health images with dynamic one-shot prompting and save only valid Turtle outputs."
SYSTEM_PROMPT_BY_TYPE = {
    "figure": DYNAMIC_PROMPT_DIR / "figure_system.md",
    "table": DYNAMIC_PROMPT_DIR / "table_system.md",
}


def image_type_from_classification_manifest(image_path: Path, manifest_path: Path) -> str | None:
    if not manifest_path.exists():
        return None
    manifest = read_json(manifest_path)
    for item in manifest.get("items", []) or []:
        if not isinstance(item, dict) or str(item.get("img_id")) != image_path.stem:
            continue
        status = str(item.get("status", ""))
        image_type = item.get("image_type") or item.get("predicted_image_type")
        if status == "classified" and isinstance(image_type, str):
            return normalize_image_type(image_type)
        return None
    return None


def build_prompt_package(
    image_path: Path,
    context: PromptBuilderContext | None = None,
    classifier: Callable[..., str] = classify_image_type,
) -> PromptPackage:
    effective_context = context or PromptBuilderContext(model=DEFAULT_MODEL, client=None)
    image_type = None
    if effective_context.classification_manifest_path is not None:
        image_type = image_type_from_classification_manifest(
            image_path,
            effective_context.classification_manifest_path,
        )

    if image_type is not None:
        pass
    elif effective_context.require_classification_manifest:
        raise ValueError(
            f"Missing collected image type for {image_path.stem} in "
            f"{effective_context.classification_manifest_path}"
        )
    elif effective_context.dry_run:
        image_type = infer_image_type_from_path(image_path)
    else:
        image_type = classifier(
            image_path,
            model=effective_context.model,
            client=effective_context.client,
        )
    if image_type not in SYSTEM_PROMPT_BY_TYPE:
        raise ValueError(f"Unsupported image type: {image_type}")

    return PromptPackage(
        system_prompt=load_text(SYSTEM_PROMPT_BY_TYPE[image_type]),
        examples=[ground_truth_example(image_type)],
        metadata={"predicted_image_type": image_type},
    )


def build_classification_request_line(key: str, image_ref: dict[str, object]) -> dict[str, object]:
    return {
        "key": key,
        "request": {
            "contents": [
                {
                    "role": "user",
                    "parts": [
                        {"text": CLASSIFICATION_PROMPT},
                        file_data_part(image_ref),
                    ],
                }
            ],
            "generation_config": {
                "responseMimeType": "application/json",
                "responseSchema": {
                    "type": "STRING",
                    "enum": list(ALLOWED_CATEGORIES),
                },
            },
        },
    }


def refresh_classification_manifest_status(config: RuntimeConfig) -> dict[str, object]:
    client = create_gemini_client()
    manifest = read_json(config.classification_manifest_path)
    batch_job_name = str(manifest.get("batch_job_name") or "")
    if not batch_job_name:
        raise ValueError(f"Missing batch_job_name in {config.classification_manifest_path}")
    batch_job = client.batches.get(name=batch_job_name)
    manifest["checked_at_utc"] = utc_now_iso()
    manifest["batch_job_state"] = state_name(batch_job)
    manifest["batch_job"] = batch_job_snapshot(batch_job)
    result_name = result_file_name(batch_job)
    if result_name:
        manifest["result_file_name"] = result_name
    write_json(config.classification_manifest_path, manifest)
    return manifest


def run_classification_batch_submit(config: RuntimeConfig, selected_images: list[Path]) -> int:
    if config.dry_run:
        print(f"[CLASSIFY-BATCH-DRY-RUN] Selected {len(selected_images)} image(s)")
        for image_path in selected_images:
            print(image_path.stem)
        print(
            "[CLASSIFY-BATCH-DRY-RUN] Classification manifest would be written to: "
            f"{config.classification_manifest_path}"
        )
        return 0

    client = create_gemini_client()
    timestamp = utc_timestamp_slug()
    batch_dir = config.classification_manifest_path.parent / f"classification_{timestamp}"
    request_jsonl_path = batch_dir / "requests.jsonl"
    result_jsonl_path = batch_dir / "results.jsonl"
    file_refs: dict[str, dict[str, object]] = {}
    request_records: list[dict[str, object]] = []
    item_records: list[dict[str, object]] = []

    print(f"[CLASSIFY-BATCH] Preparing {len(selected_images)} classification request(s)")
    for index, image_path in enumerate(selected_images, start=1):
        ttl_path = config.output_dir / f"{image_path.stem}.ttl"
        if config.skip_existing and ttl_path.exists():
            print(f"[{index}/{len(selected_images)}] skipping existing {image_path.name}")
            continue
        print(f"[{index}/{len(selected_images)}] staging {image_path.name}")
        image_ref = upload_batch_file_ref(client, image_path)
        file_refs[str(image_path.resolve())] = image_ref
        request_records.append(build_classification_request_line(image_path.stem, image_ref))
        item_records.append(
            {
                "key": image_path.stem,
                "img_id": image_path.stem,
                "source_image": str(image_path.resolve()),
                "model": config.model,
                "status": "pending",
            }
        )

    if not request_records:
        print("[CLASSIFY-BATCH] No requests to submit")
        return 0

    write_jsonl(request_jsonl_path, request_records)
    uploaded_request_file = upload_jsonl_file(
        client,
        request_jsonl_path,
        display_name=f"{config.batch_display_name or 'soil-health-dynamic-classify'}-requests-{timestamp}",
    )
    display_name = config.batch_display_name or f"soil-health-dynamic-classify-{timestamp}"
    batch_job = create_batch_job(
        client,
        model=config.model,
        input_file_name=uploaded_request_file.name,
        display_name=display_name,
    )
    manifest = {
        "created_at_utc": utc_now_iso(),
        "dataset": "soil_health",
        "purpose": "dynamic_oneshot_image_type_classification",
        "model": config.model,
        "allowed_categories": list(ALLOWED_CATEGORIES),
        "batch_job_name": batch_job.name,
        "batch_job_state": state_name(batch_job),
        "batch_display_name": display_name,
        "request_count": len(request_records),
        "request_jsonl_path": str(request_jsonl_path.resolve()),
        "result_jsonl_path": str(result_jsonl_path.resolve()),
        "output_manifest_path": str(config.manifest_path.resolve()),
        "input_file": {
            "file_name": uploaded_request_file.name,
            "file_uri": uploaded_request_file.uri,
        },
        "uploaded_files": file_refs,
        "items": item_records,
        "batch_job": batch_job_snapshot(batch_job),
    }
    write_json(config.classification_manifest_path, manifest)
    print(f"[CLASSIFY-BATCH] Submitted job: {batch_job.name}")
    print(f"[CLASSIFY-BATCH] State: {state_name(batch_job)}")
    print(f"[CLASSIFY-BATCH] Manifest: {config.classification_manifest_path}")
    return 0


def write_classification_summary_to_output_manifest(
    config: RuntimeConfig,
    classification_manifest: dict[str, object],
) -> None:
    output_manifest = load_existing_manifest(config.manifest_path)
    output_manifest["dynamic_classification"] = {
        "updated_at_utc": utc_now_iso(),
        "classification_manifest_path": str(config.classification_manifest_path.resolve()),
        "model": classification_manifest.get("model"),
        "batch_job_name": classification_manifest.get("batch_job_name"),
        "items": [
            {
                "img_id": item.get("img_id"),
                "status": item.get("status"),
                "image_type": item.get("image_type"),
                "error": item.get("error"),
            }
            for item in classification_manifest.get("items", []) or []
            if isinstance(item, dict)
        ],
    }
    write_manifest(config.manifest_path, output_manifest)


def run_classification_batch_collect(config: RuntimeConfig) -> int:
    client = create_gemini_client()
    manifest = refresh_classification_manifest_status(config)
    state = str(manifest.get("batch_job_state"))
    if state != "JOB_STATE_SUCCEEDED":
        print(f"[CLASSIFY-BATCH] Job is not ready for collection: {state}")
        return 2

    result_name = str(manifest.get("result_file_name") or "")
    if not result_name:
        raise ValueError("Classification batch job succeeded but no result file was reported")

    result_path = Path(str(manifest["result_jsonl_path"]))
    download_result_file(client, file_name=result_name, output_path=result_path)
    result_records = read_jsonl(result_path)
    items_by_key = {str(item["key"]): dict(item) for item in manifest.get("items", []) if isinstance(item, dict)}
    unknown_keys: list[str] = []

    for record in result_records:
        key = str(record.get("key", ""))
        item = items_by_key.get(key)
        if item is None:
            unknown_keys.append(key)
            continue
        error = batch_record_error(record)
        if error:
            item["status"] = "api_error"
            item["error"] = error
        else:
            try:
                item["image_type"] = normalize_image_type(extract_batch_response_text(record))
                item["status"] = "classified"
            except Exception as exc:
                item["status"] = "invalid_classification"
                item["error"] = str(exc)

    manifest["items"] = sorted(items_by_key.values(), key=lambda item: str(item.get("img_id", "")))
    manifest["collected_at_utc"] = utc_now_iso()
    manifest["collect_summary"] = dict(
        sorted(Counter(str(item.get("status", "unknown")) for item in manifest["items"]).items())
    )
    if unknown_keys:
        manifest["unknown_result_keys"] = unknown_keys
    write_json(config.classification_manifest_path, manifest)
    write_classification_summary_to_output_manifest(config, manifest)

    print("[CLASSIFY-BATCH] Collection summary")
    for status, count in manifest["collect_summary"].items():
        print(f"  {status}: {count}")
    if unknown_keys:
        print(f"  unknown result keys: {len(unknown_keys)}")
    print(f"  classification manifest: {config.classification_manifest_path}")
    print(f"  output manifest: {config.manifest_path}")
    return 0


def run_classification_batch_wait(config: RuntimeConfig) -> int:
    while True:
        manifest = refresh_classification_manifest_status(config)
        state = str(manifest.get("batch_job_state"))
        print(f"[CLASSIFY-BATCH] State: {state}")
        if state in TERMINAL_BATCH_STATES:
            break
        time.sleep(config.batch_poll_interval or BATCH_POLL_INTERVAL)
    if state == "JOB_STATE_SUCCEEDED":
        return run_classification_batch_collect(config)
    return 2


def run_classification_batch(config: RuntimeConfig, selected_images: list[Path]) -> int:
    action = config.classification_batch_action
    if action == "submit":
        return run_classification_batch_submit(config, selected_images)
    if action == "status":
        manifest = refresh_classification_manifest_status(config)
        print(f"[CLASSIFY-BATCH] Job: {manifest['batch_job_name']}")
        print(f"[CLASSIFY-BATCH] State: {manifest['batch_job_state']}")
        if manifest.get("result_file_name"):
            print(f"[CLASSIFY-BATCH] Result file: {manifest['result_file_name']}")
        error = manifest.get("batch_job", {}).get("error") if isinstance(manifest.get("batch_job"), dict) else None
        if error:
            print(f"[CLASSIFY-BATCH] Error: {error}")
        return 0
    if action == "collect":
        return run_classification_batch_collect(config)
    if action == "wait":
        return run_classification_batch_wait(config)
    raise ValueError(f"Unsupported classification batch action: {action}")


def main(argv: list[str] | None = None) -> int:
    return run_strategy(
        argv=argv,
        description=DESCRIPTION,
        default_output_dir=OUTPUT_DIR,
        prompt_builder=build_prompt_package,
        classification_batch_runner=run_classification_batch,
    )


if __name__ == "__main__":
    raise SystemExit(main())
