from __future__ import annotations

import json
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from google.genai import types


TERMINAL_BATCH_STATES = {
    "JOB_STATE_SUCCEEDED",
    "JOB_STATE_FAILED",
    "JOB_STATE_CANCELLED",
    "JOB_STATE_EXPIRED",
}


def utc_timestamp_slug() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, Enum):
        return value.name
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list | tuple):
        return [jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return jsonable(model_dump(mode="json", by_alias=True, exclude_none=True))
    return str(value)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False))
            handle.write("\n")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise ValueError(f"Expected JSON object on line {line_number}: {path}")
            records.append(payload)
    return records


def state_name(batch_job: Any) -> str:
    state = getattr(batch_job, "state", None)
    if isinstance(batch_job, dict):
        state = batch_job.get("state")
    name = getattr(state, "name", None)
    if isinstance(name, str):
        return name
    if isinstance(state, str):
        return state
    return str(state)


def result_file_name(batch_job: Any) -> str | None:
    dest = getattr(batch_job, "dest", None)
    if isinstance(batch_job, dict):
        dest = batch_job.get("dest")
    if dest is None:
        return None
    if isinstance(dest, dict):
        value = dest.get("fileName") or dest.get("file_name")
    else:
        value = getattr(dest, "file_name", None) or getattr(dest, "fileName", None)
    return str(value) if value else None


def job_error_text(batch_job: Any) -> str | None:
    error = getattr(batch_job, "error", None)
    if isinstance(batch_job, dict):
        error = batch_job.get("error")
    if error is None:
        return None
    message = getattr(error, "message", None)
    if isinstance(error, dict):
        message = error.get("message") or error.get("details")
    return str(message or error)


def batch_job_snapshot(batch_job: Any) -> dict[str, Any]:
    return {
        "name": getattr(batch_job, "name", None) if not isinstance(batch_job, dict) else batch_job.get("name"),
        "state": state_name(batch_job),
        "model": getattr(batch_job, "model", None) if not isinstance(batch_job, dict) else batch_job.get("model"),
        "dest_file_name": result_file_name(batch_job),
        "error": job_error_text(batch_job),
        "raw": jsonable(batch_job),
    }


def upload_jsonl_file(client: Any, path: Path, display_name: str) -> Any:
    return client.files.upload(
        file=path,
        config=types.UploadFileConfig(display_name=display_name, mime_type="application/jsonl"),
    )


def create_batch_job(client: Any, *, model: str, input_file_name: str, display_name: str) -> Any:
    return client.batches.create(
        model=model,
        src=input_file_name,
        config=types.CreateBatchJobConfig(display_name=display_name),
    )


def download_result_file(client: Any, *, file_name: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    content = client.files.download(file=file_name)
    if isinstance(content, str):
        data = content.encode("utf-8")
    else:
        data = bytes(content)
    output_path.write_bytes(data)
    return output_path


def batch_record_error(record: dict[str, Any]) -> str | None:
    error = record.get("error")
    if error is None:
        return None
    if isinstance(error, dict):
        return str(error.get("message") or error.get("details") or error)
    return str(error)


def extract_batch_response_text(record: dict[str, Any]) -> str:
    response = record.get("response")
    if not isinstance(response, dict):
        raise ValueError("Batch result record does not contain a response object")
    parts: list[str] = []
    for candidate in response.get("candidates", []) or []:
        content = candidate.get("content", {}) if isinstance(candidate, dict) else {}
        for part in content.get("parts", []) or []:
            if isinstance(part, dict):
                text = part.get("text")
                if isinstance(text, str) and text.strip():
                    parts.append(text.strip())
    if parts:
        return "\n".join(parts)
    raise ValueError("Batch response did not contain text content")
