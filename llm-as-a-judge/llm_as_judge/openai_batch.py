from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .datasets import CandidateRecord, group_by_item
from .judge_core import (
    _include_pair,
    _coerce_mapping,
    _direct_key,
    _order_pair_for_report,
    _pairwise_key,
    _read_json,
    _write_json,
    validate_turtle_text,
)
from .openai_provider import OpenAIJudgeProvider
from .schema_guidance import prompt_with_schema_guidance
from .schemas import DirectJudgeResult, PairwiseJudgeResult


BATCH_ENDPOINT = "/v1/responses"
DEFAULT_BATCH_COMPLETION_WINDOW = "24h"
DEFAULT_BATCH_MAX_FILE_MB = 180.0


@dataclass(frozen=True)
class BatchJudgeJob:
    custom_id: str
    mode: str
    record_a: CandidateRecord
    record_b: CandidateRecord | None = None

    @property
    def item_id(self) -> str:
        return self.record_a.item_id

    def to_manifest_item(self, *, batch_index: int | None = None) -> dict[str, Any]:
        if self.mode == "direct":
            payload = {
                **self.record_a.to_dict(),
                "custom_id": self.custom_id,
                "mode": self.mode,
                "status": "submitted",
            }
        else:
            if self.record_b is None:
                raise ValueError("Pairwise batch job is missing record_b.")
            payload = {
                "dataset": self.record_a.dataset,
                "item_id": self.record_a.item_id,
                "image_path": str(self.record_a.image_path),
                "strategy_a": self.record_a.strategy,
                "strategy_b": self.record_b.strategy,
                "ttl_a_path": str(self.record_a.ttl_path),
                "ttl_b_path": str(self.record_b.ttl_path),
                "custom_id": self.custom_id,
                "mode": self.mode,
                "status": "submitted",
            }
        if batch_index is not None:
            payload["batch_index"] = batch_index
        return payload


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _api_object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, Mapping):
        return dict(obj)
    model_dump = getattr(obj, "model_dump", None)
    if callable(model_dump):
        return model_dump(mode="json")
    to_dict = getattr(obj, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    keys = (
        "id",
        "object",
        "endpoint",
        "errors",
        "input_file_id",
        "completion_window",
        "status",
        "output_file_id",
        "error_file_id",
        "created_at",
        "in_progress_at",
        "expires_at",
        "completed_at",
        "failed_at",
        "expired_at",
        "request_counts",
        "metadata",
    )
    return {key: getattr(obj, key) for key in keys if hasattr(obj, key)}


def _file_content_text(file_response: Any) -> str:
    text = getattr(file_response, "text", None)
    if isinstance(text, str):
        return text
    read = getattr(file_response, "read", None)
    if callable(read):
        content = read()
        if isinstance(content, bytes):
            return content.decode("utf-8")
        if isinstance(content, str):
            return content
    content = getattr(file_response, "content", None)
    if isinstance(content, bytes):
        return content.decode("utf-8")
    if isinstance(content, str):
        return content
    raise ValueError("Could not read OpenAI file content as text.")


def _load_existing_direct_keys(output_path: Path) -> set[tuple[str | None, str | None, str | None]]:
    report = _read_json(output_path)
    return {
        (item.get("dataset"), item.get("strategy"), item.get("item_id"))
        for item in report.get("items", [])
        if isinstance(item, Mapping)
    }


def _load_existing_pairwise_keys(output_path: Path) -> set[tuple[str | None, str | None, str | None, str | None]]:
    report = _read_json(output_path)
    return {
        (item.get("dataset"), item.get("item_id"), item.get("strategy_a"), item.get("strategy_b"))
        for item in report.get("items", [])
        if isinstance(item, Mapping)
    }


def build_batch_jobs(
    records: Sequence[CandidateRecord],
    *,
    modes: Sequence[str],
    output_dir: Path,
    skip_existing: bool,
    pairing_mode: str = "all",
) -> list[BatchJudgeJob]:
    jobs: list[BatchJudgeJob] = []
    direct_existing = _load_existing_direct_keys(output_dir / "direct_judge_results.json") if skip_existing else set()
    pairwise_existing = (
        _load_existing_pairwise_keys(output_dir / "pairwise_judge_results.json") if skip_existing else set()
    )

    if "direct" in modes:
        for record in records:
            if skip_existing and _direct_key(record) in direct_existing:
                continue
            jobs.append(BatchJudgeJob(custom_id=f"judge-direct-{len(jobs) + 1:06d}", mode="direct", record_a=record))

    if "pairwise" in modes:
        for item_id, item_records in group_by_item(records).items():
            for record_a, record_b in combinations(item_records, 2):
                if not _include_pair(record_a, record_b, pairing_mode):
                    continue
                record_a, record_b = _order_pair_for_report(record_a, record_b, pairing_mode)
                if skip_existing and _pairwise_key(record_a, record_b) in pairwise_existing:
                    continue
                jobs.append(
                    BatchJudgeJob(
                        custom_id=f"judge-pairwise-{len(jobs) + 1:06d}",
                        mode="pairwise",
                        record_a=record_a,
                        record_b=record_b,
                    )
                )
    return jobs


def _error_report_item(job: BatchJudgeJob, error: str) -> dict[str, Any]:
    item = job.to_manifest_item()
    item["status"] = "error"
    item["error"] = error
    item.pop("custom_id", None)
    item.pop("mode", None)
    item.pop("batch_index", None)
    return item


def _write_preflight_errors(output_dir: Path, jobs: Sequence[BatchJudgeJob]) -> list[dict[str, Any]]:
    errors: list[dict[str, Any]] = []
    direct_items: list[dict[str, Any]] = []
    pairwise_items: list[dict[str, Any]] = []
    for job in jobs:
        if job.mode == "direct":
            ttl_text = job.record_a.ttl_path.read_text(encoding="utf-8")
            parse_error = validate_turtle_text(ttl_text)
            if parse_error is not None:
                item = _error_report_item(job, f"Candidate TTL is not parseable: {parse_error}")
                direct_items.append(item)
                errors.append({"custom_id": job.custom_id, **item})
        else:
            assert job.record_b is not None
            for label, record in (("A", job.record_a), ("B", job.record_b)):
                ttl_text = record.ttl_path.read_text(encoding="utf-8")
                parse_error = validate_turtle_text(ttl_text)
                if parse_error is not None:
                    item = _error_report_item(job, f"Candidate {label} TTL is not parseable: {parse_error}")
                    pairwise_items.append(item)
                    errors.append({"custom_id": job.custom_id, **item})
                    break

    if direct_items:
        _upsert_report_items(
            output_dir / "direct_judge_results.json",
            mode="direct",
            execution_mode="batch",
            items=direct_items,
        )
    if pairwise_items:
        _upsert_report_items(
            output_dir / "pairwise_judge_results.json",
            mode="pairwise",
            execution_mode="batch",
            items=pairwise_items,
        )
    return errors


def _valid_jobs(jobs: Sequence[BatchJudgeJob], preflight_errors: Sequence[Mapping[str, Any]]) -> list[BatchJudgeJob]:
    invalid_custom_ids = {str(item.get("custom_id")) for item in preflight_errors}
    return [job for job in jobs if job.custom_id not in invalid_custom_ids]


def _request_line(provider: OpenAIJudgeProvider, job: BatchJudgeJob, direct_prompt: str, pairwise_prompt: str) -> str:
    if job.mode == "direct":
        body = provider.build_direct_request_body(
            image_path=job.record_a.image_path,
            ttl_text=job.record_a.ttl_path.read_text(encoding="utf-8"),
            prompt_text=prompt_with_schema_guidance(direct_prompt, job.record_a.dataset),
        )
    else:
        if job.record_b is None:
            raise ValueError("Pairwise batch job is missing record_b.")
        body = provider.build_pairwise_request_body(
            image_path=job.record_a.image_path,
            ttl_a=job.record_a.ttl_path.read_text(encoding="utf-8"),
            ttl_b=job.record_b.ttl_path.read_text(encoding="utf-8"),
            prompt_text=prompt_with_schema_guidance(pairwise_prompt, job.record_a.dataset),
            strategy_a=job.record_a.strategy,
            strategy_b=job.record_b.strategy,
        )
    return json.dumps(
        {
            "custom_id": job.custom_id,
            "method": "POST",
            "url": BATCH_ENDPOINT,
            "body": body,
        },
        ensure_ascii=False,
    )


def _chunk_request_lines(
    provider: OpenAIJudgeProvider,
    jobs: Sequence[BatchJudgeJob],
    *,
    direct_prompt: str,
    pairwise_prompt: str,
    max_file_mb: float,
) -> list[tuple[list[BatchJudgeJob], list[str], int]]:
    max_bytes = int(max_file_mb * 1024 * 1024)
    chunks: list[tuple[list[BatchJudgeJob], list[str], int]] = []
    chunk_jobs: list[BatchJudgeJob] = []
    chunk_lines: list[str] = []
    chunk_bytes = 0
    for job in jobs:
        line = _request_line(provider, job, direct_prompt, pairwise_prompt) + "\n"
        line_bytes = len(line.encode("utf-8"))
        if line_bytes > max_bytes:
            raise ValueError(
                f"One batch request for {job.custom_id} is {line_bytes} bytes, "
                f"larger than the configured {max_file_mb} MB limit."
            )
        if chunk_lines and chunk_bytes + line_bytes > max_bytes:
            chunks.append((chunk_jobs, chunk_lines, chunk_bytes))
            chunk_jobs = []
            chunk_lines = []
            chunk_bytes = 0
        chunk_jobs.append(job)
        chunk_lines.append(line)
        chunk_bytes += line_bytes
    if chunk_lines:
        chunks.append((chunk_jobs, chunk_lines, chunk_bytes))
    return chunks


def _upsert_report_items(
    output_path: Path,
    *,
    mode: str,
    execution_mode: str,
    items: Iterable[dict[str, Any]],
) -> dict[str, Any]:
    report = _read_json(output_path)
    existing = list(report.get("items", []))
    index: dict[tuple[Any, ...], int] = {}
    for item_index, item in enumerate(existing):
        if not isinstance(item, Mapping):
            continue
        if mode == "direct":
            key = (item.get("dataset"), item.get("strategy"), item.get("item_id"))
        else:
            key = (item.get("dataset"), item.get("item_id"), item.get("strategy_a"), item.get("strategy_b"))
        index[key] = item_index

    for item in items:
        if mode == "direct":
            key = (item.get("dataset"), item.get("strategy"), item.get("item_id"))
        else:
            key = (item.get("dataset"), item.get("item_id"), item.get("strategy_a"), item.get("strategy_b"))
        if key in index:
            existing[index[key]] = item
        else:
            index[key] = len(existing)
            existing.append(item)

    updated = {
        "generated_at_utc": _utc_now(),
        "mode": mode,
        "execution_mode": execution_mode,
        "items": existing,
    }
    _write_json(output_path, updated)
    return updated


def _report_item_from_job(job: BatchJudgeJob, result: Mapping[str, Any]) -> dict[str, Any]:
    if job.mode == "direct":
        return {
            **job.record_a.to_dict(),
            "status": "success",
            "scores": DirectJudgeResult.from_mapping(_coerce_mapping(result)).to_dict(),
        }
    if job.record_b is None:
        raise ValueError("Pairwise batch job is missing record_b.")
    return {
        "dataset": job.record_a.dataset,
        "item_id": job.record_a.item_id,
        "image_path": str(job.record_a.image_path),
        "strategy_a": job.record_a.strategy,
        "strategy_b": job.record_b.strategy,
        "ttl_a_path": str(job.record_a.ttl_path),
        "ttl_b_path": str(job.record_b.ttl_path),
        "status": "success",
        "judge": PairwiseJudgeResult.from_mapping(_coerce_mapping(result)).to_dict(),
    }


def _report_error_from_job(job: BatchJudgeJob, error: str) -> dict[str, Any]:
    item = _error_report_item(job, error)
    return item


class OpenAIBatchJudgeRunner:
    def __init__(
        self,
        *,
        provider: OpenAIJudgeProvider,
        output_dir: Path,
        manifest_path: Path,
        direct_prompt: str,
        pairwise_prompt: str,
        completion_window: str = DEFAULT_BATCH_COMPLETION_WINDOW,
        max_file_mb: float = DEFAULT_BATCH_MAX_FILE_MB,
    ) -> None:
        self.provider = provider
        self.output_dir = output_dir
        self.manifest_path = manifest_path
        self.direct_prompt = direct_prompt
        self.pairwise_prompt = pairwise_prompt
        self.completion_window = completion_window
        self.max_file_mb = max_file_mb

    def _client(self) -> Any:
        return self.provider._client_instance()

    def submit(
        self,
        *,
        dataset: str,
        records: Sequence[CandidateRecord],
        modes: Sequence[str],
        strategy_selection: Mapping[str, Any] | None,
        skip_existing: bool,
        dry_run: bool,
        validation_design: str | None = None,
        pairing_mode: str = "all",
    ) -> dict[str, Any]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        all_jobs = build_batch_jobs(
            records,
            modes=modes,
            output_dir=self.output_dir,
            skip_existing=skip_existing,
            pairing_mode=pairing_mode,
        )
        preflight_errors = _write_preflight_errors(self.output_dir, all_jobs) if not dry_run else []
        jobs = _valid_jobs(all_jobs, preflight_errors)
        chunks = _chunk_request_lines(
            self.provider,
            jobs,
            direct_prompt=self.direct_prompt,
            pairwise_prompt=self.pairwise_prompt,
            max_file_mb=self.max_file_mb,
        )

        manifest: dict[str, Any] = {
            "dataset": dataset,
            "generated_at_utc": _utc_now(),
            "execution_mode": "batch",
            "provider": "openai",
            "model": self.provider.model,
            "endpoint": BATCH_ENDPOINT,
            "completion_window": self.completion_window,
            "modes": list(modes),
            "validation_design": validation_design,
            "pairing_mode": pairing_mode,
            "strategy_selection": strategy_selection,
            "dry_run": dry_run,
            "max_file_mb": self.max_file_mb,
            "job_count": len(jobs),
            "preflight_error_count": len(preflight_errors),
            "preflight_errors": preflight_errors,
            "batches": [],
            "items": [],
        }

        batch_input_dir = self.output_dir / "batch_inputs"
        batch_input_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        client = None if dry_run else self._client()

        for batch_index, (chunk_jobs, chunk_lines, chunk_bytes) in enumerate(chunks, start=1):
            input_path = batch_input_dir / f"{dataset}_judge_{timestamp}_{batch_index:03d}.jsonl"
            if not dry_run:
                input_path.write_text("".join(chunk_lines), encoding="utf-8")
                with input_path.open("rb") as handle:
                    input_file = client.files.create(file=handle, purpose="batch")
                batch = client.batches.create(
                    input_file_id=input_file.id,
                    endpoint=BATCH_ENDPOINT,
                    completion_window=self.completion_window,
                    metadata={
                        "dataset": dataset,
                        "modes": ",".join(modes),
                        "source": "multimodal-KGC llm-as-a-judge",
                    },
                )
                input_file_payload = _api_object_to_dict(input_file)
                batch_payload = _api_object_to_dict(batch)
            else:
                input_file_payload = None
                batch_payload = {
                    "id": None,
                    "status": "dry_run",
                    "input_file_id": None,
                    "output_file_id": None,
                    "error_file_id": None,
                }

            manifest["batches"].append(
                {
                    "batch_index": batch_index,
                    "input_jsonl_path": str(input_path),
                    "request_count": len(chunk_jobs),
                    "input_bytes": chunk_bytes,
                    "input_file": input_file_payload,
                    "batch": batch_payload,
                }
            )
            manifest["items"].extend(job.to_manifest_item(batch_index=batch_index) for job in chunk_jobs)

        _write_json(self.manifest_path, manifest)
        return manifest

    def _load_manifest(self) -> dict[str, Any]:
        if not self.manifest_path.exists():
            raise FileNotFoundError(f"Batch manifest not found: {self.manifest_path}")
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object in batch manifest: {self.manifest_path}")
        return data

    def status(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        client = self._client()
        for batch_entry in manifest.get("batches", []):
            batch_payload = batch_entry.get("batch") or {}
            batch_id = batch_payload.get("id")
            if not batch_id:
                continue
            batch_entry["batch"] = _api_object_to_dict(client.batches.retrieve(batch_id))
            batch_entry["last_status_check_utc"] = _utc_now()
        manifest["last_status_check_utc"] = _utc_now()
        _write_json(self.manifest_path, manifest)
        return manifest

    def cancel(self) -> dict[str, Any]:
        manifest = self._load_manifest()
        client = self._client()
        for batch_entry in manifest.get("batches", []):
            batch_payload = batch_entry.get("batch") or {}
            batch_id = batch_payload.get("id")
            if not batch_id:
                continue
            batch_entry["batch"] = _api_object_to_dict(client.batches.cancel(batch_id))
            batch_entry["cancelled_at_utc"] = _utc_now()
        manifest["cancelled_at_utc"] = _utc_now()
        _write_json(self.manifest_path, manifest)
        return manifest

    def collect(self) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, Any]]:
        manifest = self.status()
        client = self._client()
        custom_id_to_item = {
            str(item.get("custom_id")): item for item in manifest.get("items", []) if isinstance(item, Mapping)
        }
        direct_items: list[dict[str, Any]] = []
        pairwise_items: list[dict[str, Any]] = []

        for batch_entry in manifest.get("batches", []):
            batch_payload = batch_entry.get("batch") or {}
            status = batch_payload.get("status")
            if status not in {"completed", "expired", "failed"}:
                continue

            for file_key, source_label in (("output_file_id", "output"), ("error_file_id", "error_file")):
                file_id = batch_payload.get(file_key)
                if not file_id:
                    continue
                file_text = _file_content_text(client.files.content(file_id))
                local_path = self.output_dir / "batch_outputs" / f"{batch_payload.get('id')}_{source_label}.jsonl"
                local_path.parent.mkdir(parents=True, exist_ok=True)
                local_path.write_text(file_text, encoding="utf-8")
                batch_entry[f"{source_label}_jsonl_path"] = str(local_path)
                for line in file_text.splitlines():
                    if not line.strip():
                        continue
                    payload = json.loads(line)
                    custom_id = str(payload.get("custom_id"))
                    manifest_item = custom_id_to_item.get(custom_id)
                    if manifest_item is None:
                        continue
                    job = _manifest_item_to_job(manifest_item)
                    error = payload.get("error")
                    response = payload.get("response") or {}
                    response_body = response.get("body") if isinstance(response, Mapping) else None
                    status_code = response.get("status_code") if isinstance(response, Mapping) else None
                    if error is not None:
                        item = _report_error_from_job(job, json.dumps(error, ensure_ascii=False))
                    elif status_code != 200:
                        item = _report_error_from_job(job, f"Batch response status code: {status_code}")
                    elif not isinstance(response_body, Mapping):
                        item = _report_error_from_job(job, "Batch response did not contain a response body object.")
                    else:
                        try:
                            if job.mode == "direct":
                                result = OpenAIJudgeProvider.parse_direct_response_body(response_body)
                            else:
                                result = OpenAIJudgeProvider.parse_pairwise_response_body(response_body)
                            item = _report_item_from_job(job, result)
                        except Exception as exc:
                            item = _report_error_from_job(job, f"Could not parse structured judge response: {exc}")

                    if job.mode == "direct":
                        direct_items.append(item)
                    else:
                        pairwise_items.append(item)
                    manifest_item["status"] = item.get("status")
                    if item.get("error"):
                        manifest_item["error"] = item.get("error")

        direct_report = (
            _upsert_report_items(
                self.output_dir / "direct_judge_results.json",
                mode="direct",
                execution_mode="batch",
                items=direct_items,
            )
            if direct_items
            else None
        )
        pairwise_report = (
            _upsert_report_items(
                self.output_dir / "pairwise_judge_results.json",
                mode="pairwise",
                execution_mode="batch",
                items=pairwise_items,
            )
            if pairwise_items
            else None
        )
        manifest["collected_at_utc"] = _utc_now()
        _write_json(self.manifest_path, manifest)
        return direct_report, pairwise_report, manifest


def _manifest_item_to_job(item: Mapping[str, Any]) -> BatchJudgeJob:
    mode = str(item["mode"])
    if mode == "direct":
        record = CandidateRecord(
            dataset=str(item["dataset"]),
            strategy=str(item["strategy"]),
            item_id=str(item["item_id"]),
            ttl_path=Path(str(item["ttl_path"])),
            image_path=Path(str(item["image_path"])),
        )
        return BatchJudgeJob(custom_id=str(item["custom_id"]), mode=mode, record_a=record)
    record_a = CandidateRecord(
        dataset=str(item["dataset"]),
        strategy=str(item["strategy_a"]),
        item_id=str(item["item_id"]),
        ttl_path=Path(str(item["ttl_a_path"])),
        image_path=Path(str(item["image_path"])),
    )
    record_b = CandidateRecord(
        dataset=str(item["dataset"]),
        strategy=str(item["strategy_b"]),
        item_id=str(item["item_id"]),
        ttl_path=Path(str(item["ttl_b_path"])),
        image_path=Path(str(item["image_path"])),
    )
    return BatchJudgeJob(custom_id=str(item["custom_id"]), mode=mode, record_a=record_a, record_b=record_b)
