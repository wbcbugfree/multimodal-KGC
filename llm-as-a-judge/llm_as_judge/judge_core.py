from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel
from rdflib import Graph

from .datasets import CandidateRecord, group_by_item
from .schemas import DirectJudgeResult, PairwiseJudgeResult


class JudgeProvider(Protocol):
    def judge_direct(self, *, image_path: Path, ttl_text: str, prompt_text: str) -> Mapping[str, Any] | BaseModel | str:
        ...

    def judge_pairwise(
        self,
        *,
        image_path: Path,
        ttl_a: str,
        ttl_b: str,
        prompt_text: str,
        strategy_a: str,
        strategy_b: str,
    ) -> Mapping[str, Any] | BaseModel | str:
        ...


def prompt_path(name: str) -> Path:
    return Path(__file__).resolve().parent / "prompts" / name


def load_prompt(name: str) -> str:
    return prompt_path(name).read_text(encoding="utf-8")


def validate_turtle_text(ttl_text: str) -> str | None:
    try:
        Graph().parse(data=ttl_text, format="turtle")
    except Exception as exc:  # rdflib exposes parser-specific exception types.
        return f"Bad syntax: {exc}"
    return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"items": []}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {path}")
    data.setdefault("items", [])
    return data


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _coerce_mapping(payload: Mapping[str, Any] | BaseModel | str) -> Mapping[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if isinstance(payload, Mapping):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if text.startswith("```"):
            lines = text.splitlines()
            if lines and lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].startswith("```"):
                lines = lines[:-1]
            text = "\n".join(lines).strip()
        parsed = json.loads(text)
        if not isinstance(parsed, Mapping):
            raise ValueError("Judge response JSON must be an object.")
        return parsed
    raise ValueError("Judge provider returned an unsupported response type.")


def _direct_key(record: CandidateRecord) -> tuple[str, str, str]:
    return (record.dataset, record.strategy, record.item_id)


def _pairwise_key(record_a: CandidateRecord, record_b: CandidateRecord) -> tuple[str, str, str, str]:
    first, second = sorted([record_a.strategy, record_b.strategy])
    return (record_a.dataset, record_a.item_id, first, second)


class JudgeRunner:
    def __init__(
        self,
        *,
        provider: JudgeProvider,
        results_root: Path,
        max_retries: int = 2,
        direct_prompt: str | None = None,
        pairwise_prompt: str | None = None,
    ) -> None:
        self.provider = provider
        self.results_root = results_root.resolve()
        self.max_retries = max_retries
        self.direct_prompt = direct_prompt or load_prompt("direct_image_to_kg_judge.md")
        self.pairwise_prompt = pairwise_prompt or load_prompt("pairwise_image_to_kg_judge.md")

    def _attempt_direct(self, record: CandidateRecord) -> DirectJudgeResult:
        ttl_text = record.ttl_path.read_text(encoding="utf-8")
        parse_error = validate_turtle_text(ttl_text)
        if parse_error is not None:
            raise ValueError(f"Candidate TTL is not parseable: {parse_error}")
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                raw = self.provider.judge_direct(
                    image_path=record.image_path,
                    ttl_text=ttl_text,
                    prompt_text=self.direct_prompt,
                )
                return DirectJudgeResult.from_mapping(_coerce_mapping(raw))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Direct judge failed after retries: {last_error}")

    def _attempt_pairwise(self, record_a: CandidateRecord, record_b: CandidateRecord) -> PairwiseJudgeResult:
        ttl_a = record_a.ttl_path.read_text(encoding="utf-8")
        ttl_b = record_b.ttl_path.read_text(encoding="utf-8")
        for label, ttl_text in (("A", ttl_a), ("B", ttl_b)):
            parse_error = validate_turtle_text(ttl_text)
            if parse_error is not None:
                raise ValueError(f"Candidate {label} TTL is not parseable: {parse_error}")
        last_error: Exception | None = None
        for _attempt in range(self.max_retries + 1):
            try:
                raw = self.provider.judge_pairwise(
                    image_path=record_a.image_path,
                    ttl_a=ttl_a,
                    ttl_b=ttl_b,
                    prompt_text=self.pairwise_prompt,
                    strategy_a=record_a.strategy,
                    strategy_b=record_b.strategy,
                )
                return PairwiseJudgeResult.from_mapping(_coerce_mapping(raw))
            except Exception as exc:
                last_error = exc
        raise RuntimeError(f"Pairwise judge failed after retries: {last_error}")

    def run_direct(
        self,
        records: list[CandidateRecord],
        *,
        output_path: Path,
        skip_existing: bool = False,
    ) -> dict[str, Any]:
        report = _read_json(output_path)
        existing_keys = {
            (item.get("dataset"), item.get("strategy"), item.get("item_id"))
            for item in report.get("items", [])
            if isinstance(item, Mapping)
        }
        items = list(report.get("items", []))
        for record in records:
            key = _direct_key(record)
            if skip_existing and key in existing_keys:
                continue
            try:
                result = self._attempt_direct(record)
                items.append(
                    {
                        **record.to_dict(),
                        "status": "success",
                        "scores": result.to_dict(),
                    }
                )
            except Exception as exc:
                items.append(
                    {
                        **record.to_dict(),
                        "status": "error",
                        "error": str(exc),
                    }
                )
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "direct",
            "items": items,
        }
        _write_json(output_path, report)
        return report

    def run_pairwise(
        self,
        records: list[CandidateRecord],
        *,
        output_path: Path,
        skip_existing: bool = False,
    ) -> dict[str, Any]:
        report = _read_json(output_path)
        existing_keys = {
            (item.get("dataset"), item.get("item_id"), item.get("strategy_a"), item.get("strategy_b"))
            for item in report.get("items", [])
            if isinstance(item, Mapping)
        }
        items = list(report.get("items", []))
        for item_id, item_records in group_by_item(records).items():
            for record_a, record_b in combinations(item_records, 2):
                strategy_a, strategy_b = sorted([record_a.strategy, record_b.strategy])
                if record_a.strategy != strategy_a:
                    record_a, record_b = record_b, record_a
                key = _pairwise_key(record_a, record_b)
                if skip_existing and key in existing_keys:
                    continue
                try:
                    result = self._attempt_pairwise(record_a, record_b)
                    items.append(
                        {
                            "dataset": record_a.dataset,
                            "item_id": item_id,
                            "image_path": str(record_a.image_path),
                            "strategy_a": record_a.strategy,
                            "strategy_b": record_b.strategy,
                            "ttl_a_path": str(record_a.ttl_path),
                            "ttl_b_path": str(record_b.ttl_path),
                            "status": "success",
                            "judge": result.to_dict(),
                        }
                    )
                except Exception as exc:
                    items.append(
                        {
                            "dataset": record_a.dataset,
                            "item_id": item_id,
                            "image_path": str(record_a.image_path),
                            "strategy_a": record_a.strategy,
                            "strategy_b": record_b.strategy,
                            "ttl_a_path": str(record_a.ttl_path),
                            "ttl_b_path": str(record_b.ttl_path),
                            "status": "error",
                            "error": str(exc),
                        }
                    )
        report = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "mode": "pairwise",
            "items": items,
        }
        _write_json(output_path, report)
        return report
