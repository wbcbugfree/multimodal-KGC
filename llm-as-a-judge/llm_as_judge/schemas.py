from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


DIRECT_CRITERIA = (
    "relevance",
    "factuality",
    "informativeness",
    "coherence",
    "specificity",
)
DIRECT_SCORE_FIELDS = (*DIRECT_CRITERIA, "overall_score")
PAIRWISE_CHOICES = {"A", "B", "tie"}
CONFIDENCE_CHOICES = {"low", "medium", "high"}


def _require_mapping(payload: Mapping[str, Any] | dict[str, Any]) -> Mapping[str, Any]:
    if not isinstance(payload, Mapping):
        raise ValueError("Judge result must be a JSON object.")
    return payload


def _score(payload: Mapping[str, Any], key: str) -> int:
    if key not in payload:
        raise ValueError(f"Missing required score field: {key}")
    value = payload[key]
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{key} must be an integer from 1 to 5.")
    if value < 1 or value > 5:
        raise ValueError(f"{key} must be an integer from 1 to 5.")
    return value


def _string(payload: Mapping[str, Any], key: str, *, default: str = "") -> str:
    value = payload.get(key, default)
    if value is None:
        return default
    if not isinstance(value, str):
        raise ValueError(f"{key} must be a string.")
    return value.strip()


@dataclass(frozen=True)
class DirectJudgeResult:
    relevance: int
    factuality: int
    informativeness: int
    coherence: int
    specificity: int
    overall_score: int
    major_errors: list[str] = field(default_factory=list)
    reasoning_summary: str = ""

    @property
    def criteria_mean(self) -> float:
        return sum(getattr(self, criterion) for criterion in DIRECT_CRITERIA) / len(DIRECT_CRITERIA)

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | dict[str, Any]) -> "DirectJudgeResult":
        data = _require_mapping(payload)
        major_errors = data.get("major_errors", [])
        if major_errors is None:
            major_errors = []
        if not isinstance(major_errors, list) or not all(isinstance(item, str) for item in major_errors):
            raise ValueError("major_errors must be a list of strings.")
        return cls(
            relevance=_score(data, "relevance"),
            factuality=_score(data, "factuality"),
            informativeness=_score(data, "informativeness"),
            coherence=_score(data, "coherence"),
            specificity=_score(data, "specificity"),
            overall_score=_score(data, "overall_score"),
            major_errors=[item.strip() for item in major_errors if item.strip()],
            reasoning_summary=_string(data, "reasoning_summary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "relevance": self.relevance,
            "factuality": self.factuality,
            "informativeness": self.informativeness,
            "coherence": self.coherence,
            "specificity": self.specificity,
            "overall_score": self.overall_score,
            "criteria_mean": self.criteria_mean,
            "major_errors": list(self.major_errors),
            "reasoning_summary": self.reasoning_summary,
        }


@dataclass(frozen=True)
class PairwiseJudgeResult:
    winner: str
    criterion_preferences: dict[str, str]
    confidence: str
    reasoning_summary: str = ""

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | dict[str, Any]) -> "PairwiseJudgeResult":
        data = _require_mapping(payload)
        winner = _string(data, "winner")
        if winner not in PAIRWISE_CHOICES:
            raise ValueError("winner must be one of A, B, or tie.")
        confidence = _string(data, "confidence", default="medium")
        if confidence not in CONFIDENCE_CHOICES:
            raise ValueError("confidence must be one of low, medium, or high.")

        raw_preferences = data.get("criterion_preferences", {})
        if not isinstance(raw_preferences, Mapping):
            raise ValueError("criterion_preferences must be an object.")
        preferences: dict[str, str] = {}
        for criterion in DIRECT_CRITERIA:
            value = raw_preferences.get(criterion, "tie")
            if value not in PAIRWISE_CHOICES:
                raise ValueError(f"criterion_preferences.{criterion} must be A, B, or tie.")
            preferences[criterion] = value

        return cls(
            winner=winner,
            criterion_preferences=preferences,
            confidence=confidence,
            reasoning_summary=_string(data, "reasoning_summary"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "winner": self.winner,
            "criterion_preferences": dict(self.criterion_preferences),
            "confidence": self.confidence,
            "reasoning_summary": self.reasoning_summary,
        }


DIRECT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        **{
            field_name: {"type": "integer", "minimum": 1, "maximum": 5}
            for field_name in DIRECT_SCORE_FIELDS
        },
        "major_errors": {"type": "array", "items": {"type": "string"}},
        "reasoning_summary": {"type": "string"},
    },
    "required": [*DIRECT_SCORE_FIELDS, "major_errors", "reasoning_summary"],
}


PAIRWISE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "winner": {"type": "string", "enum": sorted(PAIRWISE_CHOICES)},
        "criterion_preferences": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                criterion: {"type": "string", "enum": sorted(PAIRWISE_CHOICES)}
                for criterion in DIRECT_CRITERIA
            },
            "required": list(DIRECT_CRITERIA),
        },
        "confidence": {"type": "string", "enum": sorted(CONFIDENCE_CHOICES)},
        "reasoning_summary": {"type": "string"},
    },
    "required": ["winner", "criterion_preferences", "confidence", "reasoning_summary"],
}
