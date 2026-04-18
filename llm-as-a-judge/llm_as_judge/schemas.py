from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator


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

PairwiseChoice = Literal["A", "B", "tie"]
ConfidenceChoice = Literal["low", "medium", "high"]


def _payload_to_mapping(payload: Any) -> Mapping[str, Any]:
    if isinstance(payload, BaseModel):
        return payload.model_dump()
    if isinstance(payload, Mapping):
        return payload
    raise ValueError("Judge result must be a JSON object.")


class _StrictJudgeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class DirectJudgeResult(_StrictJudgeModel):
    relevance: int = Field(ge=1, le=5)
    factuality: int = Field(ge=1, le=5)
    informativeness: int = Field(ge=1, le=5)
    coherence: int = Field(ge=1, le=5)
    specificity: int = Field(ge=1, le=5)
    overall_score: int = Field(ge=1, le=5)
    major_errors: list[str]
    reasoning_summary: str

    @property
    def criteria_mean(self) -> float:
        return sum(getattr(self, criterion) for criterion in DIRECT_CRITERIA) / len(DIRECT_CRITERIA)

    @field_validator("major_errors")
    @classmethod
    def _clean_major_errors(cls, value: list[str]) -> list[str]:
        return [item.strip() for item in value if item.strip()]

    @field_validator("reasoning_summary")
    @classmethod
    def _clean_reasoning_summary(cls, value: str) -> str:
        return value.strip()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | dict[str, Any] | BaseModel) -> "DirectJudgeResult":
        if isinstance(payload, cls):
            return payload
        return cls.model_validate(_payload_to_mapping(payload))

    def to_dict(self) -> dict[str, Any]:
        return {**self.model_dump(), "criteria_mean": self.criteria_mean}


class CriterionPreferences(_StrictJudgeModel):
    relevance: PairwiseChoice
    factuality: PairwiseChoice
    informativeness: PairwiseChoice
    coherence: PairwiseChoice
    specificity: PairwiseChoice


class PairwiseJudgeResult(_StrictJudgeModel):
    winner: PairwiseChoice
    criterion_preferences: CriterionPreferences
    confidence: ConfidenceChoice
    reasoning_summary: str

    @field_validator("reasoning_summary")
    @classmethod
    def _clean_reasoning_summary(cls, value: str) -> str:
        return value.strip()

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | dict[str, Any] | BaseModel) -> "PairwiseJudgeResult":
        if isinstance(payload, cls):
            return payload
        return cls.model_validate(_payload_to_mapping(payload))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
