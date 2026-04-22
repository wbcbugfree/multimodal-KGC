from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


DIRECT_CRITERIA = (
    "relevance",
    "factuality",
    "informativeness",
    "coherence",
    "specificity",
)
DIRECT_SCORE_FIELDS = (*DIRECT_CRITERIA, "overall_score")
PAIRWISE_CHOICES = {"A", "B", "tie"}

PairwiseChoice = Literal["A", "B", "tie"]


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

    @property
    def criteria_mean(self) -> float:
        return sum(getattr(self, criterion) for criterion in DIRECT_CRITERIA) / len(DIRECT_CRITERIA)

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

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any] | dict[str, Any] | BaseModel) -> "PairwiseJudgeResult":
        if isinstance(payload, cls):
            return payload
        return cls.model_validate(_payload_to_mapping(payload))

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump()
