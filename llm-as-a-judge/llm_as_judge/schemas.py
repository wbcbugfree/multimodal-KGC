from __future__ import annotations

from typing import Any, Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field


DIRECT_CRITERIA = (
    "visual_grounding",
    "structural_fidelity",
    "semantic_correctness",
    "completeness",
    "schema_compliance",
    "non_hallucination",
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
    visual_grounding: int = Field(ge=1, le=5)
    structural_fidelity: int = Field(ge=1, le=5)
    semantic_correctness: int = Field(ge=1, le=5)
    completeness: int = Field(ge=1, le=5)
    schema_compliance: int = Field(ge=1, le=5)
    non_hallucination: int = Field(ge=1, le=5)
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
    visual_grounding: PairwiseChoice
    structural_fidelity: PairwiseChoice
    semantic_correctness: PairwiseChoice
    completeness: PairwiseChoice
    schema_compliance: PairwiseChoice
    non_hallucination: PairwiseChoice


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
