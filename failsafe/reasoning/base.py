"""Reasoning provider interface (Phase 4).

The LLM may: compile a natural-language mission into a MissionSpec, propose candidate operating
configurations for a failure scenario given prior evidence, and analyse results into hypotheses.
It may NOT decide pass/fail, produce runtime configurations during an outage, or return anything
that is executed. Every output is schema-validated; anything that does not validate is rejected.
"""

from __future__ import annotations

from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from failsafe.experiments.schema import OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec


class RejectedOutput(Exception):
    """The model's output did not validate against the required schema."""


# ---------------------------------------------------------------------------------------------
# Structured outputs the model must produce
# ---------------------------------------------------------------------------------------------


class CandidateProposal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: OperatingConfig
    rationale: str = Field(min_length=1, max_length=600)
    expected_tradeoffs: list[str] = Field(default_factory=list, max_length=8)


class ProposalSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    hypothesis: str = Field(min_length=1, max_length=1200)
    candidates: list[CandidateProposal] = Field(min_length=1, max_length=8)
    reasoning_summary: str = Field(default="", max_length=1200)


class Finding(BaseModel):
    model_config = ConfigDict(extra="forbid")

    statement: str = Field(min_length=1, max_length=600)
    evidence: list[str] = Field(default_factory=list, max_length=8)  # experiment ids / config names cited
    confidence: Literal["low", "medium", "high"] = "medium"


class Analysis(BaseModel):
    model_config = ConfigDict(extra="forbid")

    findings: list[Finding] = Field(min_length=1, max_length=10)
    suggested_experiments: list[CandidateProposal] = Field(default_factory=list, max_length=8)
    open_questions: list[str] = Field(default_factory=list, max_length=8)


# ---------------------------------------------------------------------------------------------
# Evidence handed to the model (compact, serialisable)
# ---------------------------------------------------------------------------------------------


class Observation(BaseModel):
    """One evaluated candidate, as the model sees it."""

    model_config = ConfigDict(extra="forbid")

    config: OperatingConfig
    passed: bool
    recall: float | None
    p95_latency_ms: float | None
    precision: float | None
    frames_dropped: float | None
    frames_released: float | None
    cloud_timeouts: float | None
    cloud_rejected: float | None
    detector_ms_mean: float | None
    capability_retained: float


class PlanningContext(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mission: MissionSpec
    scenario: Scenario
    space: dict[str, list]  # the bounded knob catalogue the model may choose from
    observations: list[Observation]
    round_index: int
    max_candidates: int = 5


class ReasoningProvider(Protocol):
    name: str

    def compile_mission(self, text: str) -> MissionSpec: ...

    def propose_configs(self, ctx: PlanningContext) -> ProposalSet: ...

    def analyze_results(self, ctx: PlanningContext) -> Analysis: ...


def validate_or_reject(model_cls, payload) -> BaseModel:
    """Parse a raw JSON object into the schema; any failure is a rejection, never a guess."""
    try:
        return model_cls.model_validate(payload)
    except ValidationError as e:
        raise RejectedOutput(f"{model_cls.__name__}: {e.error_count()} validation error(s): {e.errors()[0]['msg']} at {e.errors()[0]['loc']}") from e
