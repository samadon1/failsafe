"""Experiment domain model: Scenario × OperatingConfig × Mission → ExperimentResult.

Everything here is a plain, serialisable, hashable Pydantic model. No behaviour beyond validation
and identity, so that the same objects travel unchanged between the local executor, a Nebius job,
the evaluator and the policy compiler.
"""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from failsafe.mission.schema import MissionSpec

# ---------------------------------------------------------------------------------------------
# Metrics with provenance
# ---------------------------------------------------------------------------------------------


class MetricKind(str, Enum):
    MEASURED = "measured"
    SIMULATED = "simulated"
    DERIVED = "derived"
    UNAVAILABLE = "unavailable"


class Metric(BaseModel):
    model_config = ConfigDict(extra="forbid")

    value: float | None = None
    kind: MetricKind
    unit: str = ""
    note: str = ""

    @model_validator(mode="after")
    def _value_consistency(self) -> Metric:
        if self.kind == MetricKind.UNAVAILABLE and self.value is not None:
            raise ValueError("UNAVAILABLE metric must not carry a value")
        if self.kind != MetricKind.UNAVAILABLE and self.value is None:
            raise ValueError(f"{self.kind.value} metric must carry a value")
        return self

    @classmethod
    def measured(cls, value: float, unit: str = "", note: str = "") -> Metric:
        return cls(value=float(value), kind=MetricKind.MEASURED, unit=unit, note=note)

    @classmethod
    def derived(cls, value: float, unit: str = "", note: str = "") -> Metric:
        return cls(value=float(value), kind=MetricKind.DERIVED, unit=unit, note=note)

    @classmethod
    def simulated(cls, value: float, unit: str = "", note: str = "") -> Metric:
        return cls(value=float(value), kind=MetricKind.SIMULATED, unit=unit, note=note)

    @classmethod
    def unavailable(cls, note: str = "") -> Metric:
        return cls(value=None, kind=MetricKind.UNAVAILABLE, note=note)


# ---------------------------------------------------------------------------------------------
# Scenario: the failure condition the world imposes
# ---------------------------------------------------------------------------------------------


class CloudState(str, Enum):
    """`slow` is not `dead`. These are modelled separately on purpose (see DECISIONS D-008)."""

    HEALTHY = "healthy"  # ~50 ms RTT
    SLOW = "slow"  # ~500 ms RTT
    SEVERELY_SLOW = "severely_slow"  # ~1500 ms RTT
    ZOMBIE = "zombie"  # request accepted, response after 5–10 s
    TIMEOUT = "timeout"  # never responds
    OFFLINE = "offline"  # fails immediately (connection refused / no route)


DEFAULT_CLOUD_RTT_MS: dict[CloudState, tuple[float, float]] = {
    # (base_ms, jitter_ms). Zombie/timeout/offline are handled by the injector, not by RTT.
    CloudState.HEALTHY: (50.0, 15.0),
    CloudState.SLOW: (500.0, 100.0),
    CloudState.SEVERELY_SLOW: (1500.0, 300.0),
    CloudState.ZOMBIE: (7500.0, 2500.0),
    CloudState.TIMEOUT: (0.0, 0.0),
    CloudState.OFFLINE: (0.0, 0.0),
}


class ComputePressure(str, Enum):
    NORMAL = "normal"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"


class Scenario(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    cloud_state: CloudState = CloudState.HEALTHY
    cloud_rtt_ms: float | None = Field(default=None, ge=0, description="override base RTT")
    cloud_jitter_ms: float | None = Field(default=None, ge=0)
    bandwidth_mbps: float | None = Field(default=None, gt=0, description="None = unconstrained")
    compute_pressure: ComputePressure = ComputePressure.NORMAL
    seed: int = 0

    def effective_rtt(self) -> tuple[float, float]:
        base, jitter = DEFAULT_CLOUD_RTT_MS[self.cloud_state]
        if self.cloud_rtt_ms is not None:
            base = self.cloud_rtt_ms
        if self.cloud_jitter_ms is not None:
            jitter = self.cloud_jitter_ms
        return base, jitter

    @property
    def wan_available(self) -> bool:
        return self.cloud_state not in (CloudState.TIMEOUT, CloudState.OFFLINE)

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------
# OperatingConfig: the candidate degraded mode (every knob does real work)
# ---------------------------------------------------------------------------------------------

CriticalFps = Literal[30, 15, 10, 5]
BackgroundFps = Literal[30, 15, 10, 5, 2, 1, 0]
DetectorResolution = Literal[640, 480, 320]
DropBackground = Literal["none", "lowest_priority", "all"]
BacklogPolicy = Literal["queue", "drop_oldest"]


class OperatingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""

    critical_fps: CriticalFps = 15
    background_fps: BackgroundFps = 5
    detector_resolution: DetectorResolution = 640
    local_confidence_threshold: float = Field(default=0.4, ge=0.05, le=0.95)

    cloud_confirmation: bool = True
    cloud_timeout_ms: int = Field(default=3000, ge=100, le=30000)
    # what the pipeline does with a candidate when the cloud call fails/times out
    on_cloud_failure: Literal["alert_local", "drop"] = "alert_local"
    # consecutive in-zone frames required before raising a candidate (latency ↔ precision)
    alert_confirm_frames: Literal[1, 2, 3] = 1
    historical_indexing: bool = True

    drop_background_streams: DropBackground = "none"
    backlog_policy: BacklogPolicy = "drop_oldest"

    @property
    def processing_mode(self) -> Literal["normal", "local_only"]:
        return "normal" if self.cloud_confirmation else "local_only"

    def canonical_json(self) -> str:
        d = self.model_dump(mode="json")
        d.pop("name", None)
        d.pop("description", None)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))

    @property
    def hash(self) -> str:
        """Hash of the behaviour-affecting knobs only (name/description excluded)."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------
# Corpus reference
# ---------------------------------------------------------------------------------------------


class CorpusRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["synthetic", "real"] = "synthetic"
    tier: str = "quick"
    seed: int = 1
    path: str | None = None  # for real holdout clips

    @property
    def hash(self) -> str:
        s = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(s.encode()).hexdigest()[:12]


# ---------------------------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------------------------


class Experiment(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mission: MissionSpec
    scenario: Scenario
    config: OperatingConfig
    corpus: CorpusRef = Field(default_factory=CorpusRef)
    time_scale: float = Field(default=1.0, gt=0, description="1.0 = wall-clock real time")
    hypothesis: str | None = None  # set by a search strategy (e.g. Nemotron) if any
    proposed_by: str = "manual"  # manual | grid | greedy | nemotron

    @property
    def id(self) -> str:
        s = "|".join(
            [
                self.mission.hash,
                self.scenario.hash,
                self.config.hash,
                self.corpus.hash,
                f"{self.time_scale:.4f}",
            ]
        )
        return hashlib.sha256(s.encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------------------------


class InvariantCheck(BaseModel):
    model_config = ConfigDict(extra="forbid")

    metric: str
    value: float | None
    operator: Literal[">=", "<="]
    threshold: float
    passed: bool
    note: str = ""


class VerificationResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    passed: bool
    checks: list[InvariantCheck]
    mission_hash: str


# ---------------------------------------------------------------------------------------------
# Provenance + Result
# ---------------------------------------------------------------------------------------------


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    mission_hash: str
    scenario_hash: str
    config_hash: str
    corpus_hash: str
    seed: int
    time_scale: float
    git_sha: str | None = None
    python_version: str = ""
    torch_version: str | None = None
    ultralytics_version: str | None = None
    detector_model: str = ""
    confirmer_model: str = ""
    backend: str = "local"
    hostname: str = ""
    platform: str = ""
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    nebius_job_id: str | None = None
    extra: dict[str, Any] = Field(default_factory=dict)


class AlertRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    camera: str
    scene_t: float  # scene time of the frame that triggered the alert
    emitted_wall: float  # wall-clock time alert was emitted (monotonic, relative to run start)
    frame_release_wall: float  # wall-clock time the triggering frame was released
    confirmed_by: Literal["local", "cloud"]
    confidence: float
    matched_event_id: str | None = None
    latency_ms: float | None = None  # only for the first alert matched to an event


class ExperimentResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    experiment: Experiment
    metrics: dict[str, Metric]
    verification: VerificationResult | None = None
    provenance: Provenance
    alerts: list[AlertRecord] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @property
    def passed(self) -> bool | None:
        return None if self.verification is None else self.verification.passed

    def metric_value(self, name: str) -> float | None:
        m = self.metrics.get(name)
        return None if m is None else m.value

    def to_json(self) -> str:
        return self.model_dump_json(indent=2)

    @classmethod
    def from_json(cls, s: str) -> ExperimentResult:
        return cls.model_validate_json(s)
