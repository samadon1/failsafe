"""Resilience policy: the compiled, deployable artefact.

Every mode links to the experiment that verified it. Conditions are expressed in the same terms
the runtime can observe (cloud state class, compute-pressure class, bandwidth). Conditions with no
verified mode are listed explicitly — with the reason (untested / insufficient evidence / marginal /
refuted), the measured ceilings and the candidate to repeat — and map to a *fallback* that is
labelled unverified for that condition. The runtime never pretends otherwise.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from failsafe.experiments.schema import CloudState, ComputePressure, OperatingConfig

Tier = Literal["robust", "marginal"]
Reason = Literal["untested", "insufficient_evidence", "marginal", "refuted"]


class Conditions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cloud_state: list[CloudState]
    compute_pressure: list[ComputePressure]
    bandwidth_mbps_max: float | None = None  # mode verified with bandwidth ≤ this (None = unconstrained)

    def matches(self, cloud: CloudState, compute: ComputePressure, bandwidth_mbps: float | None) -> bool:
        if cloud not in self.cloud_state or compute not in self.compute_pressure:
            return False
        if self.bandwidth_mbps_max is not None and (bandwidth_mbps is None or bandwidth_mbps > self.bandwidth_mbps_max):
            # verified only under constrained bandwidth; unconstrained is a *different* condition
            return bandwidth_mbps is None
        return True


class Verification(BaseModel):
    """What the experiments actually showed for this mode (D-037).

    Runs are pooled per configuration × scenario across corpus seeds and repeats. `recall`,
    `p95_latency_ms` and `precision` are the WORST values observed across the clean (un-contended)
    runs, not a single run's; `experiment_id` is the experiment that worst case came from and
    `experiment_ids` / `seeds` list everything pooled. `runs` counts every stored run,
    `clean_runs` the un-contended ones the admission decision was made on; `pass_rate` is over all
    runs, `clean_pass_rate` over the clean ones (1.0 by construction for an admitted mode)."""

    model_config = ConfigDict(extra="forbid")

    experiment_id: str
    experiment_ids: list[str] = Field(default_factory=list)
    seeds: list[int] = Field(default_factory=list)
    scenario: str
    recall: float | None
    p95_latency_ms: float | None
    precision: float | None
    tier: Tier
    runs: int = 1
    clean_runs: int = 1
    pass_rate: float = 1.0
    clean_pass_rate: float = 1.0
    corpus_hash: str
    git_sha: str | None = None
    scoring_rule: str = ""


class Mode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    conditions: Conditions
    config: OperatingConfig
    capability_retained: float
    verification: Verification
    sacrifices: list[str] = Field(default_factory=list)  # human-readable: what this mode gives up vs. the reference


class UnverifiedCondition(BaseModel):
    """A condition the policy refuses to serve with a verified mode, and why.

    reason:
      untested               no stored result for this scenario at all
      insufficient_evidence  a candidate passed, but on fewer clean runs than admission requires
      marginal               a candidate passed every clean run but inside the noise floor, and the
                             policy was compiled with require_robust
      refuted                every tested candidate failed at least one clean run
    `exhaustive` is set when the whole search space was tried, so `refuted` is a strong negative
    result rather than an under-searched one. `best_candidate` is the experiment to repeat first."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    conditions: Conditions
    reason: Reason = "refuted"
    candidates_tested: int
    exhaustive: bool = False
    best_candidate: str | None = None  # experiment id
    best_candidate_config: str | None = None
    best_candidate_clean_runs: int = 0
    ceilings: dict[str, float | None]
    required: dict[str, str]
    action: Literal["fallback", "halt"] = "fallback"


class Admission(BaseModel):
    """The rule a mode had to satisfy to be admitted into this policy. Recorded so the artefact
    states its own bar. The default describes the pre-D-037 behaviour, so an old policy file
    loaded through this schema is labelled honestly rather than upgraded silently."""

    model_config = ConfigDict(extra="forbid")

    min_clean_runs: int = 1
    min_clean_seeds: int = 1
    require_robust: bool = False
    rule: str = "legacy: the first un-contended run decides (pre D-037)"


class ResiliencePolicy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    policy_version: int = 2
    mission_name: str
    mission_hash: str
    corpus: str
    compiled_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    compiled_from: list[str] = Field(default_factory=list)  # experiment ids considered
    admission: Admission = Field(default_factory=Admission)
    modes: list[Mode]  # ordered: highest capability first (the "ladder")
    unverified: list[UnverifiedCondition] = Field(default_factory=list)
    fallback: Mode | None = None  # most conservative admitted mode; used fail-closed for unverified conditions
    notes: list[str] = Field(default_factory=list)

    def to_yaml(self, path: str | Path) -> Path:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False, allow_unicode=True))
        return p

    @classmethod
    def from_yaml(cls, path: str | Path) -> "ResiliencePolicy":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()))

    def select(self, cloud: CloudState, compute: ComputePressure, bandwidth_mbps: float | None = None) -> Mode | None:
        """Highest-capability verified mode whose conditions match, else None (fail closed)."""
        for m in self.modes:
            if m.conditions.matches(cloud, compute, bandwidth_mbps):
                return m
        return None

    def unverified_for(self, cloud: CloudState, compute: ComputePressure, bandwidth_mbps: float | None = None) -> UnverifiedCondition | None:
        """The recorded reason this condition has no verified mode, if the compiler saw it."""
        for u in self.unverified:
            if u.conditions.matches(cloud, compute, bandwidth_mbps):
                return u
        return None

    def ladder(self) -> str:
        lines = [f"{'admission':18} min {self.admission.min_clean_runs} clean runs across {self.admission.min_clean_seeds} seeds, all must pass"
                 + (", robust only" if self.admission.require_robust else "")]
        for m in self.modes:
            c, v = m.config, m.verification
            lines.append(
                f"{m.name:18} cap {m.capability_retained:.2f}  [{', '.join(s.value for s in m.conditions.cloud_state)} × "
                f"{', '.join(s.value for s in m.conditions.compute_pressure)}]  crit {c.critical_fps}fps bg {c.background_fps}fps "
                f"{c.detector_resolution}px cloud {'on/' + str(c.cloud_timeout_ms) + 'ms' if c.cloud_confirmation else 'off'} "
                f"idx {'on' if c.historical_indexing else 'off'}  ({v.tier}, worst of {v.clean_runs} clean runs: "
                f"recall {v.recall:.3f}, p95 {v.p95_latency_ms:.0f} ms)"
            )
        for u in self.unverified:
            why = u.reason + (" (exhaustive)" if u.exhaustive else "")
            rep = f"; repeat {u.best_candidate_config} ({u.best_candidate_clean_runs} clean run{'s' if u.best_candidate_clean_runs != 1 else ''})" if u.best_candidate else ""
            lines.append(f"{'NO VERIFIED MODE':18} [{u.scenario}] {why}; tested {u.candidates_tested}; ceilings {u.ceilings}{rep} → {u.action}")
        if self.fallback:
            lines.append(f"{'fallback':18} {self.fallback.name} (verified only under {self.fallback.verification.scenario})")
        return "\n".join(lines)
