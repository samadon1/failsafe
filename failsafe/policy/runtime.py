"""Deterministic policy runtime.

    observed signals ─► classify() ─► (cloud_state, compute_pressure, bandwidth)
                                            │
                                            ▼
                            policy.select() → verified Mode | None
                                            │
                        hysteresis: a change must persist for N probes
                                            │
                                            ▼
              activate verified mode  |  NO VERIFIED MODE → fallback (labelled unverified) + escalate

No LLM anywhere on this path. Every transition is logged with the reason and whether the active
mode is verified for the current condition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict

from failsafe.experiments.schema import CloudState, ComputePressure
from failsafe.policy.schema import Mode, ResiliencePolicy

# Classification thresholds (D-031, D-034, D-040). The RTT classes are boundaries chosen between
# the scenario RTTs (50 / 500 / 1500 / 7500 ms), not measured values; the zombie class starts at
# 3 s, i.e. beyond any cloud timeout in the search space. The slowdown floor 1.5 is measured
# (clean in-pipeline runs sit at 1.0–1.3x, D-034); 2.2 and 3.5 are asserted and pinned by tests.
RTT_HEALTHY_MS = 300.0
RTT_SLOW_MS = 1000.0
RTT_SEVERE_MS = 3000.0
# Clean in-pipeline runs measure 1.0–1.3× the isolated baseline (GIL + confirmer side-load,
# repeatability data); classification starts above that noise floor (D-034).
SLOWDOWN_MODERATE = 1.5
SLOWDOWN_SEVERE = 2.2
SLOWDOWN_CRITICAL = 3.5
TIMEOUT_RATE_ZOMBIE = 0.5


class Observed(BaseModel):
    """What the runtime can actually measure at the edge (all optional: unknown ≠ healthy)."""

    model_config = ConfigDict(extra="forbid")

    cloud_reachable: bool | None = None  # TCP-level: connection accepted?
    cloud_rtt_p95_ms: float | None = None  # over the recent probe window
    cloud_timeout_rate: float = 0.0  # fraction of recent calls that hit the caller timeout
    detector_slowdown: float = 1.0  # detector ms/frame ÷ calibrated baseline
    bandwidth_mbps: float | None = None  # estimated uplink; None = not constrained / unknown


@dataclass(frozen=True)
class Condition:
    cloud: CloudState
    compute: ComputePressure
    bandwidth_mbps: float | None

    def label(self) -> str:
        b = "" if self.bandwidth_mbps is None else f", {self.bandwidth_mbps:g} Mbps"
        return f"{self.cloud.value} × {self.compute.value}{b}"


def classify(o: Observed) -> Condition:
    if o.cloud_reachable is False:
        cloud = CloudState.OFFLINE
    elif o.cloud_timeout_rate >= TIMEOUT_RATE_ZOMBIE and (o.cloud_rtt_p95_ms is None or o.cloud_rtt_p95_ms >= RTT_SEVERE_MS):
        cloud = CloudState.ZOMBIE if o.cloud_reachable else CloudState.TIMEOUT
    elif o.cloud_rtt_p95_ms is None:
        # reachable (or unknown) but no completed round trips: treat as timeout-class, never as healthy
        cloud = CloudState.TIMEOUT if o.cloud_reachable else CloudState.OFFLINE
    elif o.cloud_rtt_p95_ms < RTT_HEALTHY_MS:
        cloud = CloudState.HEALTHY
    elif o.cloud_rtt_p95_ms < RTT_SLOW_MS:
        cloud = CloudState.SLOW
    elif o.cloud_rtt_p95_ms < RTT_SEVERE_MS:
        cloud = CloudState.SEVERELY_SLOW
    else:
        cloud = CloudState.ZOMBIE
    s = o.detector_slowdown
    compute = (
        ComputePressure.NORMAL if s < SLOWDOWN_MODERATE
        else ComputePressure.MODERATE if s < SLOWDOWN_SEVERE
        else ComputePressure.SEVERE if s < SLOWDOWN_CRITICAL
        else ComputePressure.CRITICAL
    )
    return Condition(cloud, compute, o.bandwidth_mbps)


@dataclass
class Decision:
    mode: Mode | None
    verified: bool
    changed: bool
    condition: Condition
    reason: str
    pending: str | None = None  # a candidate mode waiting for hysteresis


@dataclass
class Transition:
    at: datetime
    from_mode: str | None
    to_mode: str | None
    condition: str
    verified: bool
    reason: str


@dataclass
class PolicyRuntime:
    policy: ResiliencePolicy
    degrade_after: int = 3  # consecutive probes proposing a different mode before switching down
    recover_after: int = 5  # consecutive probes before switching to a higher-capability mode
    active: Mode | None = None
    active_verified: bool = False
    _pending: str | None = field(default=None)
    _pending_count: int = 0
    _started: bool = False
    transitions: list[Transition] = field(default_factory=list)

    def _proposal(self, cond: Condition) -> tuple[Mode | None, bool, str]:
        m = self.policy.select(cond.cloud, cond.compute, cond.bandwidth_mbps)
        if m is not None:
            return m, True, f"verified mode for {cond.label()}"
        # say *why* there is no verified mode: refuted is not the same as never tested (D-037)
        u = self.policy.unverified_for(cond.cloud, cond.compute, cond.bandwidth_mbps)
        why = f" ({u.reason}{', exhaustive' if u.exhaustive else ''}, {u.candidates_tested} candidates tested)" if u else " (condition not in the policy)"
        fb = self.policy.fallback
        if fb is not None:
            return fb, False, f"NO VERIFIED MODE AVAILABLE for {cond.label()}{why}: fallback `{fb.name}` (verified only under {fb.verification.scenario}); escalate"
        return None, False, f"NO VERIFIED MODE AVAILABLE for {cond.label()}{why} and no fallback defined: halt and escalate"

    def step(self, observed: Observed, now: datetime | None = None) -> Decision:
        now = now or datetime.now(UTC)
        cond = classify(observed)
        proposed, verified, reason = self._proposal(cond)
        proposed_name = proposed.name if proposed else None
        active_name = self.active.name if self.active else None

        if self._started and proposed_name == active_name and verified == self.active_verified:
            self._pending, self._pending_count = None, 0
            return Decision(self.active, self.active_verified, False, cond, "unchanged")

        if self._pending != proposed_name:
            self._pending, self._pending_count = proposed_name, 1
        else:
            self._pending_count += 1

        # first activation is immediate; degrading (to lower capability or to fallback) needs
        # `degrade_after` consistent probes; recovering (to higher capability) needs `recover_after`
        if self.active is None:
            needed = 1
        elif proposed is None or not verified or proposed.capability_retained <= self.active.capability_retained:
            needed = self.degrade_after
        else:
            needed = self.recover_after
        if self._pending_count < needed:
            return Decision(self.active, self.active_verified, False, cond, f"pending {proposed_name} ({self._pending_count}/{needed})", pending=proposed_name)

        self.transitions.append(Transition(now, active_name, proposed_name, cond.label(), verified, reason))
        self.active, self.active_verified = proposed, verified
        self._started = True
        self._pending, self._pending_count = None, 0
        return Decision(proposed, verified, True, cond, reason)
