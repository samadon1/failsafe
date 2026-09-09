"""Lexicographic mission optimisation (DECISIONS D-025).

Ordering of candidate configurations for one scenario:

  1. FEASIBLE      every hard invariant passes (deterministic evaluator; missing metric = fail)
  2. CAPABILITY    retain as much application capability as possible (config-side, mission-weighted)
  3. QUALITY       soft objectives with direction=maximize, in the order the mission lists them
  4. COST          soft objectives with direction=minimize, in the order the mission lists them

No scalar weighted score. Measured values are compared with tolerances so that run-to-run noise
does not decide the order. Capability is a property of the *configuration* (what the application
still does), not a measurement; the weights come from the mission's `priorities`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from failsafe.experiments.schema import ExperimentResult, OperatingConfig
from failsafe.mission.schema import MissionSpec

# Reference = the highest-capability configuration of the search space.
REFERENCE = OperatingConfig(
    name="reference", critical_fps=15, background_fps=2, detector_resolution=640,
    cloud_confirmation=True, cloud_timeout_ms=3000, historical_indexing=True,
)

N_BACKGROUND_CAMERAS = 3

# Tolerances for comparing measured soft objectives (values inside one band tie).
TOLERANCE = {
    "alert_precision": 0.01,
    "cloud_bytes_total": 250_000,  # bytes (≈ 9 confirmations)
    "cpu_percent_mean": 10.0,  # percentage points of one core
}


def _bg_cameras(cfg: OperatingConfig) -> int:
    if cfg.drop_background_streams == "all" or cfg.background_fps == 0:
        return 0
    if cfg.drop_background_streams == "lowest_priority":
        return N_BACKGROUND_CAMERAS - 1
    return N_BACKGROUND_CAMERAS


def capability_terms(cfg: OperatingConfig, ref: OperatingConfig = REFERENCE) -> dict[str, float]:
    """Retention in [0, 1] per mission capability, relative to the reference configuration."""
    return {
        # sensing fidelity on the critical camera: sampling rate × input resolution
        "critical_detection": min(1.0, cfg.critical_fps / ref.critical_fps) * min(1.0, cfg.detector_resolution / ref.detector_resolution),
        # alerting is never switched off in this space
        "emergency_alerting": 1.0,
        "cloud_confirmation": 1.0 if cfg.cloud_confirmation else 0.0,
        "historical_indexing": 1.0 if cfg.historical_indexing else 0.0,
        "background_streams": (_bg_cameras(cfg) / N_BACKGROUND_CAMERAS)
        * (min(1.0, cfg.background_fps / ref.background_fps) if ref.background_fps else 0.0),
    }


def capability_retained(cfg: OperatingConfig, mission: MissionSpec, ref: OperatingConfig = REFERENCE) -> float:
    terms = capability_terms(cfg, ref)
    weights = {k: float(mission.priorities.get(k, 0)) for k in terms}
    total = sum(weights.values())
    if total <= 0:
        return sum(terms.values()) / len(terms)
    return sum(weights[k] * terms[k] for k in terms) / total


def _band(value: float | None, tol: float, direction: str) -> float:
    """Quantise a measured value so that differences below `tol` tie. Missing → worst."""
    if value is None or math.isnan(value):
        return -math.inf
    q = math.floor(value / tol) if direction == "maximize" else -math.ceil(value / tol)
    return float(q)


# Noise floor per invariant metric (D-029): a pass whose margin to the threshold is below the
# floor is *marginal* — run-to-run noise can flip it. Recall's floor is one event (1 / n_events,
# supplied per result); latency's is the repeatability spread of p95 (≈ ±200 ms measured).
NOISE_FLOOR_MS = 250.0


def margin_tier(result: ExperimentResult, mission: MissionSpec) -> int:
    """2 = robust pass (every invariant clears its threshold by ≥ the noise floor),
    1 = marginal pass (passes, but within the noise floor of some threshold), 0 = fail."""
    if not result.passed:
        return 0
    n_events = result.metric_value("gt_events_total") or 0
    tier = 2
    for inv in mission.invariants:
        v = result.metric_value(inv.metric)
        if v is None:
            return 0
        if inv.metric == "critical_event_recall":
            floor = (1.0 / n_events) if n_events else 0.0
        elif "latency" in inv.metric:
            floor = NOISE_FLOOR_MS
        else:
            floor = 0.0
        margin = (v - inv.min) if inv.min is not None else (inv.max - v)
        if margin < floor:
            tier = 1
    return tier


@dataclass(frozen=True)
class RankKey:
    feasible: bool
    tier: int  # 2 robust · 1 marginal · 0 fail
    capability: float
    quality: tuple[float, ...]
    cost: tuple[float, ...]

    def as_tuple(self) -> tuple:
        return (self.tier, round(self.capability, 2), self.quality, self.cost)

    def __lt__(self, other: "RankKey") -> bool:  # higher is better; sort with reverse=True
        return self.as_tuple() < other.as_tuple()


def rank_key(result: ExperimentResult, mission: MissionSpec, ref: OperatingConfig = REFERENCE) -> RankKey:
    cfg = result.experiment.config
    quality, cost = [], []
    for o in mission.objectives:
        tol = TOLERANCE.get(o.metric, 1.0)
        b = _band(result.metric_value(o.metric), tol, o.direction)
        (quality if o.direction == "maximize" else cost).append(b)
    return RankKey(
        feasible=bool(result.passed),
        tier=margin_tier(result, mission),
        capability=capability_retained(cfg, mission, ref),
        quality=tuple(quality),
        cost=tuple(cost),
    )


def best_of(results: list[ExperimentResult], mission: MissionSpec) -> ExperimentResult | None:
    feasible = [r for r in results if r.passed]
    if not feasible:
        return None
    return max(feasible, key=lambda r: rank_key(r, mission).as_tuple())


# ---------------------------------------------------------------------------------------------
# Feasible set, Pareto frontier, explicit "no verified mode"
# ---------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrontierPoint:
    config_name: str
    experiment_id: str
    capability: float
    cpu_percent: float | None
    cloud_bytes: float | None
    recall: float | None
    p95_ms: float | None


def _dominates(a: FrontierPoint, b: FrontierPoint) -> bool:
    """a dominates b: no worse on every axis (capability ↑, cpu ↓, bytes ↓) and better on one."""
    ax = (a.capability, -(a.cpu_percent or 0.0), -(a.cloud_bytes or 0.0))
    bx = (b.capability, -(b.cpu_percent or 0.0), -(b.cloud_bytes or 0.0))
    return all(x >= y for x, y in zip(ax, bx)) and any(x > y for x, y in zip(ax, bx))


def frontier(results: list[ExperimentResult], mission: MissionSpec, names: dict[str, str] | None = None) -> tuple[list[FrontierPoint], list[FrontierPoint]]:
    """(feasible points, Pareto-optimal subset) for one scenario's results. `names` maps config
    hash → display name (stored results may carry legacy names)."""
    names = names or {}
    pts = [
        FrontierPoint(
            config_name=names.get(r.experiment.config.hash, r.experiment.config.name),
            experiment_id=r.experiment.id,
            capability=round(capability_retained(r.experiment.config, mission), 3),
            cpu_percent=r.metric_value("cpu_percent_mean"),
            cloud_bytes=r.metric_value("cloud_bytes_total"),
            recall=r.metric_value("critical_event_recall"),
            p95_ms=r.metric_value("alert_latency_p95_ms"),
        )
        for r in results
        if r.passed
    ]
    pareto = [p for p in pts if not any(_dominates(q, p) for q in pts if q is not p)]
    return pts, pareto


@dataclass(frozen=True)
class NoVerifiedMode:
    """Explicit negative result: no tested configuration satisfies the mission in this scenario.
    The least-bad failing configuration is NEVER promoted to 'valid'."""

    scenario: str
    candidates_tested: int
    ceilings: dict[str, float | None]  # best achieved value per invariant metric
    required: dict[str, str]  # threshold per invariant metric

    def render(self) -> str:
        lines = [f"SCENARIO: {self.scenario}", "VERIFIED MODE: NONE", f"candidates tested: {self.candidates_tested}"]
        for k, v in self.ceilings.items():
            vs = "n/a" if v is None else f"{v:.3f}"
            lines.append(f"  {k}: best achieved {vs}, required {self.required[k]}")
        lines.append("MISSION CANNOT BE MAINTAINED — fail closed / escalate")
        return "\n".join(lines)


def no_verified_mode(results: list[ExperimentResult], mission: MissionSpec, scenario: str) -> NoVerifiedMode:
    ceilings: dict[str, float | None] = {}
    required: dict[str, str] = {}
    for inv in mission.invariants:
        vals = [r.metric_value(inv.metric) for r in results if r.metric_value(inv.metric) is not None]
        if inv.min is not None:
            ceilings[inv.metric] = max(vals) if vals else None
            required[inv.metric] = f">= {inv.min}"
        else:
            ceilings[inv.metric] = min(vals) if vals else None
            required[inv.metric] = f"<= {inv.max}"
    return NoVerifiedMode(scenario=scenario, candidates_tested=len(results), ceilings=ceilings, required=required)
