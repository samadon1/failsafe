"""Deterministic verification. No LLM logic lives here, ever.

`verify` takes measured metrics and a MissionSpec and returns a VerificationResult. An invariant
whose metric is missing or UNAVAILABLE fails (fail closed): we never pass a configuration on the
strength of a number we did not measure.
"""

from __future__ import annotations

from failsafe.experiments.schema import (
    InvariantCheck,
    Metric,
    MetricKind,
    VerificationResult,
)
from failsafe.mission.schema import MissionSpec, Objective


def verify(metrics: dict[str, Metric], mission: MissionSpec) -> VerificationResult:
    checks: list[InvariantCheck] = []
    for inv in mission.invariants:
        m = metrics.get(inv.metric)
        if m is None or m.kind == MetricKind.UNAVAILABLE or m.value is None:
            note = "metric missing" if m is None else f"metric {m.kind.value}: {m.note}".strip()
            if inv.min is not None:
                checks.append(
                    InvariantCheck(
                        metric=inv.metric, value=None, operator=">=", threshold=inv.min, passed=False, note=note
                    )
                )
            if inv.max is not None:
                checks.append(
                    InvariantCheck(
                        metric=inv.metric, value=None, operator="<=", threshold=inv.max, passed=False, note=note
                    )
                )
            continue

        note = "" if m.kind == MetricKind.MEASURED else f"metric is {m.kind.value}"
        if inv.min is not None:
            checks.append(
                InvariantCheck(
                    metric=inv.metric,
                    value=m.value,
                    operator=">=",
                    threshold=inv.min,
                    passed=m.value >= inv.min,
                    note=note,
                )
            )
        if inv.max is not None:
            checks.append(
                InvariantCheck(
                    metric=inv.metric,
                    value=m.value,
                    operator="<=",
                    threshold=inv.max,
                    passed=m.value <= inv.max,
                    note=note,
                )
            )

    return VerificationResult(
        passed=all(c.passed for c in checks) and len(checks) > 0,
        checks=checks,
        mission_hash=mission.hash,
    )


def objective_score(
    metrics: dict[str, Metric],
    objectives: list[Objective],
    bounds: dict[str, tuple[float, float]],
) -> float | None:
    """Weighted, normalised soft-objective utility in [0, 1].

    `bounds[metric] = (lo, hi)` gives the normalisation range observed across the experiment set
    (so a score is only comparable within one set). Returns None if any objective metric is
    unavailable. Used only for *ranking passing configurations*, never for accept/reject.
    """
    if not objectives:
        return None
    total_w = sum(o.weight for o in objectives)
    score = 0.0
    for o in objectives:
        m = metrics.get(o.metric)
        if m is None or m.value is None:
            return None
        lo, hi = bounds.get(o.metric, (m.value, m.value))
        if hi <= lo:
            norm = 1.0
        else:
            norm = (m.value - lo) / (hi - lo)
        if o.direction == "minimize":
            norm = 1.0 - norm
        score += o.weight * max(0.0, min(1.0, norm))
    return score / total_w
