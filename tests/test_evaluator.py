from failsafe.experiments.evaluator import objective_score, verify
from failsafe.experiments.schema import Metric
from failsafe.mission.schema import Invariant, MissionSpec, Objective

MISSION = MissionSpec(
    name="m",
    invariants=[
        Invariant(metric="critical_event_recall", min=0.95),
        Invariant(metric="alert_latency_p95_ms", max=2000),
    ],
    objectives=[
        Objective(metric="alert_precision", direction="maximize", weight=2),
        Objective(metric="cloud_bytes_total", direction="minimize"),
    ],
)


def test_pass():
    v = verify(
        {"critical_event_recall": Metric.measured(0.962), "alert_latency_p95_ms": Metric.measured(1310)},
        MISSION,
    )
    assert v.passed and all(c.passed for c in v.checks)
    assert v.mission_hash == MISSION.hash


def test_fail_on_threshold():
    v = verify(
        {"critical_event_recall": Metric.measured(0.891), "alert_latency_p95_ms": Metric.measured(1310)},
        MISSION,
    )
    assert not v.passed
    failed = [c for c in v.checks if not c.passed]
    assert [c.metric for c in failed] == ["critical_event_recall"]


def test_boundary_is_inclusive():
    v = verify(
        {"critical_event_recall": Metric.measured(0.95), "alert_latency_p95_ms": Metric.measured(2000)},
        MISSION,
    )
    assert v.passed


def test_missing_metric_fails_closed():
    v = verify({"critical_event_recall": Metric.measured(0.99)}, MISSION)
    assert not v.passed
    assert any(c.metric == "alert_latency_p95_ms" and c.value is None for c in v.checks)


def test_unavailable_metric_fails_closed():
    v = verify(
        {
            "critical_event_recall": Metric.measured(0.99),
            "alert_latency_p95_ms": Metric.unavailable("time_scale != 1"),
        },
        MISSION,
    )
    assert not v.passed
    bad = next(c for c in v.checks if c.metric == "alert_latency_p95_ms")
    assert "unavailable" in bad.note


def test_simulated_metric_is_flagged_but_evaluated():
    v = verify(
        {"critical_event_recall": Metric.simulated(0.99), "alert_latency_p95_ms": Metric.measured(100)},
        MISSION,
    )
    assert v.passed
    assert "simulated" in next(c for c in v.checks if c.metric == "critical_event_recall").note


def test_objective_score_ranks_within_bounds():
    bounds = {"alert_precision": (0.8, 1.0), "cloud_bytes_total": (0.0, 1000.0)}
    good = {"alert_precision": Metric.measured(1.0), "cloud_bytes_total": Metric.measured(0.0)}
    bad = {"alert_precision": Metric.measured(0.8), "cloud_bytes_total": Metric.measured(1000.0)}
    assert objective_score(good, MISSION.objectives, bounds) == 1.0
    assert objective_score(bad, MISSION.objectives, bounds) == 0.0
    assert objective_score({"alert_precision": Metric.measured(1.0)}, MISSION.objectives, bounds) is None
