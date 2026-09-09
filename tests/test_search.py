"""Search engine tests against a synthetic outcome model (no real experiments)."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from failsafe.experiments.catalog import grid_configs
from failsafe.experiments.evaluator import verify
from failsafe.experiments.schema import (
    CloudState,
    CorpusRef,
    Experiment,
    ExperimentResult,
    Metric,
    OperatingConfig,
    Provenance,
    Scenario,
)
from failsafe.mission.schema import MissionSpec
from failsafe.search.objective import (
    REFERENCE,
    best_of,
    capability_retained,
    capability_terms,
    frontier,
    margin_tier,
    no_verified_mode,
    rank_key,
)
from failsafe.search.strategies import (
    Evaluator,
    GreedySearch,
    GridSearch,
    MissingResult,
    RandomSearch,
    ResultCache,
    greedy_move,
    simulate_random,
)

MISSION = MissionSpec.from_yaml("missions/restricted-zone.yaml")
SPACE = grid_configs()


def fake_result(cfg: OperatingConfig, scenario: Scenario, *, recall: float, p95: float, dropped: float = 0, released: float = 3000, cpu: float = 100, cloud_bytes: float = 0) -> ExperimentResult:
    exp = Experiment(mission=MISSION, scenario=scenario, config=cfg, corpus=CorpusRef(tier="quick", seed=1))
    metrics = {
        "critical_event_recall": Metric.measured(recall),
        "alert_latency_p95_ms": Metric.measured(p95),
        "alert_precision": Metric.measured(1.0),
        "frames_dropped": Metric.measured(dropped),
        "frames_released": Metric.measured(released),
        "cpu_percent_mean": Metric.measured(cpu),
        "cloud_bytes_total": Metric.measured(cloud_bytes),
        "gt_events_total": Metric.measured(57),
    }
    prov = Provenance(experiment_id=exp.id, mission_hash=MISSION.hash, scenario_hash=scenario.hash, config_hash=cfg.hash,
                      corpus_hash="x", seed=1, time_scale=1.0, started_at=datetime.now(UTC))
    return ExperimentResult(experiment=exp, metrics=metrics, verification=verify(metrics, MISSION), provenance=prov)


def zombie_world(cfg: OperatingConfig, scenario: Scenario) -> ExperimentResult:
    """Synthetic outcome model shaped like Phase 1: cloud with 3 s timeout under a zombie cloud
    blows latency; 480 px loses recall; 21+ fps demand at 640 with indexing drops frames."""
    demand = cfg.critical_fps + 3 * cfg.background_fps
    cost = {640: 47, 480: 30}[cfg.detector_resolution] + (5 if cfg.historical_indexing else 0)
    load = demand * cost / 1000
    dropped = max(0.0, (load - 0.95) * 2500)
    recall = 0.965 if cfg.detector_resolution == 640 else 0.74
    recall -= min(0.2, dropped / 5000)
    p95 = 900.0
    if cfg.cloud_confirmation:
        p95 = 5000.0 if cfg.cloud_timeout_ms >= 3000 else 1580.0
    cpu = 100 + demand * 4
    return fake_result(cfg, scenario, recall=recall, p95=p95, dropped=dropped, cpu=cpu, cloud_bytes=2e6 if cfg.cloud_confirmation else 0)


class FakeCache(ResultCache):
    def __init__(self, world, scenario):
        self._by_id = {}
        for cfg in SPACE:
            r = world(cfg, scenario)
            self._by_id[r.experiment.id] = [r]
        self.dir = None


ZOMBIE = Scenario(name="wan_zombie", cloud_state=CloudState.ZOMBIE)


@pytest.fixture
def ev():
    return Evaluator(FakeCache(zombie_world, ZOMBIE), MISSION, ZOMBIE)


def test_capability_terms_and_weights():
    assert capability_retained(REFERENCE, MISSION) == pytest.approx(1.0)
    island = OperatingConfig(name="i", critical_fps=15, background_fps=1, cloud_confirmation=False, historical_indexing=False)
    t = capability_terms(island)
    assert t["cloud_confirmation"] == 0 and t["historical_indexing"] == 0 and t["background_streams"] == 0.5
    # weights: critical 100, alerting 100, cloud 50, indexing 20, background 10 → (100+100+5)/280
    assert capability_retained(island, MISSION) == pytest.approx(205 / 280)
    survival = OperatingConfig(name="s", critical_fps=5, background_fps=0, detector_resolution=320, cloud_confirmation=False,
                               historical_indexing=False, drop_background_streams="all")
    assert capability_retained(survival, MISSION) < capability_retained(island, MISSION)


def test_rank_is_lexicographic():
    a = fake_result(REFERENCE, ZOMBIE, recall=0.97, p95=1500, cpu=200)  # feasible, full capability, costly
    b = fake_result(OperatingConfig(name="b", cloud_confirmation=False, historical_indexing=False), ZOMBIE, recall=0.99, p95=500, cpu=100)
    c = fake_result(REFERENCE, ZOMBIE, recall=0.90, p95=500)  # infeasible
    ka, kb, kc = (rank_key(x, MISSION) for x in (a, b, c))
    assert ka.as_tuple() > kb.as_tuple() > kc.as_tuple()  # capability beats quality/cost; feasibility beats all
    assert best_of([a, b, c], MISSION) is a
    assert best_of([c], MISSION) is None


def test_tolerance_bands_tie_noise():
    a = fake_result(REFERENCE, ZOMBIE, recall=0.97, p95=1500, cpu=201.0)
    b = fake_result(REFERENCE, ZOMBIE, recall=0.97, p95=1500, cpu=204.0)  # same 10 %-wide band (quantised, not ±tol)
    assert rank_key(a, MISSION).as_tuple() == rank_key(b, MISSION).as_tuple()


def test_grid_finds_max_capability_feasible(ev):
    traj = GridSearch().run(ev, SPACE)
    assert traj.evaluations == len(SPACE) == 72
    assert traj.best_config is not None
    # best: 640 px, cloud on with 1 s timeout, indexing on if it fits capacity
    assert traj.best_config.detector_resolution == 640
    assert traj.best_config.cloud_confirmation and traj.best_config.cloud_timeout_ms == 1000
    assert traj.best_capability == max(s.capability for s in traj.steps if s.passed)
    assert traj.pareto and traj.no_verified_mode is None


def test_greedy_reaches_feasible_in_few_steps(ev):
    traj = GreedySearch().run(ev, SPACE)
    assert traj.termination_reason == "first feasible found"
    assert traj.evaluations <= 4
    assert traj.steps[0].config.hash == REFERENCE.hash
    assert "cloud_timeout" in traj.steps[1].reason  # latency with cloud on → shorter timeout first
    assert traj.best_config is not None and traj.best_config.cloud_confirmation


def test_random_stops_at_first_feasible(ev):
    traj = RandomSearch(seed=3).run(ev, SPACE)
    assert traj.steps[-1].passed and all(not s.passed for s in traj.steps[:-1])
    stats = simulate_random(ev, SPACE, n_seeds=200)
    assert stats["feasible"] > 0 and stats["evaluations_mean"] > 1
    assert abs(stats["evaluations_mean"] - stats["analytic_mean"]) < 1.0


def test_no_verified_mode_is_explicit():
    def hopeless(cfg, sc):
        return fake_result(cfg, sc, recall=0.80, p95=900)

    sc = Scenario(name="compute_severe")
    ev2 = Evaluator(FakeCache(hopeless, sc), MISSION, sc)
    traj = GridSearch().run(ev2, SPACE)
    assert traj.best_config is None and traj.feasible_found == 0
    assert traj.no_verified_mode["ceilings"]["critical_event_recall"] == pytest.approx(0.80)
    nvm = no_verified_mode([hopeless(c, sc) for c in SPACE[:3]], MISSION, "compute_severe")
    assert "VERIFIED MODE: NONE" in nvm.render()
    g = GreedySearch().run(ev2, SPACE)
    assert g.best_config is None and g.termination_reason in ("no degradation move left", "cycle: candidate already evaluated", "max steps reached")


def test_frontier_dominance():
    sc = ZOMBIE
    a = fake_result(OperatingConfig(name="a", cloud_confirmation=True, cloud_timeout_ms=1000), sc, recall=0.97, p95=1500, cpu=200)
    b = fake_result(OperatingConfig(name="b", cloud_confirmation=False), sc, recall=0.97, p95=900, cpu=150)
    c = fake_result(OperatingConfig(name="c", cloud_confirmation=False, historical_indexing=False), sc, recall=0.97, p95=900, cpu=180)  # dominated by b
    pts, pareto = frontier([a, b, c], MISSION)
    assert {p.config_name for p in pts} == {"a", "b", "c"}
    assert {p.config_name for p in pareto} == {"a", "b"}


def test_greedy_move_overload_sheds_cheapest_first():
    cfg = REFERENCE
    r = fake_result(cfg, ZOMBIE, recall=0.90, p95=900, dropped=600, released=3000)
    new, reason = greedy_move(cfg, r, MISSION)
    assert new.historical_indexing is False and "overloaded" in reason
    r2 = fake_result(new, ZOMBIE, recall=0.90, p95=900, dropped=600, released=3000)
    new2, reason2 = greedy_move(new, r2, MISSION)
    assert new2.background_fps == 1


def test_evaluator_refuses_live_by_default():
    ev = Evaluator(ResultCache(__import__("pathlib").Path("/nonexistent")), MISSION, ZOMBIE)
    with pytest.raises(MissingResult):
        ev.evaluate(REFERENCE)


def test_margin_tiers_rank_robust_above_marginal():
    robust = fake_result(OperatingConfig(name="r", cloud_confirmation=False), ZOMBIE, recall=0.982, p95=900)   # 2 events above threshold
    marginal = fake_result(REFERENCE, ZOMBIE, recall=0.965, p95=900)  # 1 event above threshold → marginal
    marginal_lat = fake_result(REFERENCE, ZOMBIE, recall=0.982, p95=1900)  # within 250 ms of the 2 s limit
    assert margin_tier(robust, MISSION) == 2
    assert margin_tier(marginal, MISSION) == 1
    assert margin_tier(marginal_lat, MISSION) == 1
    assert margin_tier(fake_result(REFERENCE, ZOMBIE, recall=0.90, p95=900), MISSION) == 0
    # a robust pass with lower capability outranks a marginal pass with full capability
    assert rank_key(robust, MISSION).as_tuple() > rank_key(marginal, MISSION).as_tuple()
    assert best_of([robust, marginal], MISSION) is robust
