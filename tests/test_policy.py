"""Policy compiler + runtime tests on a synthetic result cache (no real experiments)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from failsafe.experiments.schema import CloudState, ComputePressure, CorpusRef, Metric, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.policy.compiler import compile_policy, conservativeness_key, sacrifices
from failsafe.policy.runtime import Observed, PolicyRuntime, classify
from failsafe.policy.schema import ResiliencePolicy
from failsafe.search.objective import REFERENCE
from failsafe.search.strategies import ResultCache
from tests.test_search import fake_result

MISSION = MissionSpec.from_yaml("missions/restricted-zone.yaml")
HEALTHY = Scenario(name="healthy")
ZOMBIE = Scenario(name="wan_zombie", cloud_state=CloudState.ZOMBIE)
OFFLINE = Scenario(name="wan_offline", cloud_state=CloudState.OFFLINE)
SEVERE = Scenario(name="compute_severe", compute_pressure=ComputePressure.SEVERE)
SLOW = Scenario(name="wan_slow", cloud_state=CloudState.SLOW)

NORMAL = REFERENCE.model_copy(update={"name": "normal", "cloud_timeout_ms": 1000})
ISLAND = OperatingConfig(name="island", critical_fps=15, background_fps=1, cloud_confirmation=False, historical_indexing=False)
SURVIVAL = OperatingConfig(name="survival", critical_fps=5, background_fps=0, detector_resolution=320, cloud_confirmation=False,
                           historical_indexing=False, drop_background_streams="all")


def seeded(cfg, sc, seed, **kw):
    """One run of cfg × sc on a given corpus seed (its own experiment id)."""
    r = fake_result(cfg, sc, **kw)
    return r.model_copy(update={"experiment": r.experiment.model_copy(update={"corpus": CorpusRef(tier="quick", seed=seed)})})


def runs(cfg, sc, k=2, **kw):
    """k stored runs of the same config × scenario, one per distinct corpus seed 1..k (so a pair
    of runs clears the >=2-seed admission bar, D-048)."""
    return [seeded(cfg, sc, s, **kw) for s in range(1, k + 1)]


def flagged(r):
    """Mark a run as contended (D-028); the compiler must not decide on it."""
    r.metrics["qc_suspect_contention"] = Metric.measured(1)
    return r


class Cache(ResultCache):
    def __init__(self, results):
        self._by_id = {}
        for r in results:
            self._by_id.setdefault(r.experiment.id, []).append(r)
        self.dir = None


@pytest.fixture
def cache():
    return Cache([
        *runs(NORMAL, HEALTHY, recall=0.982, p95=900, cpu=200),
        *runs(REFERENCE, HEALTHY, recall=0.947, p95=1000),           # naive default fails
        *runs(ISLAND, HEALTHY, recall=0.965, p95=950, cpu=180),       # marginal pass (1 event above the floor)
        *runs(NORMAL, ZOMBIE, recall=0.982, p95=1580, cpu=200),
        *runs(REFERENCE, ZOMBIE, recall=1.0, p95=5050),
        *runs(ISLAND, ZOMBIE, recall=0.982, p95=950, cpu=180),
        *runs(ISLAND, OFFLINE, recall=0.982, p95=920, cpu=180),
        *runs(ISLAND, SEVERE, recall=0.737, p95=1826),
        *runs(SURVIVAL, SEVERE, recall=0.298, p95=1827),
    ])


ALL = [HEALTHY, ZOMBIE, OFFLINE, SEVERE]


def test_compile_picks_best_verified_per_scenario_and_lists_unverified(cache):
    pol = compile_policy(MISSION, ALL, cache)
    by = {m.verification.scenario: m for m in pol.modes}
    assert by["healthy"].config.hash == NORMAL.hash and by["healthy"].verification.tier == "robust"
    assert by["wan_zombie"].config.hash == NORMAL.hash  # full capability, robust → beats island
    assert by["wan_offline"].config.hash == ISLAND.hash
    assert [u.scenario for u in pol.unverified] == ["compute_severe"]
    assert pol.unverified[0].reason == "refuted" and pol.unverified[0].best_candidate is None
    assert pol.unverified[0].ceilings["critical_event_recall"] == pytest.approx(0.737)
    assert pol.fallback is not None and pol.fallback.config.hash == ISLAND.hash  # no cloud wait → most conservative
    assert pol.modes[0].capability_retained >= pol.modes[-1].capability_retained
    assert "cloud confirmation off" in by["wan_offline"].sacrifices
    assert pol.admission.min_clean_runs == 2 and by["healthy"].verification.clean_runs == 2


# --- D-037: admission is decided on every clean run, not the first one ----------------------


def test_single_passing_run_is_evidence_not_verification():
    c = Cache(runs(NORMAL, HEALTHY, k=1, recall=0.982, p95=900))
    pol = compile_policy(MISSION, [HEALTHY], c)
    assert not pol.modes and pol.fallback is None
    u = pol.unverified[0]
    assert u.reason == "insufficient_evidence" and u.candidates_tested == 1
    assert u.best_candidate_config == "normal" and u.best_candidate_clean_runs == 1
    # a second clean run at the SAME seed is still not enough (D-048: needs >=2 seeds)
    c.add(seeded(NORMAL, HEALTHY, 1, recall=0.982, p95=900))
    assert not compile_policy(MISSION, [HEALTHY], c).modes
    # a run on a second seed admits it
    c.add(seeded(NORMAL, HEALTHY, 2, recall=0.982, p95=900))
    pol2 = compile_policy(MISSION, [HEALTHY], c)
    assert pol2.modes and pol2.modes[0].verification.clean_runs == 3 and pol2.modes[0].verification.seeds == [1, 2]


def test_disagreeing_clean_runs_are_not_admitted():
    # the shipped zombie-cloud defect: three clean runs on three seeds, one over the latency limit
    c = Cache([
        seeded(NORMAL, ZOMBIE, 1, recall=0.982, p95=1583),
        seeded(NORMAL, ZOMBIE, 2, recall=1.0, p95=2030),   # fails the 2000 ms invariant
        seeded(NORMAL, ZOMBIE, 3, recall=1.0, p95=1681),
    ])
    pol = compile_policy(MISSION, [ZOMBIE], c)
    assert not pol.modes
    u = pol.unverified[0]
    assert u.reason == "refuted" and u.candidates_tested == 1 and u.best_candidate is None


def test_tier_and_metrics_come_from_the_worst_clean_run():
    c = Cache([
        seeded(NORMAL, HEALTHY, 1, recall=0.982, p95=900),
        seeded(NORMAL, HEALTHY, 2, recall=0.965, p95=1800),  # passes, but inside the 250 ms latency floor
    ])
    pol = compile_policy(MISSION, [HEALTHY], c)
    v = pol.modes[0].verification
    assert v.tier == "marginal"
    assert v.recall == pytest.approx(0.965) and v.p95_latency_ms == pytest.approx(1800)
    assert v.runs == 2 and v.clean_runs == 2 and v.pass_rate == 1.0
    strict = compile_policy(MISSION, [HEALTHY], c, require_robust=True)
    assert not strict.modes and strict.unverified[0].reason == "marginal"
    assert strict.unverified[0].best_candidate_config == "normal"


def test_contended_runs_are_excluded_from_the_decision():
    c = Cache([
        seeded(NORMAL, HEALTHY, 1, recall=0.982, p95=900),
        seeded(NORMAL, HEALTHY, 2, recall=0.965, p95=928),
        flagged(seeded(NORMAL, HEALTHY, 3, recall=0.930, p95=1401)),  # failed, but under contention
    ])
    pol = compile_policy(MISSION, [HEALTHY], c)
    v = pol.modes[0].verification
    assert v.runs == 3 and v.clean_runs == 2
    assert v.pass_rate == pytest.approx(2 / 3) and v.clean_pass_rate == 1.0
    assert v.recall == pytest.approx(0.965)  # worst *clean* run, not the contended one
    # only contended runs → no clean evidence at all
    c2 = Cache([flagged(seeded(NORMAL, HEALTHY, s, recall=0.982, p95=900)) for s in (1, 2, 3)])
    assert compile_policy(MISSION, [HEALTHY], c2).unverified[0].reason == "refuted"


def test_evidence_pools_single_runs_across_corpus_seeds():
    # the shipped wan_offline data: the same config, one run on each of three corpus seeds
    c = Cache([seeded(NORMAL, OFFLINE, s, recall=1.0, p95=430) for s in (1, 2, 3)])
    pol = compile_policy(MISSION, [OFFLINE], c)
    assert pol.modes, "three passes on three seeds are three clean runs, not three lonely ones"
    v = pol.modes[0].verification
    assert v.clean_runs == 3 and v.seeds == [1, 2, 3] and len(v.experiment_ids) == 3
    assert pol.corpus == "quick/seeds 1,2,3"
    # one failing seed spoils the pool, exactly like one failing repeat
    c.add(seeded(NORMAL, OFFLINE, 4, recall=0.90, p95=430))
    assert not compile_policy(MISSION, [OFFLINE], c).modes


def test_cloud_off_evidence_pools_across_cloud_states_it_cannot_observe():
    # D-047: island has no injector, so healthy / zombie / offline are one experiment for it; two
    # clean runs on two seeds under two cloud states verify every cloud state at that pressure
    c = Cache([seeded(ISLAND, HEALTHY, 1, recall=0.982, p95=900), seeded(ISLAND, OFFLINE, 2, recall=0.982, p95=920)])
    pol = compile_policy(MISSION, [HEALTHY, OFFLINE, ZOMBIE], c)
    by = {m.verification.scenario: m for m in pol.modes}
    assert set(by) == {"healthy", "wan_offline", "wan_zombie"}
    assert all(m.verification.clean_runs == 2 and m.verification.seeds == [1, 2] for m in pol.modes)
    # one clean failure anywhere in the pool refutes the config for every cloud state
    c.add(seeded(ISLAND, ZOMBIE, 3, recall=0.930, p95=900))
    pol2 = compile_policy(MISSION, [HEALTHY, OFFLINE, ZOMBIE], c)
    assert not pol2.modes and {u.reason for u in pol2.unverified} == {"refuted"}
    # a cloud-ON config is still judged per scenario
    c3 = Cache(runs(NORMAL, HEALTHY, recall=0.982, p95=900))
    pol3 = compile_policy(MISSION, [HEALTHY, ZOMBIE], c3)
    assert [m.verification.scenario for m in pol3.modes] == ["healthy"] and pol3.unverified[0].scenario == "wan_zombie"


def test_two_runs_on_one_seed_are_not_enough(cache):
    # D-048: a config verified on a single seed's luck is insufficient_evidence, not a mode
    c = Cache([seeded(NORMAL, HEALTHY, 1, recall=0.982, p95=900), seeded(NORMAL, HEALTHY, 1, recall=0.982, p95=900)])
    pol = compile_policy(MISSION, [HEALTHY], c)
    assert not pol.modes and pol.unverified[0].reason == "insufficient_evidence"
    assert pol.admission.min_clean_seeds == 2
    # the same two runs on two seeds are enough
    ok = compile_policy(MISSION, [HEALTHY], Cache(runs(NORMAL, HEALTHY, recall=0.982, p95=900)))
    assert ok.modes and ok.modes[0].verification.seeds == [1, 2]


def test_untested_scenario_is_listed_not_dropped():
    # only a cloud-ON config has runs, so its healthy evidence says nothing about a slow cloud (D-047:
    # a cloud-OFF config's runs would have covered wan_slow, so the fixture cache is not used here)
    c = Cache(runs(NORMAL, HEALTHY, recall=0.982, p95=900))
    pol = compile_policy(MISSION, [HEALTHY, SLOW], c)
    u = {x.scenario: x for x in pol.unverified}["wan_slow"]
    assert u.reason == "untested" and u.candidates_tested == 0 and u.ceilings == {}
    assert pol.unverified_for(CloudState.SLOW, ComputePressure.NORMAL) is u


def test_refuted_is_marked_exhaustive_only_when_the_space_was_tried(cache):
    pol = compile_policy(MISSION, ALL, cache, space_size=2)
    assert {u.scenario: u.exhaustive for u in pol.unverified} == {"compute_severe": True}
    pol2 = compile_policy(MISSION, ALL, cache, space_size=72)
    assert pol2.unverified[0].exhaustive is False


# --- D-037: the fallback is the most conservative admitted config, not the first mode ---------


def test_fallback_never_waits_on_the_cloud_when_a_local_mode_is_admitted(cache):
    pol = compile_policy(MISSION, ALL, cache)
    assert pol.fallback.config.cloud_confirmation is False
    assert pol.fallback.config.hash == ISLAND.hash


def test_fallback_is_deterministic_when_every_admitted_config_is_identical():
    # the shipped degeneracy: every verified mode had the same cloud-on config
    c = Cache([*runs(NORMAL, HEALTHY, recall=0.982, p95=900), *runs(NORMAL, ZOMBIE, recall=0.982, p95=1580),
               *runs(NORMAL, OFFLINE, recall=1.0, p95=430)])
    pol = compile_policy(MISSION, [HEALTHY, ZOMBIE, OFFLINE], c)
    assert pol.fallback.config.hash == NORMAL.hash
    assert pol.fallback.verification.scenario == "healthy"  # cites the healthy evidence when it exists


def test_conservativeness_prefers_no_cloud_then_least_demand():
    k_cloud, k_island, k_survival = (conservativeness_key(c, 1.0) for c in (NORMAL, ISLAND, SURVIVAL))
    assert k_survival < k_island < k_cloud
    # capability must NOT come before cloud dependency: a lower-capability cloud-on config still ranks after island
    assert conservativeness_key(ISLAND, 0.9) < conservativeness_key(NORMAL, 0.5)


# --- unchanged behaviour ----------------------------------------------------------------------


def test_policy_yaml_roundtrip(cache, tmp_path):
    pol = compile_policy(MISSION, ALL, cache)
    p = pol.to_yaml(tmp_path / "policy.yaml")
    back = ResiliencePolicy.from_yaml(p)
    assert back.model_dump() == pol.model_dump()
    assert "NO VERIFIED MODE" in back.ladder() and "refuted" in back.ladder()


def test_legacy_policy_file_declares_its_own_bar(tmp_path):
    legacy = ResiliencePolicy(mission_name="m", mission_hash="x", corpus="c", modes=[])
    assert legacy.admission.min_clean_runs == 1 and "legacy" in legacy.admission.rule


def test_select_only_verified_conditions(cache):
    pol = compile_policy(MISSION, ALL, cache)
    assert pol.select(CloudState.HEALTHY, ComputePressure.NORMAL).config.hash == NORMAL.hash
    assert pol.select(CloudState.OFFLINE, ComputePressure.NORMAL).config.hash == ISLAND.hash
    assert pol.select(CloudState.SLOW, ComputePressure.NORMAL) is None  # never measured → not covered
    assert pol.select(CloudState.HEALTHY, ComputePressure.SEVERE) is None


def test_classify_thresholds():
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=120)).cloud == CloudState.HEALTHY
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=600)).cloud == CloudState.SLOW
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=1500)).cloud == CloudState.SEVERELY_SLOW
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=6000, cloud_timeout_rate=0.9)).cloud == CloudState.ZOMBIE
    assert classify(Observed(cloud_reachable=False)).cloud == CloudState.OFFLINE
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=None)).cloud == CloudState.TIMEOUT  # unknown ≠ healthy
    assert classify(Observed(detector_slowdown=1.0)).compute == ComputePressure.NORMAL
    assert classify(Observed(detector_slowdown=2.6)).compute == ComputePressure.SEVERE
    assert classify(Observed(detector_slowdown=4.0)).compute == ComputePressure.CRITICAL


def test_classify_boundaries_pin_every_constant():
    # each threshold is a documented constant (D-031/D-034/D-037); a silent change must fail here
    assert classify(Observed(detector_slowdown=1.49)).compute == ComputePressure.NORMAL
    assert classify(Observed(detector_slowdown=1.5)).compute == ComputePressure.MODERATE
    assert classify(Observed(detector_slowdown=2.19)).compute == ComputePressure.MODERATE
    assert classify(Observed(detector_slowdown=2.2)).compute == ComputePressure.SEVERE
    assert classify(Observed(detector_slowdown=3.5)).compute == ComputePressure.CRITICAL
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=299)).cloud == CloudState.HEALTHY
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=300)).cloud == CloudState.SLOW
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=999)).cloud == CloudState.SLOW
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=1000)).cloud == CloudState.SEVERELY_SLOW
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=3000)).cloud == CloudState.ZOMBIE
    # the timeout-rate signal: at the 0.5 boundary with a severe RTT it is zombie, below it is not
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=3500, cloud_timeout_rate=0.5)).cloud == CloudState.ZOMBIE
    assert classify(Observed(cloud_reachable=True, cloud_rtt_p95_ms=1500, cloud_timeout_rate=0.49)).cloud == CloudState.SEVERELY_SLOW


def test_runtime_transitions_with_hysteresis_and_fail_closed(cache):
    pol = compile_policy(MISSION, ALL, cache)
    rt = PolicyRuntime(pol, degrade_after=3, recover_after=5)
    t0 = datetime.now(UTC)
    healthy = Observed(cloud_reachable=True, cloud_rtt_p95_ms=150)
    d = rt.step(healthy, t0)
    assert d.changed and d.verified and d.mode.name == "healthy"  # first activation immediate

    # WAN cut: needs 3 consecutive probes
    off = Observed(cloud_reachable=False)
    d1 = rt.step(off, t0 + timedelta(seconds=1))
    d2 = rt.step(off, t0 + timedelta(seconds=2))
    assert not d1.changed and not d2.changed and d2.pending == "wan_offline"
    d3 = rt.step(off, t0 + timedelta(seconds=3))
    assert d3.changed and d3.mode.name == "wan_offline" and d3.verified
    assert d3.mode.config.cloud_confirmation is False

    # flapping back to healthy for 2 probes does not recover (needs 5)
    for i in range(4):
        assert not rt.step(healthy, t0 + timedelta(seconds=10 + i)).changed
    d = rt.step(healthy, t0 + timedelta(seconds=15))
    assert d.changed and d.mode.name == "healthy"

    # severe compute pressure: no verified mode → fallback, labelled unverified, after 3 probes
    sev = Observed(cloud_reachable=True, cloud_rtt_p95_ms=150, detector_slowdown=2.8)
    for i in range(2):
        rt.step(sev, t0 + timedelta(seconds=20 + i))
    d = rt.step(sev, t0 + timedelta(seconds=23))
    assert d.changed and not d.verified and d.mode.name == pol.fallback.name
    assert "NO VERIFIED MODE" in d.reason and "refuted" in d.reason  # the runtime says why
    assert [t.to_mode for t in rt.transitions] == ["healthy", "wan_offline", "healthy", pol.fallback.name]
    assert rt.transitions[-1].verified is False


def test_runtime_without_fallback_halts():
    pol = ResiliencePolicy(mission_name="m", mission_hash="x", corpus="c", modes=[], unverified=[], fallback=None)
    rt = PolicyRuntime(pol)
    d = rt.step(Observed(cloud_reachable=True, cloud_rtt_p95_ms=100))
    assert d.mode is None and not d.verified and "halt" in d.reason


def test_sacrifices_text():
    assert sacrifices(REFERENCE) == []
    s = sacrifices(SURVIVAL)
    assert "cloud confirmation off" in s and any("320" in x for x in s) and any("5 fps" in x for x in s)
