"""Timeline demo under a simulated clock: the world cuts the WAN, the runtime notices through
health pings, switches to a verified mode, and switches back when the WAN returns."""

from __future__ import annotations

import pytest

from failsafe.corpus.assets import CutoutLibrary
from failsafe.corpus.scene import generate_scene
from failsafe.demo.timeline import Phase, TimelineDemo
from failsafe.experiments.schema import CloudState, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.policy.compiler import compile_policy
from failsafe.search.objective import REFERENCE
from failsafe.workload.clock import SimulatedClock
from failsafe.workload.cloud import AlwaysConfirm
from tests.test_pipeline import OracleSource
from tests.test_policy import Cache, runs

MISSION = MissionSpec.from_yaml("missions/restricted-zone.yaml")
HEALTHY = Scenario(name="healthy", cloud_rtt_ms=0, cloud_jitter_ms=0)
OFFLINE = Scenario(name="wan_offline", cloud_state=CloudState.OFFLINE)
NAIVE = REFERENCE.model_copy(update={"name": "naive_drop_on_fail", "on_cloud_failure": "drop", "critical_fps": 30, "background_fps": 30})
NORMAL = REFERENCE.model_copy(update={"name": "normal", "cloud_timeout_ms": 1000, "critical_fps": 30, "background_fps": 30})
ISLAND = OperatingConfig(name="island", critical_fps=30, background_fps=30, cloud_confirmation=False, historical_indexing=False)


@pytest.fixture(scope="module")
def source():
    cut = CutoutLibrary()
    return OracleSource(generate_scene("smoke", 1), cut)


def _policy():
    # two clean runs across two seeds per experiment: the admission bar needs >=2 runs, >=2 seeds
    cache = Cache([
        *runs(NORMAL, HEALTHY, recall=0.982, p95=900),
        *runs(ISLAND, OFFLINE, recall=0.982, p95=900),
    ])
    return compile_policy(MISSION, [HEALTHY, OFFLINE], cache, mode_names={"healthy": "normal", "wan_offline": "offline-cloud"})


def _phases(source):
    d = source.duration_s
    return [Phase(0.0, HEALTHY), Phase(d * 0.35, OFFLINE), Phase(d * 0.7, HEALTHY)]


def test_naive_system_loses_events_while_offline(source):
    demo = TimelineDemo(source, MISSION, _phases(source), NAIVE, source.oracle, SimulatedClock(), AlwaysConfirm(), policy=None)
    rep = demo.run("naive")
    off = next(p for p in rep.phases if p.name == "wan_offline")
    assert off.events >= 1 and off.detected == 0  # drop-on-fail + dead cloud = every intrusion missed
    assert rep.transitions == [] and rep.config_changes == []


def test_policy_switches_to_island_and_back(source):
    pol = _policy()
    demo = TimelineDemo(source, MISSION, _phases(source), pol.modes[0].config, source.oracle, SimulatedClock(), AlwaysConfirm(),
                        policy=pol, probe_interval_s=0.5, degrade_after=2, recover_after=3)
    rep = demo.run("policy")
    names = [t["to"] for t in rep.transitions]
    assert names[:3] == ["normal", "offline-cloud", "normal"], names
    assert all(t["verified"] for t in rep.transitions)
    # island mode was applied to the pipeline while offline
    assert any(n == "offline-cloud" for _, n in rep.config_changes)
    off = next(p for p in rep.phases if p.name == "wan_offline")
    assert off.events >= 1 and off.detected == off.events  # local alerting kept every intrusion
    assert "offline-cloud" in off.active_modes
    # reaction time: the switch happened within a few probe intervals of the cut
    cut_t = _phases(source)[1].start_t
    switch_t = next(t["scene_t"] for t in rep.transitions if t["to"] == "offline-cloud")
    assert 0 <= switch_t - cut_t <= 3.0
    assert rep.overall["critical_event_recall"] == 1.0


def test_unverified_condition_reports_fallback(source):
    pol = _policy()
    zombie = Scenario(name="wan_zombie", cloud_state=CloudState.ZOMBIE, cloud_rtt_ms=3000, cloud_jitter_ms=0)
    phases = [Phase(0.0, HEALTHY), Phase(source.duration_s * 0.4, zombie)]
    demo = TimelineDemo(source, MISSION, phases, pol.modes[0].config, source.oracle, SimulatedClock(), AlwaysConfirm(),
                        policy=pol, probe_interval_s=0.5, degrade_after=2, recover_after=3, ping_timeout_s=0.3)
    rep = demo.run("zombie-unverified")
    # zombie was never verified in this tiny policy → fallback, labelled UNVERIFIED
    assert any((not t["verified"]) and "NO VERIFIED MODE" in t["reason"] for t in rep.transitions)
