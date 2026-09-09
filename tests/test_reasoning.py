"""Reasoning layer tests: no network, no API key. The mock provider drives the LLM search loop
against the synthetic outcome model from test_search."""

from __future__ import annotations

import pytest

from failsafe.experiments.catalog import SEARCH_SPACE, grid_configs
from failsafe.experiments.schema import CloudState, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.reasoning.base import PlanningContext, ProposalSet, RejectedOutput, validate_or_reject
from failsafe.reasoning.mock import MockReasoningProvider
from failsafe.reasoning.nemotron import analysis_prompt, extract_json_object, planning_prompt
from failsafe.reasoning.planner import LLMSearch, snap_to_space
from failsafe.search.objective import REFERENCE
from failsafe.search.strategies import Evaluator
from tests.test_search import FakeCache, zombie_world

MISSION = MissionSpec.from_yaml("missions/restricted-zone.yaml")
SPACE = grid_configs()
ZOMBIE = Scenario(name="wan_zombie", cloud_state=CloudState.ZOMBIE)


@pytest.fixture
def ev():
    return Evaluator(FakeCache(zombie_world, ZOMBIE), MISSION, ZOMBIE)


def test_mock_planner_reaches_a_verified_mode(ev):
    traj = LLMSearch(MockReasoningProvider()).run(ev, SPACE)
    assert traj.termination_reason == "first feasible found"
    assert traj.best_config is not None and traj.evaluations <= 4
    assert traj.strategy == "llm:mock"
    assert all(s.source == "cached" for s in traj.steps)  # replay, no live runs
    assert any("hypothesis" in n for n in traj.notes)


def test_malformed_model_output_is_rejected_not_guessed(ev):
    prov = MockReasoningProvider(invalid_on_round=0)
    traj = LLMSearch(prov, max_rejections=1).run(ev, SPACE)
    assert traj.evaluations == 0 and traj.best_config is None
    assert "rejected" in traj.termination_reason
    assert traj.no_verified_mode is not None  # explicit negative, not a silent nothing


def test_out_of_space_candidates_are_rejected(ev):
    outside = OperatingConfig(name="x", critical_fps=30, background_fps=30, detector_resolution=320)  # not in the space
    prov = MockReasoningProvider(scripted=[[outside], [REFERENCE.model_copy(update={"cloud_timeout_ms": 1000, "name": "ok"})]])
    traj = LLMSearch(prov).run(ev, SPACE)
    assert any("outside the search space" in n for n in traj.notes)
    assert traj.steps[0].config.cloud_timeout_ms == 1000  # only the in-space one was evaluated
    assert snap_to_space(outside, SPACE) is None


def test_schema_validation_rejects_bad_payloads():
    with pytest.raises(RejectedOutput):
        validate_or_reject(ProposalSet, {"hypothesis": "h", "candidates": []})
    with pytest.raises(RejectedOutput):
        validate_or_reject(ProposalSet, {"hypothesis": "h", "candidates": [{"config": {"name": "c", "critical_fps": 7}, "rationale": "r"}]})
    ok = validate_or_reject(ProposalSet, {"hypothesis": "h", "candidates": [{"config": {"name": "c"}, "rationale": "r"}]})
    assert isinstance(ok, ProposalSet)


def test_extract_json_object_variants():
    assert extract_json_object('{"a": 1}') == {"a": 1}
    assert extract_json_object('Sure:\n```json\n{"a": [1,2]}\n```') == {"a": [1, 2]}
    assert extract_json_object('prefix text {"a": {"b": 2}} suffix') == {"a": {"b": 2}}
    with pytest.raises(RejectedOutput):
        extract_json_object("no json here")
    with pytest.raises(RejectedOutput):
        extract_json_object("[1, 2, 3]")


def test_prompts_carry_evidence_and_space():
    ctx = PlanningContext(mission=MISSION, scenario=ZOMBIE, space={k: list(v) for k, v in SEARCH_SPACE.items()}, observations=[], round_index=0)
    p = planning_prompt(ctx)
    assert "critical_event_recall >= 0.95" in p and "alert_latency_p95_ms <= 2000" in p
    assert "wan_zombie" in p and "cloud: ['off', 'on_3000', 'on_1000']" in p
    assert "(none yet)" in p
    a = analysis_prompt(ctx)
    assert "OBSERVATIONS" in a


def test_mock_compile_mission_validates():
    m = MockReasoningProvider().compile_mission("Detect people in the loading zone; alerts within 2 s; survive WAN loss.")
    assert {i.metric for i in m.invariants} == {"critical_event_recall", "alert_latency_p95_ms"}
