import pytest
from pydantic import ValidationError

from failsafe.experiments.schema import (
    CloudState,
    CorpusRef,
    Experiment,
    Metric,
    MetricKind,
    OperatingConfig,
    Scenario,
)
from failsafe.mission.schema import Invariant, MissionSpec, Objective


def _mission(**kw) -> MissionSpec:
    base = dict(
        name="m",
        invariants=[Invariant(metric="critical_event_recall", min=0.95)],
        objectives=[Objective(metric="alert_precision", direction="maximize")],
    )
    base.update(kw)
    return MissionSpec(**base)


def test_invariant_requires_a_bound():
    with pytest.raises(ValidationError):
        Invariant(metric="x")


def test_invariant_min_le_max():
    with pytest.raises(ValidationError):
        Invariant(metric="x", min=2, max=1)


def test_mission_rejects_metric_in_both_roles():
    with pytest.raises(ValidationError):
        _mission(objectives=[Objective(metric="critical_event_recall", direction="maximize")])


def test_mission_rejects_unknown_fields():
    with pytest.raises(ValidationError):
        MissionSpec(name="m", invariants=[Invariant(metric="x", min=0)], bogus=1)


def test_mission_hash_is_stable_and_content_addressed():
    a, b = _mission(), _mission()
    assert a.hash == b.hash
    c = _mission(invariants=[Invariant(metric="critical_event_recall", min=0.9)])
    assert c.hash != a.hash


def test_mission_yaml_roundtrip(tmp_path):
    m = _mission()
    p = tmp_path / "m.yaml"
    m.to_yaml(p)
    assert MissionSpec.from_yaml(p).hash == m.hash


def test_repo_mission_file_loads():
    m = MissionSpec.from_yaml("missions/restricted-zone.yaml")
    assert {i.metric for i in m.invariants} == {"critical_event_recall", "alert_latency_p95_ms"}
    assert m.survivability.wan_outage == "required"


def test_metric_kind_consistency():
    with pytest.raises(ValidationError):
        Metric(value=1.0, kind=MetricKind.UNAVAILABLE)
    with pytest.raises(ValidationError):
        Metric(value=None, kind=MetricKind.MEASURED)
    assert Metric.unavailable("no gpu").value is None


def test_operating_config_rejects_out_of_catalogue_values():
    with pytest.raises(ValidationError):
        OperatingConfig(name="c", critical_fps=7)
    with pytest.raises(ValidationError):
        OperatingConfig(name="c", detector_resolution=1000)
    with pytest.raises(ValidationError):
        OperatingConfig(name="c", backlog_policy="explode")


def test_operating_config_hash_ignores_name():
    a = OperatingConfig(name="a", background_fps=2)
    b = OperatingConfig(name="b", background_fps=2)
    c = OperatingConfig(name="a", background_fps=1)
    assert a.hash == b.hash != c.hash
    assert a.processing_mode == "normal"
    assert OperatingConfig(name="x", cloud_confirmation=False).processing_mode == "local_only"


def test_scenario_rtt_and_wan():
    s = Scenario(name="s", cloud_state=CloudState.SLOW)
    assert s.effective_rtt()[0] == 500.0
    assert s.wan_available
    assert not Scenario(name="o", cloud_state=CloudState.OFFLINE).wan_available
    assert Scenario(name="z", cloud_state=CloudState.ZOMBIE).wan_available  # slow != dead
    assert Scenario(name="r", cloud_rtt_ms=123).effective_rtt()[0] == 123


def test_experiment_id_changes_with_time_scale():
    e1 = Experiment(mission=_mission(), scenario=Scenario(name="s"), config=OperatingConfig(name="c"))
    e2 = e1.model_copy(update={"time_scale": 4.0})
    assert e1.id != e2.id
    assert e1.id == Experiment(mission=_mission(), scenario=Scenario(name="s"), config=OperatingConfig(name="c")).id


def test_experiment_json_roundtrip():
    e = Experiment(
        mission=_mission(),
        scenario=Scenario(name="s", cloud_state=CloudState.ZOMBIE, bandwidth_mbps=2),
        config=OperatingConfig(name="c", cloud_confirmation=False),
        corpus=CorpusRef(tier="quick", seed=7),
    )
    e2 = Experiment.model_validate_json(e.model_dump_json())
    assert e2.id == e.id
