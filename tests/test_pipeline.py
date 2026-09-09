"""Pipeline plumbing tests with an oracle detector (reads the scene) and a simulated clock.
These test wiring and matching, NOT detector quality — that is what real-time experiments measure."""

from __future__ import annotations

import numpy as np
import pytest

from failsafe.corpus.assets import CutoutLibrary
from failsafe.corpus.scene import generate_scene
from failsafe.experiments.evaluator import verify
from failsafe.experiments.schema import CloudState, Metric, OperatingConfig, Scenario
from failsafe.faults.network import NetworkInjector
from failsafe.mission.schema import MissionSpec
from failsafe.workload.clock import SimulatedClock
from failsafe.workload.cloud import RESPONSE_BYTES, AlwaysConfirm, ConfirmationRequest, ConfirmationResponse
from failsafe.workload.metrics import ResourceSamples, compute_metrics
from failsafe.workload.pipeline import Pipeline, active_cameras, build_schedule
from failsafe.workload.source import SyntheticSceneSource
from failsafe.workload.zone import Detection


class OracleDetector:
    """Returns the exact boxes of the persons currently in frame (from the scene), so pipeline
    recall depends only on sampling and plumbing."""

    name = "oracle"

    def __init__(self, source: SyntheticSceneSource):
        self.source = source
        self.infer_ms: list[float] = []
        self._t: float | None = None
        self._cam: str | None = None

    def bind(self, cam: str, t: float) -> None:
        self._cam, self._t = cam, t

    def detect(self, frame: np.ndarray, imgsz: int, conf: float) -> list[Detection]:
        self.infer_ms.append(0.0)
        out = []
        for tr in self.source.renderer.active_tracks(self._cam, self._t):
            fx, fy = tr.foot_at(self._t)
            w = tr.height_px * 0.35
            out.append(Detection(fx - w / 2, fy - tr.height_px, fx + w / 2, fy, 0.99))
        return out


class OracleSource(SyntheticSceneSource):
    """Binds (camera, t) into the oracle detector when a frame is fetched."""

    def __init__(self, scene, cutouts):
        super().__init__(scene, cutouts)
        self.oracle = OracleDetector(self)

    def frame(self, camera: str, scene_t: float):
        self.oracle.bind(camera, scene_t)
        return np.zeros((self.scene.height, self.scene.width, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def source():
    cut = CutoutLibrary()
    return OracleSource(generate_scene("smoke", 1), cut)


MISSION = MissionSpec.from_yaml("missions/restricted-zone.yaml")


def _run(source, config, scenario=None, confirmer=None, injector=None):
    scenario = scenario or Scenario(name="t")
    clock = SimulatedClock()
    pipe = Pipeline(source, config, scenario, source.oracle, clock, confirmer=confirmer, injector=injector, time_scale=1.0)
    out = pipe.run()
    metrics, alerts = compute_metrics(
        out.alerts, source.ground_truth, real_time=False, frames_processed=out.frames_processed,
        frames_dropped=out.frames_dropped, run_wall_s=max(out.run_wall_s, 1e-6), detector_ms=out.detector_ms,
        baseline_detector_ms=None, link=out.link, resources=ResourceSamples(), cameras_active=out.cameras_active,
    )
    return out, metrics, alerts


def test_schedule_respects_fps_and_camera_drops(source):
    cfg = OperatingConfig(name="c", critical_fps=15, background_fps=5)
    sched = build_schedule(source, cfg)
    per_cam = {}
    for t, cam in sched:
        per_cam.setdefault(cam, []).append(t)
    assert len(per_cam["A"]) == int(source.duration_s * 15)
    assert len(per_cam["B"]) == int(source.duration_s * 5)
    assert sched == sorted(sched)
    assert active_cameras(source, OperatingConfig(name="x", drop_background_streams="all")) == ["A"]
    assert "D" not in active_cameras(source, OperatingConfig(name="x", drop_background_streams="lowest_priority"))
    assert active_cameras(source, OperatingConfig(name="x", background_fps=0)) == ["A"]


def test_local_only_oracle_reaches_full_recall(source):
    cfg = OperatingConfig(name="island", critical_fps=30, background_fps=30, cloud_confirmation=False, historical_indexing=False)
    out, m, alerts = _run(source, cfg)
    assert m["critical_event_recall"].value == 1.0
    assert m["alert_precision"].value == 1.0  # oracle never fires on near-misses (foot-point rule)
    assert out.frames_dropped == 0
    assert m["alert_latency_p95_ms"].kind.value == "unavailable"  # simulated clock → not real time
    assert all(a.confirmed_by == "local" for a in alerts)


def test_low_fps_misses_short_events():
    cut = CutoutLibrary()
    quick = OracleSource(generate_scene("quick", 1), cut)
    full = OperatingConfig(name="fast", critical_fps=30, background_fps=30, cloud_confirmation=False, historical_indexing=False)
    slow = OperatingConfig(name="slow", critical_fps=5, background_fps=1, cloud_confirmation=False, historical_indexing=False)
    _, m_full, _ = _run(quick, full)
    _, m_slow, _ = _run(quick, slow)
    assert m_full["critical_event_recall"].value == 1.0
    assert m_slow["critical_event_recall"].value < m_full["critical_event_recall"].value
    assert m_slow["frames_processed"].value < m_full["frames_processed"].value


def test_cloud_confirmation_path_counts_bytes_and_confirms(source):
    cfg = OperatingConfig(name="normal", critical_fps=30, background_fps=30, historical_indexing=False)
    out, m, alerts = _run(source, cfg, confirmer=AlwaysConfirm(), injector=NetworkInjector(Scenario(name="h", cloud_rtt_ms=0, cloud_jitter_ms=0)))
    assert m["critical_event_recall"].value == 1.0
    assert m["cloud_requests"].value >= len(source.ground_truth.events)
    assert m["cloud_bytes_total"].value > 1000
    assert all(a.confirmed_by == "cloud" for a in alerts)


def test_offline_cloud_fails_over_to_local_alert(source):
    cfg = OperatingConfig(name="normal", critical_fps=30, background_fps=30, historical_indexing=False)
    sc = Scenario(name="off", cloud_state=CloudState.OFFLINE)
    out, m, alerts = _run(source, cfg, scenario=sc, confirmer=AlwaysConfirm(), injector=NetworkInjector(sc))
    assert m["critical_event_recall"].value == 1.0
    assert m["cloud_unavailable"].value >= 1
    assert all(a.confirmed_by == "local" for a in alerts)


def test_offline_cloud_with_drop_policy_loses_recall(source):
    cfg = OperatingConfig(name="strict", critical_fps=30, background_fps=30, historical_indexing=False, on_cloud_failure="drop")
    sc = Scenario(name="off", cloud_state=CloudState.OFFLINE)
    _, m, alerts = _run(source, cfg, scenario=sc, confirmer=AlwaysConfirm(), injector=NetworkInjector(sc))
    assert m["critical_event_recall"].value == 0.0
    assert alerts == []


def test_cloud_rejection_suppresses_alert(source):
    class Never:
        name = "never"

        def confirm(self, req: ConfirmationRequest):
            return ConfirmationResponse(False, 0.0, "no"), RESPONSE_BYTES

    cfg = OperatingConfig(name="normal", critical_fps=30, background_fps=30, historical_indexing=False)
    _, m, alerts = _run(source, cfg, confirmer=Never(), injector=NetworkInjector(Scenario(name="h", cloud_rtt_ms=0, cloud_jitter_ms=0)))
    assert alerts == []
    assert m["critical_event_recall"].value == 0.0


def test_pipeline_requires_confirmer_when_cloud_enabled(source):
    with pytest.raises(ValueError):
        Pipeline(source, OperatingConfig(name="n"), Scenario(name="s"), source.oracle, SimulatedClock())


def test_verify_marks_latency_unavailable_as_fail(source):
    cfg = OperatingConfig(name="island", critical_fps=30, background_fps=30, cloud_confirmation=False, historical_indexing=False)
    _, m, _ = _run(source, cfg)
    v = verify(m, MISSION)
    assert not v.passed  # recall ok, latency UNAVAILABLE → fail closed
    assert next(c for c in v.checks if c.metric == "alert_latency_p95_ms").value is None
    m["alert_latency_p95_ms"] = Metric.measured(500)
    assert verify(m, MISSION).passed
