"""The idle gate and host calibration (D-041), without loading a detector or replaying anything."""

from __future__ import annotations

import json

from failsafe.experiments import runner as R


class _Det:  # stands in for the YOLO detector; never called because calibrate_detector is patched
    name = "fake-detector"


def _runner(tmp_path, host_baseline, **kw):
    r = R.LocalRunner.__new__(R.LocalRunner)  # skip __init__: no weights, no cloud server
    r.artifacts_dir = tmp_path
    r.detector = _Det()
    r._cloud = None
    r.wait_for_idle = kw.get("wait_for_idle", True)
    r.idle_wait_s = 0.0
    r.idle_max_waits = kw.get("idle_max_waits", 5)
    r.gate_frames = 6
    r.busy_policy = kw.get("busy_policy", "abort")
    r._host_baseline = host_baseline
    return r


def _patch_readings(monkeypatch, fn):
    """The gate takes a sustained reading on a calibrated host and a quick one otherwise."""
    monkeypatch.setattr(R, "calibrate_detector", fn)
    monkeypatch.setattr(R, "sustained_reading", fn)


def test_pressure_scenarios_run_last_and_order_is_otherwise_stable():
    from failsafe.experiments.schema import ComputePressure, Scenario
    from failsafe.experiments.sweep import pressure_last

    sev = Scenario(name="compute_severe", compute_pressure=ComputePressure.SEVERE)
    a, b = Scenario(name="healthy"), Scenario(name="wan_offline")
    assert [s.name for s in pressure_last([sev, a, b])] == ["healthy", "wan_offline", "compute_severe"]
    assert [s.name for s in pressure_last([b, a])] == ["wan_offline", "healthy"]  # stable


def test_quarantined_results_never_reach_the_cache_at_any_depth(tmp_path):
    (tmp_path / "abc").mkdir(); (tmp_path / "abc" / "run1.json").write_text("{}")
    (tmp_path / "stale").mkdir(); (tmp_path / "stale" / "old.json").write_text("{}")
    (tmp_path / "stale" / "contaminated-2026-09-06" / "abc").mkdir(parents=True)
    (tmp_path / "stale" / "contaminated-2026-09-06" / "abc" / "run2.json").write_text("{}")
    assert [p.name for p in R.result_files(tmp_path)] == ["run1.json"]  # D-046: nested stale/ excluded too
    assert [p.name for p in R.result_files(tmp_path, "abc")] == ["run1.json"]


def test_two_readings_are_stable_only_when_close():
    assert R.is_stable(40.0, 44.0)
    assert not R.is_stable(40.0, 60.0)


def test_gate_waits_until_the_host_is_idle(monkeypatch, tmp_path):
    readings = iter([100.0, 90.0, 42.0])  # busy, busy, idle
    _patch_readings(monkeypatch, lambda *a, **k: next(readings))
    r = _runner(tmp_path, {640: 40.0})
    assert r.measure_baseline_when_idle(None, 640, 0.4) == 42.0


def test_gate_aborts_after_max_waits_by_default(monkeypatch, tmp_path):
    calls = []
    _patch_readings(monkeypatch, lambda *a, **k: calls.append(1) or 100.0)
    r = _runner(tmp_path, {640: 40.0}, idle_max_waits=3)
    try:
        r.measure_baseline_when_idle(None, 640, 0.4)
    except R.HostBusy as e:
        assert "2.50x" in str(e) and len(calls) == 4  # first reading + 3 waits, then refuse
    else:
        raise AssertionError("a host that stays busy past the wait budget must abort the replay")


def test_gate_can_run_flagged_instead_when_asked(monkeypatch, tmp_path):
    _patch_readings(monkeypatch, lambda *a, **k: 100.0)
    r = _runner(tmp_path, {640: 40.0}, idle_max_waits=2, busy_policy="run")
    assert r.measure_baseline_when_idle(None, 640, 0.4) == 100.0  # the QC check flags it later


def test_uncalibrated_host_never_waits(monkeypatch, tmp_path):
    calls = []
    _patch_readings(monkeypatch, lambda *a, **k: calls.append(1) or 100.0)
    r = _runner(tmp_path, None)
    assert r.measure_baseline_when_idle(None, 640, 0.4) == 100.0 and len(calls) == 1


def test_gate_can_be_switched_off(monkeypatch, tmp_path):
    _patch_readings(monkeypatch, lambda *a, **k: 100.0)
    r = _runner(tmp_path, {640: 40.0}, wait_for_idle=False)
    assert r.measure_baseline_when_idle(None, 640, 0.4) == 100.0


def test_sustained_reading_uses_the_last_third():
    class Src:
        critical_camera = "A"
        duration_s = 100.0
        def frame(self, cam, t): return None
    class Det:  # fast for the first 20 frames, then throttled
        n = 0
        def detect(self, *a):
            self.n += 1
            import time as _t; _t.sleep(0.0005 if self.n <= 20 else 0.004)
    v = R.sustained_reading(Det(), Src(), 640, 0.4, n=30)
    assert v > 2.0  # the throttled tail, not the fast start (fast ≈0.5 ms, slow ≈4 ms)


def test_calibration_roundtrip(tmp_path):
    r = _runner(tmp_path, None)
    p = r.save_host_calibration({640: 41.0, 480: 27.0})
    assert p.parent == tmp_path / "calibration"
    assert R.load_host_baseline(tmp_path) == {640: 41.0, 480: 27.0}
    assert r._host_baseline == {640: 41.0, 480: 27.0}
    d = json.loads(p.read_text())
    assert d["hostname"] and d["detector"] == "fake-detector" and "measured_at" in d
    assert R.load_host_baseline(tmp_path, hostname="some-other-host") is None


def test_calibration_record_only_ratchets_down_unless_reset(tmp_path):
    r = _runner(tmp_path, None)
    r.save_host_calibration({640: 41.0})
    r.save_host_calibration({640: 46.0})            # slower reading: the record keeps 41
    assert R.load_host_baseline(tmp_path) == {640: 41.0}
    r.save_host_calibration({640: 39.0})            # faster reading: the record improves
    assert R.load_host_baseline(tmp_path) == {640: 39.0}
    r.save_host_calibration({640: 46.0}, reset=True)  # explicit reset replaces it
    assert R.load_host_baseline(tmp_path) == {640: 46.0}


def test_calibrate_host_refuses_a_warm_machine_that_is_stable_but_slow(monkeypatch, tmp_path):
    r = _runner(tmp_path, None)
    r.remote_cameras = True
    r.save_host_calibration({640: 40.0})
    monkeypatch.setattr(R, "calibrate_detector", lambda *a, **k: 60.0)  # stable pair, 1.5x the record
    monkeypatch.setattr(R, "RemoteSyntheticSource", _Src)
    try:
        r.calibrate_host((640,), n=5)
    except RuntimeError as e:
        assert "slow" in str(e) and "cool" in str(e)
    else:
        raise AssertionError("a uniformly slow (warm) machine must be refused")
    assert r.calibrate_host((640,), n=5, reset=True) == {640: 60.0}  # unless explicitly reset


class _Src:  # stands in for RemoteSyntheticSource (a class, so the runner's isinstance check works)
    def __init__(self, *a, **k):
        self.closed = False

    def close(self):
        self.closed = True


def test_calibrate_host_refuses_an_unstable_machine(monkeypatch, tmp_path):
    readings = iter([40.0, 70.0])  # the two readings disagree: not idle
    monkeypatch.setattr(R, "calibrate_detector", lambda *a, **k: next(readings))
    monkeypatch.setattr(R, "RemoteSyntheticSource", _Src)
    r = _runner(tmp_path, None)
    r.remote_cameras = True
    try:
        r.calibrate_host((640,), n=5)
    except RuntimeError as e:
        assert "not idle" in str(e)
    else:
        raise AssertionError("an unstable pair of readings must be refused")
