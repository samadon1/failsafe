"""Run one Experiment locally: build the source, the pipeline with fault injectors, replay in real
time, measure, verify, attach provenance, persist. This is the unit of work a Nebius job executes."""

from __future__ import annotations

import json
import logging
import platform
import socket
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from failsafe.corpus.assets import CutoutLibrary
from failsafe.experiments.evaluator import verify
from failsafe.experiments.schema import (
    ComputePressure,
    Experiment,
    ExperimentResult,
    Metric,
    Provenance,
)
from failsafe.faults.compute import ComputePressureInjector
from failsafe.faults.network import NetworkInjector
from failsafe.workload.clock import RealTimeClock
from failsafe.workload.cloud import CloudServerProcess, ConfirmationService
from failsafe.workload.detector import Detector, make_detector
from failsafe.workload.metrics import ResourceMonitor, compute_metrics
from failsafe.workload.pipeline import Pipeline, make_indexer
from failsafe.workload.remote_source import RemoteSyntheticSource
from failsafe.workload.source import FileVideoSource, FrameSource, SyntheticSceneSource

log = logging.getLogger(__name__)

ARTIFACTS = Path("artifacts")


def result_files(experiments_dir: Path, experiment_id: str | None = None) -> list[Path]:
    """All stored result files (one per run; repeats live side by side under <id>/)."""
    if experiment_id is not None:
        return sorted((experiments_dir / experiment_id).glob("*.json")) if (experiments_dir / experiment_id).is_dir() else []
    # anything under stale/ at any depth is quarantined and must never reach the cache (D-046)
    return sorted(p for p in experiments_dir.rglob("*.json") if "stale" not in p.relative_to(experiments_dir).parts)


def git_sha() -> str | None:
    """Short HEAD SHA, suffixed `-dirty` when tracked files have uncommitted changes (D-038), so a
    recorded SHA pins the code that actually ran rather than the last commit."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short=12", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"], text=True, stderr=subprocess.DEVNULL).strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return None


def _versions() -> tuple[str, str | None, str | None]:
    py = sys.version.split()[0]
    try:
        import torch

        tv = torch.__version__
    except Exception:
        tv = None
    try:
        import ultralytics

        uv = ultralytics.__version__
    except Exception:
        uv = None
    return py, tv, uv


def calibrate_detector(detector: Detector, source: FrameSource, resolution: int, conf: float, n: int = 30) -> float:
    """Unpressured detector ms/frame at this resolution (baseline for compute_pressure_observed).
    Median of n frames, measured with nothing else running (called once per resolution per
    runner session, before any cloud server or pressure workers exist — D-028)."""
    ms: list[float] = []
    cam = source.critical_camera
    for i in range(n):
        t = min(source.duration_s - 0.1, i * 0.7)
        t0 = time.perf_counter()
        detector.detect(source.frame(cam, t), resolution, conf)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ms[5:]) if len(ms) > 5 else statistics.median(ms)


CONTENTION_RATIO = 1.25  # detector ms/frame ÷ baseline above which a no-pressure run is suspect
GATE_FRAMES = 300  # ~14 s at 45 ms/frame: long enough for thermal throttling to show before a run (D-043)


def sustained_reading(detector: Detector, source: FrameSource, resolution: int, conf: float, n: int = GATE_FRAMES) -> float:
    """Detector ms/frame under sustained load: the median of the LAST third of n consecutive frames,
    so a machine that starts fast and throttles within seconds reads slow, not fast (D-043)."""
    ms: list[float] = []
    cam = source.critical_camera
    for i in range(n):
        t = min(source.duration_s - 0.1, (i % 200) * 0.7)
        t0 = time.perf_counter()
        detector.detect(source.frame(cam, t), resolution, conf)
        ms.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(ms[-max(10, n // 3):])


CALIBRATION_DIRNAME = "calibration"
STABLE_TOLERANCE = 0.15  # two consecutive isolated measurements must agree within this (D-041)


def host_calibration_path(artifacts_dir: Path = ARTIFACTS, hostname: str | None = None) -> Path:
    return artifacts_dir / CALIBRATION_DIRNAME / f"{hostname or socket.gethostname()}.json"


def load_host_baseline(artifacts_dir: Path = ARTIFACTS, hostname: str | None = None) -> dict[int, float] | None:
    """Isolated detector ms/frame per resolution for a host, written by `failsafe calibrate` on an
    idle machine (D-041). None when the host was never calibrated."""
    p = host_calibration_path(artifacts_dir, hostname)
    if not p.exists():
        return None
    return {int(k): float(v) for k, v in json.loads(p.read_text())["baseline_ms"].items()}


class HostBusy(RuntimeError):
    """The idle gate exhausted its wait budget and the runner was told not to replay on a loaded
    machine (D-044). A campaign stops here rather than producing results the QC check will discard."""


def is_stable(a: float, b: float, tol: float = STABLE_TOLERANCE) -> bool:
    """Two readings of the same thing agree closely enough to count as an idle-machine measurement."""
    return abs(a - b) / max(min(a, b), 1e-9) <= tol


class LocalRunner:
    """Holds the expensive, reusable pieces (detector, cutouts, cloud server) across experiments."""

    def __init__(
        self,
        detector_kind: str = "yolo",
        detector_weights: str = "yolov8n.pt",
        detector_device: str = "cpu",
        confirmer_weights: str = "yolov8m.pt",
        artifacts_dir: Path = ARTIFACTS,
        confirmer: ConfirmationService | None = None,
        remote_cameras: bool = True,
        wait_for_idle: bool = True,
        idle_wait_s: float = 30.0,
        idle_max_waits: int = 20,
        gate_frames: int = GATE_FRAMES,
        busy_policy: str = "abort",  # when the gate's wait budget runs out: "abort" (raise HostBusy) or "run" (flagged)
    ):
        self.artifacts_dir = artifacts_dir
        self.busy_policy = busy_policy
        # the idle gate (D-041, D-043): with a calibrated host, a replay does not start while the
        # machine is busy or warm; the reading is taken under sustained load so throttling shows
        self.wait_for_idle, self.idle_wait_s, self.idle_max_waits = wait_for_idle, idle_wait_s, idle_max_waits
        self.gate_frames = gate_frames
        self._host_baseline = load_host_baseline(artifacts_dir)  # isolated ms/frame per resolution, None if uncalibrated
        # render synthetic cameras in a child process so simulation cost is not edge cost (D-018)
        self.remote_cameras = remote_cameras
        self.detector = make_detector(detector_kind, detector_weights, detector_device)
        self.cutouts = CutoutLibrary()
        self._confirmer = confirmer
        self._confirmer_weights = confirmer_weights
        self._cloud: CloudServerProcess | None = None

    @property
    def confirmer(self) -> ConfirmationService:
        if self._confirmer is not None:
            return self._confirmer
        if self._cloud is None:
            log.info("starting stand-in cloud server (%s)", self._confirmer_weights)
            self._cloud = CloudServerProcess(self._confirmer_weights)
        return self._cloud

    def close(self) -> None:
        if self._cloud is not None:
            self._cloud.close()
            self._cloud = None

    def measure_baseline_when_idle(self, source: FrameSource, resolution: int, conf: float) -> float:
        """Session baseline for one run, measured now. With a calibrated host, wait while the
        machine is busy (reading > CONTENTION_RATIO × isolated), up to idle_max_waits × idle_wait_s;
        after that run anyway — the QC check will flag the result (D-041)."""
        isolated = (self._host_baseline or {}).get(resolution)
        waits = 0
        while True:
            # calibrated host: a sustained reading (the last third of ~300 frames) so a warm machine that
            # throttles within seconds cannot pass the gate on its first fast frames (D-043)
            b = (sustained_reading(self.detector, source, resolution, conf, self.gate_frames) if isolated is not None
                 else calibrate_detector(self.detector, source, resolution, conf))
            busy = isolated is not None and b / isolated > CONTENTION_RATIO
            if not busy or not self.wait_for_idle or waits >= self.idle_max_waits:
                if busy:
                    # the abort policy applies only when the gate is on and its budget is spent;
                    # with the gate switched off the run proceeds and the QC check flags it
                    if self.wait_for_idle and self.busy_policy == "abort":
                        raise HostBusy(f"host still busy after {waits} waits: detector {b:.0f} ms vs isolated {isolated:.0f} ms ({b / isolated:.2f}x); refusing to replay on a loaded machine (D-044)")
                    log.warning("host busy (detector %.0f ms vs isolated %.0f ms, %.2fx); running anyway, result will be flagged", b, isolated, b / isolated)
                return b
            waits += 1
            log.info("host busy: detector %.0f ms vs isolated %.0f ms (%.2fx); waiting %.0f s (%d/%d)", b, isolated, b / isolated, self.idle_wait_s, waits, self.idle_max_waits)
            time.sleep(self.idle_wait_s)

    def calibrate_host(self, resolutions: tuple[int, ...] = (640, 480, 320), n: int = 60, tier: str = "quick", seed: int = 1, conf: float = 0.4, reset: bool = False) -> dict[int, float]:
        """Isolated detector ms/frame per resolution, each measured twice. Refuses when the two
        readings disagree by more than STABLE_TOLERANCE (the machine is not idle), and when the
        reading is more than CONTENTION_RATIO slower than this host's recorded idle speed (the
        machine is uniformly slow, i.e. warm; a throttled machine is stable but not idle, D-043).
        `reset=True` accepts a slower reading as the new record."""
        source = RemoteSyntheticSource(tier, seed) if self.remote_cameras else SyntheticSceneSource.from_tier(tier, seed, self.cutouts)
        prior = {} if reset else (load_host_baseline(self.artifacts_dir) or {})
        out: dict[int, float] = {}
        try:
            for res in resolutions:
                a = calibrate_detector(self.detector, source, res, conf, n)
                b = calibrate_detector(self.detector, source, res, conf, n)
                if not is_stable(a, b):
                    raise RuntimeError(f"host not idle: {res} px measured {a:.1f} then {b:.1f} ms/frame (more than {STABLE_TOLERANCE:.0%} apart); close other work and retry")
                v = min(a, b)
                if res in prior and v / prior[res] > CONTENTION_RATIO:
                    raise RuntimeError(f"host is slow: {res} px reads {v:.1f} ms/frame, {v / prior[res]:.2f}x its recorded idle speed ({prior[res]:.1f} ms); let it cool and retry, or pass --reset to accept the new speed")
                out[res] = v
        finally:
            if isinstance(source, RemoteSyntheticSource):
                source.close()
        return out

    def save_host_calibration(self, baselines: dict[int, float], reset: bool = False) -> Path:
        """Record the host's idle speed. The record only ratchets down (the fastest idle reading is
        the best estimate of 'idle'); `reset=True` replaces it."""
        prior = {} if reset else (load_host_baseline(self.artifacts_dir) or {})
        merged = {res: min(v, prior.get(res, v)) for res, v in baselines.items()}
        merged.update({res: v for res, v in prior.items() if res not in merged})
        p = host_calibration_path(self.artifacts_dir)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "measured_at": datetime.now(UTC).isoformat(),
            "git_sha": git_sha(),
            "detector": getattr(self.detector, "name", "?"),
            "baseline_ms": {str(k): v for k, v in merged.items()},
        }, indent=1))
        self._host_baseline = dict(merged)
        return p

    def source_for(self, exp: Experiment) -> FrameSource:
        if exp.corpus.kind == "real":
            if not exp.corpus.path:
                raise ValueError("real corpus needs a path")
            return FileVideoSource(Path(exp.corpus.path))
        if self.remote_cameras:
            return RemoteSyntheticSource(exp.corpus.tier, exp.corpus.seed)
        return SyntheticSceneSource.from_tier(exp.corpus.tier, exp.corpus.seed, self.cutouts)

    def run(self, exp: Experiment, persist: bool = True) -> ExperimentResult:
        started = datetime.now(UTC)
        source = self.source_for(exp)
        cfg, sc = exp.config, exp.scenario
        np.random.seed(sc.seed)

        # Session baseline for THIS run, never cached across runs: a baseline measured under load
        # would hide the load. With a calibrated host, wait for it to be idle first (D-041).
        if self._cloud is not None:
            time.sleep(1.0)  # let the idle cloud server settle before measuring
        baseline_ms = self.measure_baseline_when_idle(source, cfg.detector_resolution, cfg.local_confidence_threshold)

        confirmer = self.confirmer if cfg.cloud_confirmation else None
        exclude = set()
        if isinstance(source, RemoteSyntheticSource):
            exclude.add(source.pid)
        if isinstance(confirmer, CloudServerProcess):
            exclude.add(confirmer.pid)
        injector = NetworkInjector(sc) if cfg.cloud_confirmation else None
        indexer = make_indexer(self.artifacts_dir, exp.id) if cfg.historical_indexing else None

        pressure = ComputePressureInjector(sc.compute_pressure)
        pressure.start()
        exclude.update(p.pid for p in pressure._procs if p.pid)
        monitor = ResourceMonitor(exclude_pids=exclude).start()
        try:
            clock = RealTimeClock()
            pipe = Pipeline(
                source,
                cfg,
                sc,
                self.detector,
                clock,
                confirmer=confirmer,
                injector=injector,
                indexer=indexer,
                time_scale=exp.time_scale,
            )
            out = pipe.run()
        finally:
            resources = monitor.stop()
            pressure.stop()
            if isinstance(source, RemoteSyntheticSource):
                source.close()

        metrics, alerts = compute_metrics(
            out.alerts,
            source.ground_truth,
            real_time=(exp.time_scale == 1.0),
            frames_processed=out.frames_processed,
            frames_dropped=out.frames_dropped,
            run_wall_s=out.run_wall_s,
            detector_ms=out.detector_ms,
            baseline_detector_ms=baseline_ms,
            link=out.link,
            resources=resources,
            cameras_active=out.cameras_active,
        )
        metrics["detector_ms_baseline"] = Metric.measured(baseline_ms, "ms", "unpressured calibration at this resolution")
        metrics["driver_max_lag_ms"] = Metric.measured(out.driver_max_lag_s * 1000.0, "ms", "how late the replay driver released a frame")
        if out.driver_lag_s:
            lag_sorted = sorted(out.driver_lag_s)
            metrics["driver_lag_p50_ms"] = Metric.measured(lag_sorted[len(lag_sorted) // 2] * 1000.0, "ms", "replay driver release lag (harness, not edge)")
            metrics["driver_lag_p95_ms"] = Metric.measured(lag_sorted[int(0.95 * (len(lag_sorted) - 1))] * 1000.0, "ms", "replay driver release lag (harness, not edge)")
        metrics["frames_released"] = Metric.measured(out.frames_released, "count")
        metrics["cloud_confirmed"] = Metric.measured(out.cloud_confirmed, "count", "candidates confirmed by the cloud path")
        metrics["cloud_rejected"] = Metric.measured(out.cloud_rejected, "count", "candidates the confirmer rejected (no alert)")
        metrics["cloud_failed_over"] = Metric.measured(out.cloud_failed_over, "count", "cloud calls that failed/timed out")
        metrics["compute_pressure_workers"] = Metric.measured(pressure.n_workers, "count")
        notes = list(out.notes)
        # QC (D-028, D-041): in a no-pressure scenario the detector should run at its baseline
        # speed. Three checks, any one flags the run: slower than its session baseline; slower than
        # the isolated host baseline; or a session baseline that was itself measured under load.
        isolated = (self._host_baseline or {}).get(cfg.detector_resolution)
        why: list[str] = []
        if sc.compute_pressure == ComputePressure.NORMAL and out.detector_ms:
            mean_ms = statistics.fmean(out.detector_ms)
            if baseline_ms and mean_ms / baseline_ms > CONTENTION_RATIO:
                why.append(f"detector {mean_ms / baseline_ms:.2f}x its session baseline")
            if isolated:
                if mean_ms / isolated > CONTENTION_RATIO:
                    why.append(f"detector {mean_ms / isolated:.2f}x the isolated host baseline")
                if baseline_ms / isolated > CONTENTION_RATIO:
                    why.append(f"session baseline {baseline_ms / isolated:.2f}x the isolated host baseline")
        suspect = bool(why)
        if suspect:
            notes.append("QC: " + "; ".join(why) + " in a no-pressure scenario: external CPU contention, treat as suspect (D-028, D-041)")
        metrics["qc_suspect_contention"] = Metric.measured(1.0 if suspect else 0.0, "flag", "1 = detector slowdown without injected pressure")
        metrics["detector_ms_isolated"] = (
            Metric.measured(isolated, "ms", "isolated host baseline from `failsafe calibrate`") if isolated
            else Metric.unavailable("host not calibrated: run `failsafe calibrate` on an idle machine")
        )

        if cfg.cloud_confirmation:
            notes.append(f"cloud confirmer is a local stand-in ({confirmer.name}); not a VLM (DECISIONS D-006)")
        if sc.compute_pressure != ComputePressure.NORMAL:
            notes.append(f"compute pressure realised by {pressure.n_workers} busy processes; see compute_pressure_observed")

        verification = verify(metrics, exp.mission)
        py, tv, uv = _versions()
        prov = Provenance(
            experiment_id=exp.id,
            mission_hash=exp.mission.hash,
            scenario_hash=sc.hash,
            config_hash=cfg.hash,
            corpus_hash=source.corpus_hash,
            seed=exp.corpus.seed,  # the seed that varies between runs (D-038); the scenario seed is inside scenario_hash
            time_scale=exp.time_scale,
            git_sha=git_sha(),
            python_version=py,
            torch_version=tv,
            ultralytics_version=uv,
            detector_model=self.detector.name,
            confirmer_model=confirmer.name if confirmer is not None else "",
            backend="local",
            hostname=socket.gethostname(),
            platform=platform.platform(),
            started_at=started,
            finished_at=datetime.now(UTC),
        )
        result = ExperimentResult(experiment=exp, metrics=metrics, verification=verification, provenance=prov, alerts=alerts, notes=notes)
        if persist:
            d = self.artifacts_dir / "experiments" / exp.id
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{started.strftime('%Y%m%dT%H%M%SZ')}.json").write_text(result.to_json())
        return result
