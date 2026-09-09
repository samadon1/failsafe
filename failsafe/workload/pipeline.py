"""The restricted-zone monitoring workload under real-time replay.

    FrameSource ─► Sampler (per-camera FPS, backlog policy) ─► detector worker
                                                                  │  YOLO @ resolution
                                                                  │  zone logic + hysteresis
                                                                  ▼
                                                     candidate critical event
                                                        │              │
                                              cloud_confirmation   local only
                                                        │              │
                                    NetworkInjector → ConfirmationService
                                                        │
                                                      ALERT (timestamped)

The driver thread releases frames at wall-clock `T0 + scene_t / time_scale`. One detector worker
thread processes them (a single edge accelerator). Cloud confirmations run on a small thread pool
so that a slow cloud stalls *alerting*, not *detection* — which is exactly the failure mode we
want to be able to observe.
"""

from __future__ import annotations

import logging
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from failsafe.corpus.scene import BACKGROUND_PRIORITY
from failsafe.experiments.schema import AlertRecord, OperatingConfig, Scenario
from failsafe.faults.network import CloudTimeout, CloudUnavailable, LinkStats, NetworkInjector
from failsafe.workload.clock import Clock
from failsafe.workload.cloud import ConfirmationRequest, ConfirmationService, encode_jpeg
from failsafe.workload.detector import Detector
from failsafe.workload.indexing import FrameIndexer
from failsafe.workload.source import FrameSource
from failsafe.workload.zone import ZoneAlertState, detections_in_zone

log = logging.getLogger(__name__)

MAX_BACKLOG_DROP_OLDEST = 3  # frames buffered before the sampler starts discarding
DRAIN_TIMEOUT_S = 20.0


@dataclass
class FrameItem:
    camera: str
    scene_t: float
    release_wall: float
    frame: np.ndarray


@dataclass
class RunOutput:
    alerts: list[AlertRecord]
    frames_released: int
    frames_processed: int
    frames_dropped: int
    run_wall_s: float
    detector_ms: list[float]
    link: LinkStats | None
    cameras_active: list[str]
    driver_max_lag_s: float
    driver_lag_s: list[float]
    drain_truncated: bool
    cloud_confirmed: int = 0
    cloud_rejected: int = 0
    cloud_failed_over: int = 0
    notes: list[str] = field(default_factory=list)


def active_cameras(source: FrameSource, config: OperatingConfig) -> list[str]:
    cams = list(source.cameras)
    if config.drop_background_streams == "all":
        cams = [source.critical_camera]
    elif config.drop_background_streams == "lowest_priority":
        for c in BACKGROUND_PRIORITY:
            if c in cams and c != source.critical_camera:
                cams.remove(c)
                break
    if config.background_fps == 0:
        cams = [c for c in cams if c == source.critical_camera]
    return cams


def build_schedule(source: FrameSource, config: OperatingConfig, start_t: float = 0.0) -> list[tuple[float, str]]:
    """(scene_t, camera) pairs in release order, aligned to the native frame grid, for scene
    times > start_t (start_t=0 → whole corpus)."""
    items: list[tuple[float, int, str]] = []
    cams = active_cameras(source, config)
    for order, cam in enumerate(sorted(cams, key=lambda c: (c != source.critical_camera, c))):
        fps = config.critical_fps if cam == source.critical_camera else config.background_fps
        if fps <= 0:
            continue
        step = source.native_fps / fps
        n = int(source.duration_s * fps)
        for k in range(n):
            frame_idx = int(round(k * step))
            t = frame_idx / source.native_fps
            if t > start_t or (start_t == 0.0 and t == 0.0):
                items.append((t, order, cam))
    items.sort()
    return [(t, cam) for t, _, cam in items]


class Pipeline:
    def __init__(
        self,
        source: FrameSource,
        config: OperatingConfig,
        scenario: Scenario,
        detector: Detector,
        clock: Clock,
        *,
        confirmer: ConfirmationService | None = None,
        injector: NetworkInjector | None = None,
        indexer: FrameIndexer | None = None,
        time_scale: float = 1.0,
        max_confirm_workers: int = 2,
        stream: bool = True,
        on_tick=None,
    ):
        self.source = source
        self.config = config
        self.stream = stream  # False → request/response frames (needed when the schedule can change mid-run)
        self.on_tick = on_tick  # optional callback(scene_t) invoked by the driver ~every frame (runtime probes)
        self._reconfig: OperatingConfig | None = None
        self._reconfig_lock = threading.Lock()
        self.config_changes: list[tuple[float, str]] = []  # (scene_t, config name)
        self.scene_t: float = 0.0
        self.scenario = scenario
        self.detector = detector
        self.clock = clock
        self.time_scale = time_scale
        self.confirmer = confirmer
        self.injector = injector if injector is not None else (NetworkInjector(scenario) if confirmer is not None else None)
        self.indexer = indexer
        if config.cloud_confirmation and confirmer is None:
            raise ValueError("cloud_confirmation=True requires a ConfirmationService")

        self._queue: deque[FrameItem] = deque()
        self._cv = threading.Condition()
        self._done_releasing = False
        self._alerts: list[AlertRecord] = []
        self._alerts_lock = threading.Lock()
        self._states = {
            c: ZoneAlertState(confirm_frames=config.alert_confirm_frames) for c in source.cameras
        }
        self._inflight: set[str] = set()
        self._inflight_lock = threading.Lock()
        # a pool exists whenever a confirmer is available, so a later reconfiguration can turn the cloud path on
        self._pool = ThreadPoolExecutor(max_workers=max_confirm_workers) if confirmer is not None else None
        self.frames_processed = 0
        self.frames_dropped = 0
        self.cloud_confirmed = 0
        self.cloud_rejected = 0
        self.cloud_failed_over = 0
        self.detector_ms: list[float] = []
        self.cameras_active = active_cameras(source, config)
        self.notes: list[str] = []

    # -- reconfiguration ---------------------------------------------------------------------

    def apply_config(self, cfg: OperatingConfig) -> None:
        """Switch operating configuration; takes effect at the next released frame. Cloud path
        can only be enabled if a confirmer was supplied at construction."""
        if cfg.cloud_confirmation and self.confirmer is None:
            raise ValueError("cannot enable cloud_confirmation without a ConfirmationService")
        with self._reconfig_lock:
            self._reconfig = cfg

    def _take_reconfig(self) -> OperatingConfig | None:
        with self._reconfig_lock:
            cfg, self._reconfig = self._reconfig, None
        return cfg

    # -- driver -------------------------------------------------------------------------------

    def run(self) -> RunOutput:
        schedule = build_schedule(self.source, self.config)
        real_time = self.clock.is_real_time
        start_stream = getattr(self.source, "start_stream", None)
        if start_stream is not None and self.stream:
            start_stream(schedule)  # remote camera process renders ahead of consumption
        # Under a simulated clock processing takes zero simulated time, so frames are processed
        # lock-step on the driver thread (no backlog can form). Real time uses a worker thread.
        worker = threading.Thread(target=self._worker, name="detector-worker", daemon=True) if real_time else None
        t_start = self.clock.now()
        if worker is not None:
            worker.start()
        max_lag = 0.0
        lags: list[float] = []
        released = 0
        pending = deque(schedule)
        while pending:
            new_cfg = self._take_reconfig()
            if new_cfg is not None:
                self.config = new_cfg
                self.config_changes.append((self.scene_t, new_cfg.name))
                self.cameras_active = active_cameras(self.source, new_cfg)
                pending = deque(build_schedule(self.source, new_cfg, start_t=self.scene_t))
                if not pending:
                    break
            scene_t, cam = pending.popleft()
            self.scene_t = scene_t
            if self.on_tick is not None:
                self.on_tick(scene_t)
            due = t_start + scene_t / self.time_scale
            self.clock.sleep_until(due)
            lag = self.clock.now() - due
            max_lag = max(max_lag, lag)
            lags.append(lag)
            frame = self.source.frame(cam, scene_t)
            item = FrameItem(camera=cam, scene_t=scene_t, release_wall=due, frame=frame)
            released += 1
            if worker is None:
                self._process(item)
                continue
            with self._cv:
                if self.config.backlog_policy == "drop_oldest":
                    while len(self._queue) >= MAX_BACKLOG_DROP_OLDEST:
                        self._queue.popleft()
                        self.frames_dropped += 1
                self._queue.append(item)
                self._cv.notify()
        # drain
        with self._cv:
            self._done_releasing = True
            self._cv.notify_all()
        drain_truncated = False
        if worker is not None:
            worker.join(timeout=DRAIN_TIMEOUT_S)
            drain_truncated = worker.is_alive()
        if drain_truncated:
            with self._cv:
                self.frames_dropped += len(self._queue)
                self._queue.clear()
            self.notes.append(f"drain truncated after {DRAIN_TIMEOUT_S}s; remaining backlog discarded")
        if self._pool is not None:
            self._pool.shutdown(wait=True)
        run_wall = self.clock.now() - t_start
        if self.indexer is not None:
            self.indexer.close()
        return RunOutput(
            alerts=list(self._alerts),
            frames_released=released,
            frames_processed=self.frames_processed,
            frames_dropped=self.frames_dropped,
            run_wall_s=run_wall,
            detector_ms=list(self.detector_ms),
            link=self.injector.stats if self.injector is not None else None,
            cameras_active=self.cameras_active,
            driver_max_lag_s=max_lag,
            driver_lag_s=lags,
            drain_truncated=drain_truncated,
            notes=self.notes,
            cloud_confirmed=self.cloud_confirmed,
            cloud_rejected=self.cloud_rejected,
            cloud_failed_over=self.cloud_failed_over,
        )

    # -- worker -------------------------------------------------------------------------------

    def _next_item(self) -> FrameItem | None:
        with self._cv:
            while not self._queue and not self._done_releasing:
                self._cv.wait(timeout=0.5 if self.clock.is_real_time else None)
                if not self.clock.is_real_time and not self._queue and not self._done_releasing:
                    continue
            if self._queue:
                return self._queue.popleft()
            return None

    def _worker(self) -> None:
        while True:
            item = self._next_item()
            if item is None:
                return
            try:
                self._process(item)
            except Exception as e:  # pragma: no cover - keep the worker alive
                log.exception("frame processing failed: %s", e)

    def _process(self, item: FrameItem) -> None:
        cfg = self.config
        dets = self.detector.detect(item.frame, cfg.detector_resolution, cfg.local_confidence_threshold)
        ms = getattr(self.detector, "infer_ms", None)
        if ms:
            self.detector_ms.append(ms[-1])
        self.frames_processed += 1
        if self.indexer is not None and cfg.historical_indexing:
            self.indexer.index(item.camera, item.scene_t, item.frame)

        zone = self.source.zone(item.camera)
        in_zone = detections_in_zone(dets, zone)
        state = self._states[item.camera]
        with self._inflight_lock:
            busy = item.camera in self._inflight
        if busy:
            # a confirmation for this camera is in flight; don't advance the alert state so the
            # candidate is re-raised on the next frame if the cloud rejects or fails
            return
        if not state.update(item.scene_t, bool(in_zone)):
            return
        best = max(in_zone, key=lambda d: d.confidence)
        if not cfg.cloud_confirmation:
            self._emit(item, "local", best.confidence)
            return
        with self._inflight_lock:
            self._inflight.add(item.camera)
        req = ConfirmationRequest(
            camera=item.camera, scene_t=item.scene_t, jpeg=encode_jpeg(item.frame), candidate=best, zone=zone
        )
        if not self.clock.is_real_time:
            # simulated time: the round trip takes zero simulated time → resolve inline
            self._confirm(item, req)
            return
        assert self._pool is not None
        self._pool.submit(self._confirm, item, req)

    def _confirm(self, item: FrameItem, req: ConfirmationRequest) -> None:
        cfg = self.config
        assert self.injector is not None and self.confirmer is not None
        try:
            resp = self.injector.call(len(req.jpeg), cfg.cloud_timeout_ms / 1000.0, self.confirmer.confirm, req)
            if resp.confirmed:
                self.cloud_confirmed += 1
                self._emit(item, "cloud", resp.confidence)
            else:
                # rejected: allow a later frame to re-raise a candidate
                self.cloud_rejected += 1
                self._states[item.camera].reject()
        except (CloudTimeout, CloudUnavailable):
            self.cloud_failed_over += 1
            if cfg.on_cloud_failure == "alert_local":
                self._emit(item, "local", req.candidate.confidence)
            else:
                self._states[item.camera].reject()
        except Exception as e:  # pragma: no cover
            log.exception("confirmation failed: %s", e)
            if cfg.on_cloud_failure == "alert_local":
                self._emit(item, "local", req.candidate.confidence)
        finally:
            with self._inflight_lock:
                self._inflight.discard(item.camera)

    def _emit(self, item: FrameItem, by: str, confidence: float) -> None:
        rec = AlertRecord(
            camera=item.camera,
            scene_t=item.scene_t,
            emitted_wall=self.clock.now(),
            frame_release_wall=item.release_wall,
            confirmed_by=by,  # type: ignore[arg-type]
            confidence=confidence,
        )
        with self._alerts_lock:
            self._alerts.append(rec)


def make_indexer(artifacts_dir: Path, run_id: str) -> FrameIndexer:
    return FrameIndexer(artifacts_dir / "index" / f"{run_id}.sqlite")
