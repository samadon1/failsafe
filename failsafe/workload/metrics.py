"""Metrics collector: turns alerts + ground truth + resource samples into labelled Metrics.

Definitions live in docs/EXPERIMENTS.md. Everything here is MEASURED unless stated; latency is
UNAVAILABLE when the run was not real-time.
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

import psutil

from failsafe.corpus.ground_truth import GRACE_S, LEAD_S, GroundTruth
from failsafe.experiments.schema import AlertRecord, Metric
from failsafe.faults.network import LinkStats


@dataclass
class ResourceSamples:
    proc_cpu_percent: list[float] = field(default_factory=list)
    sys_cpu_percent: list[float] = field(default_factory=list)
    rss_mb: list[float] = field(default_factory=list)


class ResourceMonitor:
    """Samples this process tree's CPU % and RSS at `interval` seconds on a daemon thread."""

    def __init__(self, interval: float = 0.5, exclude_pids: set[int] | None = None):
        self.interval = interval
        self.samples = ResourceSamples()
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._proc = psutil.Process()
        # PIDs that are NOT part of the edge device (e.g. the stand-in cloud server, pressure workers)
        self._exclude = set(exclude_pids or ())

    def start(self) -> ResourceMonitor:
        self._proc.cpu_percent(None)
        psutil.cpu_percent(None)
        self._thread.start()
        return self

    def stop(self) -> ResourceSamples:
        self._stop.set()
        self._thread.join(timeout=2)
        return self.samples

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            try:
                cpu = self._proc.cpu_percent(None)
                rss = self._proc.memory_info().rss
                for ch in self._proc.children(recursive=True):
                    if ch.pid in self._exclude:
                        continue
                    try:
                        cpu += ch.cpu_percent(None)
                        rss += ch.memory_info().rss
                    except psutil.Error:
                        pass
                self.samples.proc_cpu_percent.append(cpu)
                self.samples.sys_cpu_percent.append(psutil.cpu_percent(None))
                self.samples.rss_mb.append(rss / 1e6)
            except psutil.Error:
                pass


def _p(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    s = sorted(values)
    k = (len(s) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def compute_metrics(
    alerts: list[AlertRecord],
    gt: GroundTruth,
    *,
    real_time: bool,
    frames_processed: int,
    frames_dropped: int,
    run_wall_s: float,
    detector_ms: list[float],
    baseline_detector_ms: float | None,
    link: LinkStats | None,
    resources: ResourceSamples,
    cameras_active: list[str],
    grace_s: float = GRACE_S,
    lead_s: float = LEAD_S,
) -> tuple[dict[str, Metric], list[AlertRecord]]:
    """Match alerts to GT events, annotate the alert records, and build the metric dict.

    `grace_s` / `lead_s` are the matching window (D-021); they default to the scoring rule in
    force and are overridable so the window's influence can be measured without rewriting results."""
    # -- matching -----------------------------------------------------------------------------
    first_alert_for_event: dict[str, AlertRecord] = {}
    annotated: list[AlertRecord] = []
    for a in sorted(alerts, key=lambda r: r.emitted_wall):
        ev = gt.match_alert(a.camera, a.scene_t, grace_s, lead_s)
        rec = a.model_copy()
        if ev is not None:
            rec.matched_event_id = ev.id
            if ev.id not in first_alert_for_event:
                first_alert_for_event[ev.id] = rec
        annotated.append(rec)

    # latency: alert emission − wall-clock at which the event's first in-zone frame was *due*.
    # The due time is what a real camera would have delivered; we use the release time of the
    # triggering frame minus its scene offset from the event start.
    latencies_ms: list[float] = []
    for ev_id, rec in first_alert_for_event.items():
        ev = next(e for e in gt.events if e.id == ev_id)
        event_start_wall = rec.frame_release_wall - (rec.scene_t - ev.start)
        # an alert raised slightly before the analytic start (LEAD_S) is early, not late → 0
        lat = max(0.0, (rec.emitted_wall - event_start_wall) * 1000.0)
        rec.latency_ms = lat
        latencies_ms.append(lat)

    n_events = len(gt.events)
    n_detected = len(first_alert_for_event)
    n_alerts = len(annotated)
    n_true = sum(1 for r in annotated if r.matched_event_id is not None)

    m: dict[str, Metric] = {}
    m["critical_event_recall"] = Metric.measured(n_detected / n_events if n_events else 0.0, "ratio", f"{n_detected}/{n_events} GT events")
    m["alert_precision"] = Metric.measured(n_true / n_alerts if n_alerts else 1.0, "ratio", f"{n_true}/{n_alerts} alerts attributable")
    m["alerts_total"] = Metric.measured(n_alerts, "count")
    m["false_alerts"] = Metric.measured(n_alerts - n_true, "count")
    m["gt_events_total"] = Metric.measured(n_events, "count")
    # events on cameras that were switched off cannot be detected — report separately
    off_cams = [c for c in {e.camera for e in gt.events} if c not in cameras_active]
    m["gt_events_on_dropped_cameras"] = Metric.measured(sum(1 for e in gt.events if e.camera in off_cams), "count")

    if real_time:
        if latencies_ms:
            m["alert_latency_mean_ms"] = Metric.measured(statistics.fmean(latencies_ms), "ms")
            m["alert_latency_median_ms"] = Metric.measured(statistics.median(latencies_ms), "ms")
            m["alert_latency_p95_ms"] = Metric.measured(_p(latencies_ms, 0.95), "ms", f"n={len(latencies_ms)}")
            m["alert_latency_max_ms"] = Metric.measured(max(latencies_ms), "ms")
        else:
            note = "no GT event was detected"
            for k in ("alert_latency_mean_ms", "alert_latency_median_ms", "alert_latency_p95_ms", "alert_latency_max_ms"):
                m[k] = Metric.unavailable(note)
    else:
        for k in ("alert_latency_mean_ms", "alert_latency_median_ms", "alert_latency_p95_ms", "alert_latency_max_ms"):
            m[k] = Metric.unavailable("time_scale != 1.0 (not real-time replay)")

    m["frames_processed"] = Metric.measured(frames_processed, "count")
    m["frames_dropped"] = Metric.measured(frames_dropped, "count")
    m["frames_processed_per_s"] = Metric.measured(frames_processed / run_wall_s if run_wall_s > 0 else 0.0, "fps")
    if detector_ms:
        m["detector_ms_mean"] = Metric.measured(statistics.fmean(detector_ms), "ms", f"n={len(detector_ms)}")
        m["detector_ms_p95"] = Metric.measured(_p(detector_ms, 0.95), "ms")
        if baseline_detector_ms:
            m["compute_pressure_observed"] = Metric.derived(
                statistics.fmean(detector_ms) / baseline_detector_ms, "x", "detector ms/frame ÷ unpressured baseline"
            )
    else:
        m["detector_ms_mean"] = Metric.unavailable("no frames processed")
        m["detector_ms_p95"] = Metric.unavailable("no frames processed")

    if link is not None:
        m["cloud_requests"] = Metric.measured(link.requests, "count")
        m["cloud_bytes_total"] = Metric.measured(link.bytes_sent + link.bytes_received, "bytes")
        m["cloud_timeouts"] = Metric.measured(link.timeouts, "count")
        m["cloud_unavailable"] = Metric.measured(link.unavailable, "count")
        if link.rtts_ms:
            m["cloud_rtt_p50_ms"] = Metric.measured(statistics.median(link.rtts_ms), "ms")
            m["cloud_rtt_p95_ms"] = Metric.measured(_p(link.rtts_ms, 0.95), "ms")
        else:
            m["cloud_rtt_p50_ms"] = Metric.unavailable("no completed cloud round trips")
            m["cloud_rtt_p95_ms"] = Metric.unavailable("no completed cloud round trips")
    else:
        m["cloud_requests"] = Metric.measured(0, "count")
        m["cloud_bytes_total"] = Metric.measured(0, "bytes", "cloud path disabled")
        m["cloud_timeouts"] = Metric.measured(0, "count")
        m["cloud_unavailable"] = Metric.measured(0, "count")

    if resources.proc_cpu_percent:
        m["cpu_percent_mean"] = Metric.measured(statistics.fmean(resources.proc_cpu_percent), "%", "process tree, psutil")
        m["system_cpu_percent_mean"] = Metric.measured(statistics.fmean(resources.sys_cpu_percent), "%", "whole machine")
        m["rss_mb_peak"] = Metric.measured(max(resources.rss_mb), "MB")
    else:
        for k in ("cpu_percent_mean", "system_cpu_percent_mean", "rss_mb_peak"):
            m[k] = Metric.unavailable("no resource samples")
    m["gpu_util_percent"] = Metric.unavailable("no NVML GPU on this host")
    m["gpu_mem_mb"] = Metric.unavailable("no NVML GPU on this host")
    m["run_wall_s"] = Metric.measured(run_wall_s, "s")
    return m, annotated


def now() -> float:
    return time.perf_counter()
