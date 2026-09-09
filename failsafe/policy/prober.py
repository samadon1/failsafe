"""Runtime condition prober: turns what the edge can measure into an `Observed` sample.

Signals: the network injector's rolling window of recent cloud outcomes (confirmations *and*
lightweight health pings, so recovery is noticed even when the cloud path is switched off), the
detector's recent ms/frame against its calibrated baseline, and the scenario's bandwidth (a real
system would estimate uplink; here it is the injected value, labelled as such).
"""

from __future__ import annotations

import statistics
import threading
import time
from dataclasses import dataclass, field

from failsafe.faults.network import NetworkInjector
from failsafe.policy.runtime import Observed


@dataclass
class Prober:
    injector: NetworkInjector | None
    detector_ms: list[float]  # shared, append-only list from the pipeline
    baseline_ms: float
    window_s: float = 10.0
    ping_interval_s: float = 2.0
    ping_timeout_s: float = 1.5
    _last_ping: float = field(default=float("-inf"))
    _ping_lock: threading.Lock = field(default_factory=threading.Lock)
    _ping_thread: threading.Thread | None = None

    def maybe_ping(self, wall_now: float, wait: bool = False) -> None:
        """Fire a health ping at most every ping_interval_s. In real time it runs in the background
        (a zombie cloud must not stall the driver); under a simulated clock `wait=True` joins it so
        the next observation sees the outcome."""
        if self.injector is None or wall_now - self._last_ping < self.ping_interval_s:
            return
        if self._ping_thread is not None and self._ping_thread.is_alive():
            return
        self._last_ping = wall_now
        self._ping_thread = threading.Thread(target=self.injector.ping, args=(self.ping_timeout_s,), daemon=True)
        self._ping_thread.start()
        if wait:
            self._ping_thread.join(timeout=self.ping_timeout_s + 1.0)

    def observe(self) -> Observed | None:
        """An Observed sample, or None while there is no cloud evidence at all (startup)."""
        reachable: bool | None = None
        rtt_p95: float | None = None
        timeout_rate = 0.0
        if self.injector is not None:
            recent = self.injector.window(self.window_s)
            if not recent:
                return None
            if recent:
                kinds = [k for _, k, _ in recent]
                # network state is judged from ping round trips (no server inference time in
                # them); confirmation RTTs are a fallback when no ping has completed yet (D-034)
                pings = sorted(r for _, k, r in recent if k == "ping_ok" and r is not None)
                rtts = pings or sorted(r for _, k, r in recent if k == "ok" and r is not None)
                reachable = kinds[-1] != "unavailable"
                timeout_rate = kinds.count("timeout") / len(kinds)
                if rtts:
                    rtt_p95 = rtts[min(len(rtts) - 1, int(0.95 * (len(rtts) - 1)))]
        recent_det = self.detector_ms[-40:]
        slowdown = (statistics.fmean(recent_det) / self.baseline_ms) if recent_det and self.baseline_ms else 1.0
        bw = self.injector.scenario.bandwidth_mbps if self.injector is not None else None
        return Observed(cloud_reachable=reachable, cloud_rtt_p95_ms=rtt_p95, cloud_timeout_rate=timeout_rate, detector_slowdown=slowdown, bandwidth_mbps=bw)


def now() -> float:
    return time.perf_counter()
