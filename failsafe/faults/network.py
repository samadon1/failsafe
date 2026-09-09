"""Application-level network fault injector for the cloud path.

Wraps any ConfirmationService call with the Scenario's cloud state:

  healthy / slow / severely_slow   → RTT delay (base + jitter), plus bandwidth transfer time
  zombie                           → request accepted, response arrives after a long delay
  timeout                          → never responds (caller's timeout fires)
  offline                          → fails immediately (connection refused)

Bandwidth is a shared link: transfers are serialised through a lock so concurrent requests
congest each other, as they would on a real uplink. All behaviour is seeded and recorded.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field

import numpy as np

from failsafe.experiments.schema import CloudState, Scenario


class CloudUnavailable(Exception):
    """Immediate failure (offline)."""


class CloudTimeout(Exception):
    """Caller-side timeout expired."""


@dataclass
class LinkStats:
    requests: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0
    timeouts: int = 0
    unavailable: int = 0
    rtts_ms: list[float] = field(default_factory=list)  # observed end-to-end round trips
    transfer_wait_s: float = 0.0  # time spent waiting for the shared link


PING_BYTES = 200


class NetworkInjector:
    def __init__(self, scenario: Scenario, seed: int | None = None):
        self.scenario = scenario
        self.rng = np.random.default_rng(scenario.seed if seed is None else seed)
        self._link_lock = threading.Lock()
        self.stats = LinkStats()
        self._stats_lock = threading.Lock()
        # rolling window of recent outcomes for the runtime prober: (wall_s, kind, rtt_ms|None)
        self.recent: list[tuple[float, str, float | None]] = []
        self.clock_fn = time.perf_counter  # the pipeline's clock under simulation (demo sets it)

    def set_scenario(self, scenario: Scenario) -> None:
        """Change the cloud condition mid-run (demo timelines). Takes effect on the next call."""
        self.scenario = scenario

    def _note(self, kind: str, rtt_ms: float | None) -> None:
        with self._stats_lock:
            self.recent.append((self.clock_fn(), kind, rtt_ms))
            if len(self.recent) > 200:
                del self.recent[:-200]

    def window(self, seconds: float) -> list[tuple[float, str, float | None]]:
        cutoff = self.clock_fn() - seconds
        with self._stats_lock:
            return [x for x in self.recent if x[0] >= cutoff]

    def ping(self, timeout_s: float = 1.0) -> tuple[str, float | None]:
        """Lightweight health probe through the same fault model (a real system would ping its
        endpoint). Returns (outcome, rtt_ms): outcome ∈ ping_ok | timeout | unavailable. Ping
        round trips carry no server inference time, so they are tagged separately and are what
        the runtime classifies network state from (D-034)."""
        t0 = time.perf_counter()
        try:
            self.call(PING_BYTES, timeout_s, lambda: (None, PING_BYTES))
            rtt = (time.perf_counter() - t0) * 1000.0
            self._note("ping_ok", rtt)
            return "ping_ok", rtt
        except CloudUnavailable:
            return "unavailable", None
        except CloudTimeout:
            return "timeout", None

    # -- shaping ------------------------------------------------------------------------------

    def _transfer(self, n_bytes: int) -> None:
        """Occupy the shared link for the time it takes to move n_bytes at the scenario bandwidth."""
        mbps = self.scenario.bandwidth_mbps
        if mbps is None:
            return
        seconds = n_bytes * 8 / (mbps * 1e6)
        t0 = time.perf_counter()
        with self._link_lock:
            waited = time.perf_counter() - t0
            time.sleep(seconds)
        with self._stats_lock:
            self.stats.transfer_wait_s += waited

    def _rtt_delay(self) -> float:
        base, jitter = self.scenario.effective_rtt()
        if jitter <= 0:
            return base / 1000.0
        return max(0.0, self.rng.normal(base, jitter / 2)) / 1000.0

    # -- public -------------------------------------------------------------------------------

    def call(self, request_bytes: int, timeout_s: float, fn, *args, **kwargs):
        """Run `fn` as if it were a remote call. Returns (result, response_bytes_estimate).

        `fn` must return (result, response_bytes).
        """
        state = self.scenario.cloud_state
        t_start = time.perf_counter()
        with self._stats_lock:
            self.stats.requests += 1

        if state == CloudState.OFFLINE:
            with self._stats_lock:
                self.stats.unavailable += 1
            self._note("unavailable", None)
            raise CloudUnavailable("connection refused")

        if state == CloudState.TIMEOUT:
            time.sleep(timeout_s)
            with self._stats_lock:
                self.stats.timeouts += 1
                self.stats.bytes_sent += request_bytes
            self._note("timeout", None)
            raise CloudTimeout(f"no response after {timeout_s:.2f}s")

        delay = self._rtt_delay()  # for zombie this is 5–10 s
        if delay >= timeout_s:
            time.sleep(timeout_s)
            with self._stats_lock:
                self.stats.timeouts += 1
                self.stats.bytes_sent += request_bytes
            self._note("timeout", None)
            raise CloudTimeout(f"no response after {timeout_s:.2f}s (server would answer at {delay:.2f}s)")

        # uplink transfer + half RTT, server work, half RTT + downlink
        time.sleep(delay / 2)
        self._transfer(request_bytes)
        remaining = timeout_s - (time.perf_counter() - t_start)
        if remaining <= 0:
            with self._stats_lock:
                self.stats.timeouts += 1
                self.stats.bytes_sent += request_bytes
            self._note("timeout", None)
            raise CloudTimeout("timeout during upload")
        result, response_bytes = fn(*args, **kwargs)
        time.sleep(delay / 2)
        self._transfer(response_bytes)
        elapsed = time.perf_counter() - t_start
        with self._stats_lock:
            self.stats.bytes_sent += request_bytes
            self.stats.bytes_received += response_bytes
            self.stats.rtts_ms.append(elapsed * 1000.0)
        if elapsed > timeout_s:
            with self._stats_lock:
                self.stats.timeouts += 1
            self._note("timeout", None)
            raise CloudTimeout(f"response arrived after timeout ({elapsed:.2f}s > {timeout_s:.2f}s)")
        self._note("ok", elapsed * 1000.0)
        return result
