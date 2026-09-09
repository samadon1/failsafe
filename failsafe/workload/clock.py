"""Clocks. RealTimeClock is the only clock allowed for latency measurements (DECISIONS D-007).
SimulatedClock exists for unit tests so pipeline logic can be exercised without waiting."""

from __future__ import annotations

import threading
import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...
    def sleep_until(self, t: float) -> None: ...
    def sleep(self, seconds: float) -> None: ...
    @property
    def is_real_time(self) -> bool: ...


class RealTimeClock:
    """Monotonic wall clock, seconds since construction."""

    is_real_time = True

    def __init__(self) -> None:
        self._t0 = time.perf_counter()

    def now(self) -> float:
        return time.perf_counter() - self._t0

    def sleep_until(self, t: float) -> None:
        d = t - self.now()
        if d > 0:
            time.sleep(d)

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            time.sleep(seconds)


class SimulatedClock:
    """Manually advanced clock. `sleep_until` advances time instead of blocking. Thread-safe."""

    is_real_time = False

    def __init__(self, start: float = 0.0) -> None:
        self._t = start
        self._lock = threading.Lock()

    def now(self) -> float:
        with self._lock:
            return self._t

    def advance(self, seconds: float) -> None:
        with self._lock:
            self._t += max(0.0, seconds)

    def sleep_until(self, t: float) -> None:
        with self._lock:
            self._t = max(self._t, t)

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)
