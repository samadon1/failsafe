"""Compute pressure: competing CPU-bound processes.

Levels map to a number of busy workers relative to the core count. The *observed* impact (detector
ms/frame under pressure ÷ baseline) is what gets recorded — we never claim "50 % GPU" because a
knob says so.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time

import numpy as np

from failsafe.experiments.schema import ComputePressure

WORKER_FRACTION: dict[ComputePressure, float] = {
    ComputePressure.NORMAL: 0.0,
    ComputePressure.MODERATE: 0.5,
    ComputePressure.SEVERE: 1.0,
    ComputePressure.CRITICAL: 2.0,
}


DEFAULT_MAX_S = 1800.0  # a worker never outlives this, whatever happens to its parent (D-044)


def _keep_going(stop_flag, parent_pid: int, deadline: float) -> bool:
    """A worker runs only while its parent asks, its parent is alive, and its lifetime cap holds.
    The parent check is what stops a force-killed campaign from leaving busy loops behind: a
    SIGKILLed parent never sets the flag, and `daemon=True` does not help on an unclean exit (D-044)."""
    return not stop_flag.is_set() and os.getppid() == parent_pid and time.monotonic() < deadline


def _busy(stop_flag, seed: int, parent_pid: int, max_s: float) -> None:
    rng = np.random.default_rng(seed)
    a = rng.random((256, 256)).astype(np.float32)
    deadline = time.monotonic() + max_s
    while _keep_going(stop_flag, parent_pid, deadline):
        a = a @ a
        a /= np.abs(a).max() + 1e-6


class ComputePressureInjector:
    def __init__(self, level: ComputePressure, cores: int | None = None, max_s: float = DEFAULT_MAX_S):
        self.level = level
        cores = cores or os.cpu_count() or 4
        self.n_workers = int(round(cores * WORKER_FRACTION[level]))
        self.max_s = max_s
        self._ctx = mp.get_context("spawn")
        self._stop = self._ctx.Event()
        self._procs: list[mp.Process] = []

    def __enter__(self) -> ComputePressureInjector:
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()

    def start(self) -> None:
        for i in range(self.n_workers):
            p = self._ctx.Process(target=_busy, args=(self._stop, i, os.getpid(), self.max_s), daemon=True)
            p.start()
            self._procs.append(p)
        if self._procs:
            time.sleep(0.5)  # let them spin up before measurement starts

    def stop(self) -> None:
        self._stop.set()
        for p in self._procs:
            p.join(timeout=2)
            if p.is_alive():
                p.terminate()
        self._procs.clear()
