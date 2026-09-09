"""Compute-pressure workers must never outlive the campaign that started them (D-044)."""

from __future__ import annotations

import os
import time

from failsafe.experiments.schema import ComputePressure
from failsafe.faults import compute as C


class _Flag:
    def __init__(self, v: bool = False):
        self.v = v

    def is_set(self) -> bool:
        return self.v


def test_worker_predicate_stops_on_flag_dead_parent_or_deadline():
    now = time.monotonic()
    parent = os.getppid()
    assert C._keep_going(_Flag(False), parent, now + 10)
    assert not C._keep_going(_Flag(True), parent, now + 10)  # asked to stop
    assert not C._keep_going(_Flag(False), parent + 12345, now + 10)  # parent is gone (ppid changed)
    assert not C._keep_going(_Flag(False), parent, now - 1)  # lifetime cap reached


def test_a_worker_dies_at_its_lifetime_cap_even_if_never_stopped():
    inj = C.ComputePressureInjector(ComputePressure.MODERATE, cores=2, max_s=1.0)  # one worker
    inj.start()
    assert inj._procs and inj._procs[0].is_alive()
    inj._procs[0].join(timeout=10)
    assert not inj._procs[0].is_alive(), "a worker must exit on its own at max_s"
    inj._procs.clear()
