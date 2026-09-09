"""Zone logic: which detections count as a person inside the restricted zone, plus per-camera
alert state (confirm-frames hysteresis and cooldown so one continuous intrusion = one alert)."""

from __future__ import annotations

from dataclasses import dataclass, field

from failsafe.corpus.scene import point_in_convex_polygon


@dataclass(frozen=True)
class Detection:
    x0: float
    y0: float
    x1: float
    y1: float
    confidence: float

    @property
    def foot(self) -> tuple[float, float]:
        return ((self.x0 + self.x1) / 2.0, self.y1)

    @property
    def height(self) -> float:
        return self.y1 - self.y0


def detections_in_zone(dets: list[Detection], zone: list[tuple[float, float]]) -> list[Detection]:
    return [d for d in dets if point_in_convex_polygon(d.foot, zone)]


@dataclass
class ZoneAlertState:
    """Per-camera alert state.

    * `confirm_frames` consecutive in-zone frames are required before a candidate is raised
      (1 = react on the first frame; higher trades latency for precision).
    * after a candidate, further candidates are suppressed for `refractory_s` of scene time. A
      continuous intrusion therefore produces one alert per refractory period rather than one per
      frame; a false alarm cannot mask a real intrusion for longer than the refractory period.
      (Ground-truth events are generated ≥ MIN_EVENT_GAP_S apart, which exceeds the refractory.)
    """

    confirm_frames: int = 1
    refractory_s: float = 1.5
    _streak: int = 0
    _last_candidate_t: float | None = None

    def reject(self) -> None:
        """The candidate was rejected (cloud said no / dropped): allow a later frame to re-raise."""
        self._streak = 0
        self._last_candidate_t = None

    def update(self, scene_t: float, in_zone: bool) -> bool:
        """Returns True if a new alert candidate should be raised for this frame."""
        if not in_zone:
            self._streak = 0
            return False
        self._streak += 1
        if self._streak < self.confirm_frames:
            return False
        if self._last_candidate_t is not None and scene_t - self._last_candidate_t < self.refractory_s:
            return False
        self._last_candidate_t = scene_t
        return True
