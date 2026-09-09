"""Analytic ground truth: maximal intervals where a track's foot point is inside its camera's zone,
sampled on the native frame grid. Also the matching rules used by the metrics collector."""

from __future__ import annotations

from dataclasses import dataclass, field

from failsafe.corpus.scene import Scene, Track, point_in_convex_polygon

GRACE_S = 2.0  # an alert up to this long after the person left the zone still counts
LEAD_S = 0.5  # an alert up to this long *before* the analytic start still counts: the detector's
#             box bottom crosses the painted line a frame or two before the exact foot point does


@dataclass(frozen=True)
class GroundTruthEvent:
    id: str
    camera: str
    track_id: str
    start: float  # scene time of first in-zone frame
    end: float  # scene time of last in-zone frame
    duration_s: float
    height_px: float
    occlusion: float
    alpha: float
    speed_px_s: float

    def window(self, grace: float = GRACE_S, lead: float = LEAD_S) -> tuple[float, float]:
        return self.start - lead, self.end + grace


@dataclass
class GroundTruth:
    events: list[GroundTruthEvent]
    distractor_tracks: list[Track] = field(default_factory=list)

    def for_camera(self, camera: str) -> list[GroundTruthEvent]:
        return [e for e in self.events if e.camera == camera]

    def match_alert(
        self, camera: str, scene_t: float, grace: float = GRACE_S, lead: float = LEAD_S
    ) -> GroundTruthEvent | None:
        """The GT event (on this camera) whose window contains scene_t, earliest start wins."""
        best = None
        for e in self.events:
            if e.camera != camera:
                continue
            s, w_end = e.window(grace, lead)
            # 1 ms tolerance: GT times are rounded to 4 decimals, frame times are exact k/fps
            if s - 1e-3 <= scene_t <= w_end + 1e-3 and (best is None or e.start < best.start):
                best = e
        return best


def compute_ground_truth(scene: Scene) -> GroundTruth:
    events: list[GroundTruthEvent] = []
    distractors: list[Track] = []
    dt = 1.0 / scene.native_fps
    for tr in scene.tracks:
        poly = scene.camera(tr.camera).zone
        f0 = int(tr.t_start * scene.native_fps)
        f1 = int(tr.t_end * scene.native_fps) + 1
        in_zone_frames: list[float] = []
        for f in range(f0, f1 + 1):
            t = f * dt
            p = tr.foot_at(t)
            if p is not None and point_in_convex_polygon(p, poly):
                in_zone_frames.append(t)
        if not in_zone_frames:
            distractors.append(tr)
            continue
        # straight chords through a convex polygon → one contiguous interval
        start, end = round(in_zone_frames[0], 4), round(in_zone_frames[-1], 4)
        events.append(
            GroundTruthEvent(
                id=f"{tr.id}",
                camera=tr.camera,
                track_id=tr.id,
                start=start,
                end=end,
                duration_s=round(end - start + dt, 4),
                height_px=tr.height_px,
                occlusion=tr.occlusion,
                alpha=tr.alpha,
                speed_px_s=tr.speed_px_s,
            )
        )
    events.sort(key=lambda e: (e.camera, e.start))
    return GroundTruth(events=events, distractor_tracks=distractors)
