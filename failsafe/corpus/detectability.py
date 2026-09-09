"""Offline detectability analysis (not real-time; recall only, no latency).

For every ground-truth event, sample a few in-zone frames and ask the local detector at each
resolution whether it sees a person whose foot point is inside the zone. This maps the
*detector-side* landscape (person height, contrast, occlusion × resolution) independently of
sampling/latency effects, and tells us whether the corpus is tuned so that the best configuration
can reach the recall invariant at all.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from failsafe.corpus.ground_truth import GroundTruthEvent
from failsafe.workload.detector import Detector
from failsafe.workload.source import SyntheticSceneSource
from failsafe.workload.zone import detections_in_zone


@dataclass
class EventDetectability:
    event: GroundTruthEvent
    frames_sampled: int
    hits: dict[int, int]  # resolution → frames where a person was detected with foot in zone
    near_miss_fp: dict[int, int] | None = None


def analyse(
    source: SyntheticSceneSource,
    detector: Detector,
    resolutions: tuple[int, ...] = (640, 480, 320),
    conf: float = 0.4,
    frames_per_event: int = 3,
) -> list[EventDetectability]:
    out: list[EventDetectability] = []
    for e in source.ground_truth.events:
        ts = np.linspace(e.start, e.end, frames_per_event) if e.end > e.start else [e.start]
        hits = {r: 0 for r in resolutions}
        zone = source.zone(e.camera)
        for t in ts:
            frame = source.frame(e.camera, float(t))
            for r in resolutions:
                dets = detector.detect(frame, r, conf)
                if detections_in_zone(dets, zone):
                    hits[r] += 1
        out.append(EventDetectability(e, len(ts), hits))
    return out


def near_miss_false_positives(
    source: SyntheticSceneSource,
    detector: Detector,
    resolutions: tuple[int, ...] = (640, 480, 320),
    conf: float = 0.4,
    frames_per_track: int = 3,
) -> dict[int, tuple[int, int]]:
    """resolution → (near-miss frames that produced an in-zone detection, frames sampled)."""
    counts = {r: 0 for r in resolutions}
    n = 0
    for tr in source.ground_truth.distractor_tracks:
        if tr.kind != "near_miss":
            continue
        for t in np.linspace(tr.t_start, tr.t_end, frames_per_track):
            frame = source.frame(tr.camera, float(t))
            zone = source.zone(tr.camera)
            n += 1
            for r in resolutions:
                if detections_in_zone(detector.detect(frame, r, conf), zone):
                    counts[r] += 1
    return {r: (c, n) for r, c in counts.items()}


def summarise(rows: list[EventDetectability], resolutions=(640, 480, 320)) -> dict:
    def rate(sel):
        sel = list(sel)
        if not sel:
            return {r: None for r in resolutions}
        return {r: round(sum(1 for x in sel if x.hits[r] > 0) / len(sel), 3) for r in resolutions}

    return {
        "events": len(rows),
        "event_detected_any_frame": rate(rows),
        "by_height": {
            "<40px": rate(x for x in rows if x.event.height_px < 40),
            "40-80px": rate(x for x in rows if 40 <= x.event.height_px < 80),
            "80-130px": rate(x for x in rows if 80 <= x.event.height_px < 130),
            ">=130px": rate(x for x in rows if x.event.height_px >= 130),
        },
        "by_occlusion": {
            "none": rate(x for x in rows if x.event.occlusion == 0),
            "occluded": rate(x for x in rows if x.event.occlusion > 0),
        },
        "by_contrast": {
            "alpha<0.7": rate(x for x in rows if x.event.alpha < 0.7),
            "alpha>=0.7": rate(x for x in rows if x.event.alpha >= 0.7),
        },
    }
