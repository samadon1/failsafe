"""Offline precision probe: do distractors produce false in-zone candidates, and does the heavier
confirmer reject them?

For near-miss tracks (a person walking just outside the zone) we sample frames where *no*
ground-truth person is in the zone, run the local detector at several confidence thresholds, and
count in-zone detections (false candidates). Each false candidate is then handed to the confirmer
(the Phase 1 stand-in for the cloud VLM) to see whether it would have been rejected. This maps the
precision landscape: local_confidence_threshold × cloud_confirmation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from failsafe.corpus.scene import point_in_convex_polygon
from failsafe.workload.cloud import ConfirmationRequest, ConfirmationService, encode_jpeg
from failsafe.workload.detector import Detector
from failsafe.workload.source import SyntheticSceneSource
from failsafe.workload.zone import detections_in_zone


@dataclass
class ProbeRow:
    threshold: float
    frames: int = 0
    false_candidates: int = 0
    rejected_by_confirmer: int = 0
    by_margin: dict[str, list[int]] = field(default_factory=lambda: {"<6px": [0, 0], "6-12px": [0, 0], ">12px": [0, 0]})


def _margin_bucket(track, zone) -> str:
    # distance from the track's path to the nearest zone edge ≈ the generator's margin
    (x0, y0), (x1, y1) = track.waypoints[0], track.waypoints[-1]
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    best = 1e9
    n = len(zone)
    for i in range(n):
        ax, ay = zone[i]
        bx, by = zone[(i + 1) % n]
        dx, dy = bx - ax, by - ay
        t = max(0.0, min(1.0, ((mx - ax) * dx + (my - ay) * dy) / (dx * dx + dy * dy)))
        px, py = ax + t * dx, ay + t * dy
        best = min(best, ((mx - px) ** 2 + (my - py) ** 2) ** 0.5)
    return "<6px" if best < 6 else ("6-12px" if best < 12 else ">12px")


def probe(
    source: SyntheticSceneSource,
    detector: Detector,
    confirmer: ConfirmationService | None,
    thresholds: tuple[float, ...] = (0.15, 0.25, 0.4),
    resolution: int = 640,
    frames_per_track: int = 5,
) -> list[ProbeRow]:
    rows = {t: ProbeRow(t) for t in thresholds}
    gt = source.ground_truth
    for tr in gt.distractor_tracks:
        if tr.kind != "near_miss":
            continue
        zone = source.zone(tr.camera)
        bucket = _margin_bucket(tr, zone)
        for t in np.linspace(tr.t_start + 0.1, tr.t_end - 0.1, frames_per_track):
            t = float(t)
            # skip frames where a real intrusion is happening on this camera
            if any(e.camera == tr.camera and e.start - 0.5 <= t <= e.end + 0.5 for e in gt.events):
                continue
            # skip if any *other* active track has its true foot inside the zone
            skip = False
            for other in source.renderer.active_tracks(tr.camera, t):
                f = other.foot_at(t)
                if f is not None and point_in_convex_polygon(f, zone):
                    skip = True
            if skip:
                continue
            frame = source.frame(tr.camera, t)
            for thr in thresholds:
                row = rows[thr]
                row.frames += 1
                row.by_margin[bucket][1] += 1
                cands = detections_in_zone(detector.detect(frame, resolution, thr), zone)
                if not cands:
                    continue
                row.false_candidates += 1
                row.by_margin[bucket][0] += 1
                if confirmer is not None:
                    best = max(cands, key=lambda d: d.confidence)
                    resp, _ = confirmer.confirm(
                        ConfirmationRequest(tr.camera, t, encode_jpeg(frame), best, zone)
                    )
                    if not resp.confirmed:
                        row.rejected_by_confirmer += 1
    return [rows[t] for t in thresholds]


def hallucination_probe(source: SyntheticSceneSource, detector: Detector, thresholds=(0.15, 0.25, 0.4), n: int = 40, resolution: int = 640) -> dict[float, tuple[int, int]]:
    """In-zone detections on frames with NO person present at all."""
    out = {t: [0, 0] for t in thresholds}
    for cam in source.cameras:
        zone = source.zone(cam)
        ts = np.linspace(0.5, source.duration_s - 0.5, n)
        for t in ts:
            t = float(t)
            if source.renderer.active_tracks(cam, t):
                continue
            frame = source.frame(cam, t)
            for thr in thresholds:
                out[thr][1] += 1
                if detections_in_zone(detector.detect(frame, resolution, thr), zone):
                    out[thr][0] += 1
    return {k: (v[0], v[1]) for k, v in out.items()}
