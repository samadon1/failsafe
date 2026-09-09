"""Export a synthetic camera stream to MP4 (demo/inspection), and a labelling helper for real
holdout clips (zone polygon + event intervals → <clip>.labels.json)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2

from failsafe.corpus.render import SceneRenderer
from failsafe.corpus.scene import Scene


def export_camera_mp4(renderer: SceneRenderer, camera: str, dest: Path, t0: float = 0.0, t1: float | None = None) -> Path:
    scene: Scene = renderer.scene
    t1 = scene.duration_s if t1 is None else min(t1, scene.duration_s)
    dest.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    vw = cv2.VideoWriter(str(dest), fourcc, scene.native_fps, (scene.width, scene.height))
    dt = 1.0 / scene.native_fps
    f = int(t0 * scene.native_fps)
    while f * dt <= t1:
        vw.write(renderer.render(camera, f * dt))
        f += 1
    vw.release()
    return dest


def write_labels(
    clip: Path,
    zone: list[tuple[float, float]],
    events: list[tuple[float, float]],
    camera: str = "REAL",
) -> Path:
    """Write ground-truth labels for a real clip. `events` are (start_s, end_s) foot-in-zone intervals."""
    dest = clip.with_suffix(".labels.json")
    dest.write_text(
        json.dumps(
            {"clip": clip.name, "camera": camera, "zone": zone, "events": [{"start": s, "end": e} for s, e in events]},
            indent=2,
        )
    )
    return dest
