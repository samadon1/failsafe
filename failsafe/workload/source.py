"""Frame sources. A FrameSource yields frames for (camera, scene_t) on a native FPS grid and
carries ground truth. `SyntheticSceneSource` renders on the fly from a Scene (deterministic);
`FileVideoSource` reads a real holdout clip with a labels file."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from failsafe.corpus.assets import CutoutLibrary
from failsafe.corpus.ground_truth import GroundTruth, GroundTruthEvent, compute_ground_truth
from failsafe.corpus.render import SceneRenderer
from failsafe.corpus.scene import CRITICAL_CAMERA, Scene, generate_scene


class FrameSource(Protocol):
    cameras: list[str]
    critical_camera: str
    native_fps: int
    duration_s: float
    ground_truth: GroundTruth
    corpus_hash: str

    def zone(self, camera: str) -> list[tuple[float, float]]: ...
    def frame(self, camera: str, scene_t: float) -> np.ndarray: ...


class SyntheticSceneSource:
    def __init__(self, scene: Scene, cutouts: CutoutLibrary):
        self.scene = scene
        self.renderer = SceneRenderer(scene, cutouts)
        self.cameras = [c.name for c in scene.cameras]
        self.critical_camera = CRITICAL_CAMERA
        self.native_fps = scene.native_fps
        self.duration_s = scene.duration_s
        self.ground_truth = compute_ground_truth(scene)
        self.corpus_hash = scene.hash

    @classmethod
    def from_tier(cls, tier: str, seed: int, cutouts: CutoutLibrary | None = None) -> SyntheticSceneSource:
        cutouts = cutouts or CutoutLibrary()
        return cls(generate_scene(tier, seed), cutouts)

    def zone(self, camera: str) -> list[tuple[float, float]]:
        return self.scene.camera(camera).zone

    def frame(self, camera: str, scene_t: float) -> np.ndarray:
        return self.renderer.render(camera, scene_t)


class FileVideoSource:
    """A single real clip + `<clip>.labels.json` (zone polygon + event intervals). Frames are
    decoded sequentially and cached ahead so real-time replay is not stalled by decoding."""

    def __init__(self, clip: Path, camera: str = "REAL"):
        self.clip = Path(clip)
        labels = json.loads(self.clip.with_suffix(".labels.json").read_text())
        self._zone = [tuple(p) for p in labels["zone"]]
        self.cameras = [camera]
        self.critical_camera = camera
        cap = cv2.VideoCapture(str(self.clip))
        self.native_fps = int(round(cap.get(cv2.CAP_PROP_FPS))) or 30
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.duration_s = n / self.native_fps
        self._frames: list[np.ndarray] = []
        ok, fr = cap.read()
        while ok:
            self._frames.append(fr)
            ok, fr = cap.read()
        cap.release()
        events = [
            GroundTruthEvent(
                id=f"{camera}-ev{i:03d}",
                camera=camera,
                track_id=f"real-{i}",
                start=float(e["start"]),
                end=float(e["end"]),
                duration_s=float(e["end"]) - float(e["start"]),
                height_px=float(e.get("height_px", 0.0)),
                occlusion=0.0,
                alpha=1.0,
                speed_px_s=0.0,
            )
            for i, e in enumerate(labels["events"])
        ]
        self.ground_truth = GroundTruth(events=events)
        self.corpus_hash = f"real:{self.clip.name}:{n}"

    def zone(self, camera: str) -> list[tuple[float, float]]:
        return self._zone

    def frame(self, camera: str, scene_t: float) -> np.ndarray:
        i = min(len(self._frames) - 1, max(0, int(round(scene_t * self.native_fps))))
        return self._frames[i]
