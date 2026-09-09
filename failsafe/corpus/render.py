"""Deterministic frame renderer: Scene + CutoutLibrary → BGR frame for (camera, t).

Backgrounds are procedurally drawn once per camera from `background_seed` (floor gradient,
shelving, painted zone outline, mild texture). Persons are alpha-composited at their foot point
with the track's contrast/brightness, then occluders are drawn on top.
"""

from __future__ import annotations

import cv2
import numpy as np

from failsafe.corpus.assets import CutoutLibrary
from failsafe.corpus.scene import Scene, Track


def draw_background(seed: int, width: int, height: int, zone: list[tuple[float, float]]) -> np.ndarray:
    rng = np.random.default_rng(seed)
    img = np.zeros((height, width, 3), dtype=np.uint8)
    horizon = int(height * rng.uniform(0.28, 0.36))
    wall = np.array([rng.integers(95, 135), rng.integers(100, 140), rng.integers(105, 150)], dtype=np.float32)
    floor_top = np.array([rng.integers(70, 100)] * 3, dtype=np.float32) + rng.uniform(-8, 8, 3)
    floor_bot = floor_top * rng.uniform(1.25, 1.55)
    img[:horizon] = wall
    for y in range(horizon, height):
        f = (y - horizon) / max(1, height - horizon)
        img[y] = np.clip(floor_top * (1 - f) + floor_bot * f, 0, 255)
    # shelving / racks on the wall band
    x = 0
    while x < width:
        w = int(rng.integers(50, 130))
        if rng.uniform() < 0.7:
            c = tuple(int(v) for v in rng.integers(40, 90, 3))
            cv2.rectangle(img, (x + 4, int(horizon * 0.25)), (x + w - 4, horizon + 6), c, -1)
            for yy in range(int(horizon * 0.35), horizon, max(12, int(rng.integers(14, 26)))):
                cv2.line(img, (x + 6, yy), (x + w - 6, yy), (c[0] + 30, c[1] + 30, c[2] + 30), 2)
        x += w
    # pallets / boxes on the floor away from the zone
    for _ in range(int(rng.integers(3, 7))):
        bx = int(rng.integers(0, width - 60))
        by = int(rng.integers(horizon + 10, height - 40))
        bw, bh = int(rng.integers(30, 70)), int(rng.integers(18, 40))
        c = tuple(int(v) for v in rng.integers(90, 170, 3))
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), c, -1)
        cv2.rectangle(img, (bx, by), (bx + bw, by + bh), tuple(max(0, v - 40) for v in c), 1)
    # texture
    noise = rng.normal(0, 4, (height, width, 1)).astype(np.float32)
    img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    # painted restricted-zone outline (hatched yellow)
    pts = np.array(zone, dtype=np.int32).reshape(-1, 1, 2)
    overlay = img.copy()
    cv2.fillPoly(overlay, [pts], (40, 190, 230))
    img = cv2.addWeighted(overlay, 0.18, img, 0.82, 0)
    cv2.polylines(img, [pts], True, (30, 200, 240), 2)
    return img


class SceneRenderer:
    def __init__(self, scene: Scene, cutouts: CutoutLibrary):
        self.scene = scene
        self.cutouts = cutouts
        self.backgrounds: dict[str, np.ndarray] = {
            c.name: draw_background(c.background_seed, scene.width, scene.height, c.zone) for c in scene.cameras
        }
        self._tracks_by_cam: dict[str, list[Track]] = {c.name: scene.tracks_for(c.name) for c in scene.cameras}

    def active_tracks(self, camera: str, t: float) -> list[Track]:
        return [tr for tr in self._tracks_by_cam[camera] if tr.t_start <= t <= tr.t_end]

    def render(self, camera: str, t: float) -> np.ndarray:
        frame = self.backgrounds[camera].copy()
        active = self.active_tracks(camera, t)
        # draw farther (higher on screen = smaller y) first so nearer persons overlap correctly
        active.sort(key=lambda tr: (tr.foot_at(t) or (0, 0))[1])
        for tr in active:
            foot = tr.foot_at(t)
            if foot is None:
                continue
            self._composite(frame, tr, foot)
        for tr in active:
            if tr.occluder is not None:
                o = tr.occluder
                cv2.rectangle(frame, (o.x0, o.y0), (o.x1, o.y1), o.color, -1)
                cv2.rectangle(frame, (o.x0, o.y0), (o.x1, o.y1), tuple(v + 25 for v in o.color), 1)
        return frame

    def _composite(self, frame: np.ndarray, tr: Track, foot: tuple[float, float]) -> None:
        h_px = max(8, int(round(tr.height_px)))
        cut = self.cutouts.get(tr.cutout, h_px, tr.flip)
        ch, cw = cut.shape[:2]
        x0 = int(round(foot[0] - cw / 2))
        y0 = int(round(foot[1] - ch))  # foot point is the bottom-centre of the cutout
        H, W = frame.shape[:2]
        fx0, fy0 = max(0, x0), max(0, y0)
        fx1, fy1 = min(W, x0 + cw), min(H, y0 + ch)
        if fx1 <= fx0 or fy1 <= fy0:
            return
        sub = cut[fy0 - y0 : fy1 - y0, fx0 - x0 : fx1 - x0]
        rgb = sub[:, :, :3].astype(np.float32) + tr.brightness
        rgb = np.clip(rgb, 0, 255)
        a = (sub[:, :, 3:4].astype(np.float32) / 255.0) * tr.alpha
        dst = frame[fy0:fy1, fx0:fx1].astype(np.float32)
        frame[fy0:fy1, fx0:fx1] = np.clip(rgb * a + dst * (1 - a), 0, 255).astype(np.uint8)
