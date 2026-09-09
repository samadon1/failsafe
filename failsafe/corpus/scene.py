"""Synthetic scene specification and seeded generator.

A Scene is a deterministic function of (tier, seed). It describes 4 static cameras, each with a
restricted-zone polygon, and a list of Tracks: a person cutout moving along a piecewise-linear path.
Ground truth is analytic (foot point inside polygon, sampled on the native 30 FPS grid), so recall
and precision computed against it are exact.

Variation per track (so that FPS / resolution / compute pressure materially affect outcomes):
  * target in-zone duration    0.15 s .. 4 s  (log-uniform)
  * person height (px)         HEIGHT_RANGE_PX (log-uniform)
  * walking speed              derived from chord length / duration, clamped
  * contrast                   alpha scale + brightness shift
  * occlusion                  overhead occluder hiding OCCLUSION_LEVELS of the person from the top
  * path geometry              straight chords through the polygon; near-miss distractors run
                               parallel to an edge a few px outside; wanderers stay away from it.

Nothing here touches the detector. The renderer (render.py) turns a Scene into frames.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

NATIVE_FPS = 30
FRAME_W, FRAME_H = 640, 360
MIN_EVENT_GAP_S = 1.6  # zone clear time between consecutive zone events on one camera (> alert cooldown)

TrackKind = Literal["zone_event", "near_miss", "wanderer"]


class TierSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    duration_s: float
    # expected number of zone-event tracks per camera (Poisson-ish scheduling, not exact)
    zone_events: dict[str, int]
    near_misses: dict[str, int]
    wanderers: dict[str, int]


TIERS: dict[str, TierSpec] = {
    # tiny: for unit tests / smoke runs
    "smoke": TierSpec(
        name="smoke",
        duration_s=30.0,
        zone_events={"A": 4, "B": 1, "C": 1, "D": 0},
        near_misses={"A": 2, "B": 1, "C": 0, "D": 1},
        wanderers={"A": 1, "B": 1, "C": 1, "D": 1},
    ),
    # quick: local iteration and the Phase 1 report (~3 min wall-clock per experiment)
    "quick": TierSpec(
        name="quick",
        duration_s=180.0,
        zone_events={"A": 40, "B": 8, "C": 8, "D": 8},
        near_misses={"A": 20, "B": 8, "C": 8, "D": 8},
        wanderers={"A": 6, "B": 14, "C": 14, "D": 14},
    ),
    # full: Nebius campaigns (one experiment = 15 min wall-clock)
    "full": TierSpec(
        name="full",
        duration_s=900.0,
        zone_events={"A": 200, "B": 40, "C": 40, "D": 40},
        near_misses={"A": 100, "B": 40, "C": 40, "D": 40},
        wanderers={"A": 30, "B": 70, "C": 70, "D": 70},
    ),
}

CAMERA_ROLES: dict[str, str] = {
    "A": "critical loading zone",
    "B": "warehouse aisle",
    "C": "storage area",
    "D": "entrance",
}
CRITICAL_CAMERA = "A"
# lowest priority first (used by drop_background_streams=lowest_priority)
BACKGROUND_PRIORITY = ["D", "C", "B"]


class Occluder(BaseModel):
    model_config = ConfigDict(extra="forbid")

    x0: int
    y0: int
    x1: int
    y1: int
    color: tuple[int, int, int]


class Track(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    camera: str
    kind: TrackKind
    cutout: int  # index into the cutout library (mod len at render time)
    flip: bool
    height_px: float
    t_start: float
    # piecewise-linear foot-point path: waypoints (x, y) reached at times (t_start + cumulative)
    waypoints: list[tuple[float, float]]
    segment_durations: list[float]
    alpha: float = Field(ge=0.3, le=1.0)  # contrast: 1.0 = full, lower = blends into background
    brightness: float = Field(ge=-60, le=60)
    occlusion: float = Field(ge=0.0, le=0.5)  # fraction of height hidden from the bottom
    occluder: Occluder | None = None
    target_duration_s: float | None = None  # zone_event only (what the generator aimed for)
    speed_px_s: float

    @property
    def t_end(self) -> float:
        return self.t_start + sum(self.segment_durations)

    def foot_at(self, t: float) -> tuple[float, float] | None:
        """Foot-point position at scene time t, or None if the track is not active."""
        if t < self.t_start or t > self.t_end:
            return None
        u = t - self.t_start
        for i, d in enumerate(self.segment_durations):
            if u <= d or i == len(self.segment_durations) - 1:
                (x0, y0), (x1, y1) = self.waypoints[i], self.waypoints[i + 1]
                f = 0.0 if d <= 0 else max(0.0, min(1.0, u / d))
                return (x0 + (x1 - x0) * f, y0 + (y1 - y0) * f)
            u -= d
        return None


class CameraSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    role: str
    zone: list[tuple[float, float]]  # convex polygon, foot-point coordinates
    background_seed: int
    is_critical: bool


class Scene(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tier: str
    seed: int
    duration_s: float
    width: int = FRAME_W
    height: int = FRAME_H
    native_fps: int = NATIVE_FPS
    cameras: list[CameraSpec]
    tracks: list[Track]

    @property
    def hash(self) -> str:
        s = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(s.encode()).hexdigest()[:12]

    def camera(self, name: str) -> CameraSpec:
        return next(c for c in self.cameras if c.name == name)

    def tracks_for(self, camera: str) -> list[Track]:
        return [t for t in self.tracks if t.camera == camera]

    @property
    def n_frames(self) -> int:
        return int(round(self.duration_s * self.native_fps))


# ---------------------------------------------------------------------------------------------
# Geometry helpers
# ---------------------------------------------------------------------------------------------


def point_in_convex_polygon(p: tuple[float, float], poly: list[tuple[float, float]]) -> bool:
    """Inclusive containment test for a convex polygon given in consistent winding order."""
    x, y = p
    sign = 0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        cross = (x1 - x0) * (y - y0) - (y1 - y0) * (x - x0)
        if abs(cross) < 1e-9:
            continue
        s = 1 if cross > 0 else -1
        if sign == 0:
            sign = s
        elif s != sign:
            return False
    return True


def _polygon_chord(
    poly: list[tuple[float, float]], entry: tuple[float, float], direction: tuple[float, float]
) -> float:
    """Length of the ray from `entry` (on the boundary) in `direction` until it exits the polygon."""
    lo, hi = 0.0, 4000.0
    dx, dy = direction
    # bisection on the exit distance (convex → single exit)
    def inside(d: float) -> bool:
        return point_in_convex_polygon((entry[0] + dx * d, entry[1] + dy * d), poly)

    if not inside(1.0):
        return 0.0
    for _ in range(40):
        mid = (lo + hi) / 2
        if inside(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _edge_tangent_normal(poly, i):
    (x0, y0), (x1, y1) = poly[i], poly[(i + 1) % len(poly)]
    ex, ey = x1 - x0, y1 - y0
    L = math.hypot(ex, ey)
    tx, ty = ex / L, ey / L
    # inward normal: the one pointing toward the centroid
    cx = sum(p[0] for p in poly) / len(poly)
    cy = sum(p[1] for p in poly) / len(poly)
    nx, ny = -ty, tx
    mx, my = (x0 + x1) / 2, (y0 + y1) / 2
    if (cx - mx) * nx + (cy - my) * ny < 0:
        nx, ny = -nx, -ny
    return (tx, ty), (nx, ny), L


# ---------------------------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------------------------


def _make_cameras(rng: np.random.Generator) -> list[CameraSpec]:
    cams = []
    for i, (name, role) in enumerate(CAMERA_ROLES.items()):
        # a convex quad on the floor with mild perspective, placed differently per camera
        cx = rng.uniform(220, 420)
        cy = rng.uniform(220, 300)
        w_top = rng.uniform(110, 170)
        w_bot = w_top * rng.uniform(1.15, 1.45)
        h = rng.uniform(70, 110)
        zone = [
            (cx - w_top / 2, cy - h / 2),
            (cx + w_top / 2, cy - h / 2),
            (cx + w_bot / 2, cy + h / 2),
            (cx - w_bot / 2, cy + h / 2),
        ]
        cams.append(
            CameraSpec(
                name=name,
                role=role,
                zone=[(round(x, 1), round(y, 1)) for x, y in zone],
                background_seed=int(rng.integers(0, 2**31 - 1)),
                is_critical=(name == CRITICAL_CAMERA),
            )
        )
    return cams


# Corpus difficulty profile (DECISIONS D-016): calibrated so the full-fidelity configuration can
# reach the recall invariant while reduced resolution / FPS materially lose events.
HEIGHT_RANGE_PX = (45.0, 200.0)  # log-uniform
ALPHA_RANGE = (0.7, 1.0)  # contrast (1.0 = opaque)
OCCLUSION_LEVELS = (0.0, 0.0, 0.0, 0.15, 0.25)  # fraction of the person hidden from the top
# in-zone duration ranges (s, log-uniform). The critical camera gets the fast, hard events; the
# background cameras cover non-critical areas where people linger — that is what makes reducing
# background FPS a *cheap* degradation rather than a free one.
DURATION_RANGE_S = {"critical": (0.15, 4.0), "background": (0.8, 4.0)}
CUTOUT_INDEX_SPACE = 8  # scenes index cutouts modulo the library size, so hashes don't depend on it


def _sample_height(rng: np.random.Generator) -> float:
    lo, hi = HEIGHT_RANGE_PX
    return float(math.exp(rng.uniform(math.log(lo), math.log(hi))))


def _clip_point(p, w=FRAME_W, h=FRAME_H, margin=4):
    return (min(max(p[0], margin), w - margin), min(max(p[1], margin), h - margin))


def _zone_event_track(rng, cam: CameraSpec, tid: str, t_start: float, cutout_lib_size: int) -> Track:
    poly = cam.zone
    lo, hi = DURATION_RANGE_S["critical" if cam.is_critical else "background"]
    target = float(math.exp(rng.uniform(math.log(lo), math.log(hi))))
    speed_pref = float(rng.uniform(40, 200))
    want_len = speed_pref * target

    best = None
    for _ in range(60):
        i = int(rng.integers(0, len(poly)))
        (tx, ty), (nx, ny), L = _edge_tangent_normal(poly, i)
        f = float(rng.uniform(0.05, 0.95))
        entry = (poly[i][0] + tx * L * f, poly[i][1] + ty * L * f)
        phi = float(rng.uniform(math.radians(8), math.radians(172)))
        # direction = cos(phi)*tangent + sin(phi)*inward normal
        d = (math.cos(phi) * tx + math.sin(phi) * nx, math.cos(phi) * ty + math.sin(phi) * ny)
        chord = _polygon_chord(poly, entry, d)
        if chord < 2:
            continue
        err = abs(chord - want_len)
        if best is None or err < best[0]:
            best = (err, entry, d, chord)
    assert best is not None
    _, entry, d, chord = best
    speed = max(25.0, min(300.0, chord / target))
    in_zone_s = chord / speed

    lead_in = float(rng.uniform(0.6, 1.6))
    lead_out = float(rng.uniform(0.4, 1.2))
    p0 = _clip_point((entry[0] - d[0] * speed * lead_in, entry[1] - d[1] * speed * lead_in))
    p1 = (entry[0] - d[0] * 0.5, entry[1] - d[1] * 0.5)  # just outside
    exit_pt = (entry[0] + d[0] * chord, entry[1] + d[1] * chord)
    p2 = (exit_pt[0] + d[0] * 0.5, exit_pt[1] + d[1] * 0.5)  # just outside
    p3 = _clip_point((exit_pt[0] + d[0] * speed * lead_out, exit_pt[1] + d[1] * speed * lead_out))

    def seg_t(a, b):
        return math.hypot(b[0] - a[0], b[1] - a[1]) / speed

    waypoints = [p0, p1, p2, p3]
    durs = [seg_t(p0, p1), seg_t(p1, p2), seg_t(p2, p3)]

    height = _sample_height(rng)
    occl = float(rng.choice(OCCLUSION_LEVELS))
    occluder = None
    if occl > 0:
        # occluder box covering the bottom `occl` of the person around the zone chord midpoint
        mx, my = (entry[0] + exit_pt[0]) / 2, (entry[1] + exit_pt[1]) / 2
        half_w = max(24.0, chord / 2 + height * 0.15)
        # overhead occluder (hanging shelving/sign): hides the top `occl` of the person, feet stay
        # visible so the foot-point rule remains satisfiable — occlusion makes detection harder,
        # not impossible
        occluder = Occluder(
            x0=int(max(0, mx - half_w)),
            y0=int(max(0, my - height - 4)),
            x1=int(min(FRAME_W, mx + half_w)),
            y1=int(my - height * (1 - occl)),
            color=(int(rng.integers(60, 120)),) * 3,
        )
    return Track(
        id=tid,
        camera=cam.name,
        kind="zone_event",
        cutout=int(rng.integers(0, max(1, cutout_lib_size))),
        flip=bool(rng.integers(0, 2)),
        height_px=round(height, 1),
        t_start=round(t_start, 3),
        waypoints=[(round(x, 1), round(y, 1)) for x, y in waypoints],
        segment_durations=[round(x, 3) for x in durs],
        alpha=round(float(rng.uniform(*ALPHA_RANGE)), 2),
        brightness=round(float(rng.uniform(-45, 35)), 1),
        occlusion=occl,
        occluder=occluder,
        target_duration_s=round(target, 3),
        speed_px_s=round(speed, 1),
    )


def _near_miss_track(rng, cam: CameraSpec, tid: str, t_start: float, cutout_lib_size: int) -> Track:
    poly = cam.zone
    i = int(rng.integers(0, len(poly)))
    (tx, ty), (nx, ny), L = _edge_tangent_normal(poly, i)
    margin = float(rng.uniform(2.0, 18.0))  # px outside the edge
    # walk along the edge, outside it, over a random portion (and a bit beyond)
    a = float(rng.uniform(-0.3, 0.3))
    b = float(rng.uniform(0.7, 1.3))
    if rng.integers(0, 2):
        a, b = b, a
    x0, y0 = poly[i]
    p0 = _clip_point((x0 + tx * L * a - nx * margin, y0 + ty * L * a - ny * margin))
    p1 = _clip_point((x0 + tx * L * b - nx * margin, y0 + ty * L * b - ny * margin))
    speed = float(rng.uniform(30, 150))
    dur = max(0.5, math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / speed)
    height = _sample_height(rng)
    return Track(
        id=tid,
        camera=cam.name,
        kind="near_miss",
        cutout=int(rng.integers(0, max(1, cutout_lib_size))),
        flip=bool(rng.integers(0, 2)),
        height_px=round(height, 1),
        t_start=round(t_start, 3),
        waypoints=[(round(p0[0], 1), round(p0[1], 1)), (round(p1[0], 1), round(p1[1], 1))],
        segment_durations=[round(dur, 3)],
        alpha=round(float(rng.uniform(*ALPHA_RANGE)), 2),
        brightness=round(float(rng.uniform(-40, 30)), 1),
        occlusion=0.0,
        speed_px_s=round(speed, 1),
    )


def _wanderer_track(rng, cam: CameraSpec, tid: str, t_start: float, cutout_lib_size: int) -> Track:
    poly = cam.zone
    def far(p):
        return all(math.hypot(p[0] - v[0], p[1] - v[1]) > 40 for v in poly) and not point_in_convex_polygon(p, poly)

    def crosses(a, b):
        return any(
            point_in_convex_polygon((a[0] + (b[0] - a[0]) * k / 60, a[1] + (b[1] - a[1]) * k / 60), poly)
            for k in range(61)
        )

    p0 = p1 = None
    for _ in range(400):
        a = (float(rng.uniform(10, FRAME_W - 10)), float(rng.uniform(120, FRAME_H - 10)))
        b = (float(rng.uniform(10, FRAME_W - 10)), float(rng.uniform(120, FRAME_H - 10)))
        if far(a) and far(b) and not crosses(a, b):
            p0, p1 = a, b
            break
    if p0 is None:
        p0, p1 = (20.0, 340.0), (60.0, 330.0)
    speed = float(rng.uniform(20, 120))
    dur = max(0.8, math.hypot(p1[0] - p0[0], p1[1] - p0[1]) / speed)
    return Track(
        id=tid,
        camera=cam.name,
        kind="wanderer",
        cutout=int(rng.integers(0, max(1, cutout_lib_size))),
        flip=bool(rng.integers(0, 2)),
        height_px=round(_sample_height(rng), 1),
        t_start=round(t_start, 3),
        waypoints=[(round(p0[0], 1), round(p0[1], 1)), (round(p1[0], 1), round(p1[1], 1))],
        segment_durations=[round(dur, 3)],
        alpha=round(float(rng.uniform(*ALPHA_RANGE)), 2),
        brightness=round(float(rng.uniform(-40, 30)), 1),
        occlusion=0.0,
        speed_px_s=round(speed, 1),
    )


def generate_scene(tier: str = "quick", seed: int = 1, cutout_lib_size: int = CUTOUT_INDEX_SPACE) -> Scene:
    if tier not in TIERS:
        raise ValueError(f"unknown tier {tier!r}; choose from {sorted(TIERS)}")
    spec = TIERS[tier]
    rng = np.random.default_rng(seed)
    cameras = _make_cameras(rng)
    tracks: list[Track] = []

    for cam in cameras:
        # zone events are scheduled sequentially with gaps so at most one is in the zone at a time
        n_events = spec.zone_events.get(cam.name, 0)
        crng = np.random.default_rng(seed * 1000 + ord(cam.name))
        t = float(crng.uniform(1.0, 4.0))
        k = 0
        while k < n_events and t < spec.duration_s - 6:
            tr = _zone_event_track(crng, cam, f"{cam.name}-ev{k:03d}", t, cutout_lib_size)
            tracks.append(tr)
            # spacing: always leave the zone clear for >= MIN_EVENT_GAP_S (so consecutive intrusions
            # are unambiguous events), then scale the gap by the requested density
            mean_gap = max(MIN_EVENT_GAP_S, (spec.duration_s / max(1, n_events)) - (tr.t_end - tr.t_start))
            t = tr.t_end + float(crng.uniform(MIN_EVENT_GAP_S, max(MIN_EVENT_GAP_S + 0.1, 1.5 * mean_gap)))
            k += 1
        # distractors: independent schedules (may overlap each other and zone events)
        for kind, n, fn in (
            ("near_miss", spec.near_misses.get(cam.name, 0), _near_miss_track),
            ("wanderer", spec.wanderers.get(cam.name, 0), _wanderer_track),
        ):
            for j in range(n):
                ts = float(crng.uniform(0.5, spec.duration_s - 3))
                tracks.append(fn(crng, cam, f"{cam.name}-{kind}{j:03d}", ts, cutout_lib_size))

    tracks.sort(key=lambda tr: (tr.t_start, tr.id))
    return Scene(tier=tier, seed=seed, duration_s=spec.duration_s, cameras=cameras, tracks=tracks)
