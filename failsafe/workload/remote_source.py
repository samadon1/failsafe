"""Run a synthetic FrameSource in a child process ("the cameras").

Rendering frames is simulation overhead, not edge work: it must neither be charged to the edge
device's CPU accounting nor contend with the detector for the GIL. The child renders and streams
frames over a pipe; the parent computes ground truth itself (the scene is deterministic).

Two access modes:
  * request/response  — `frame(cam, t)` for ad-hoc access (calibration, tests);
  * stream            — `start_stream(schedule)` makes the child render the schedule in order and
                        push frames ahead of consumption (pipe back-pressure bounds the prefetch).
"""

from __future__ import annotations

import multiprocessing as mp
import pickle

import numpy as np

from failsafe.corpus.ground_truth import compute_ground_truth
from failsafe.corpus.scene import CRITICAL_CAMERA, generate_scene


def _camera_main(conn, tier: str, seed: int) -> None:  # pragma: no cover - child process
    from failsafe.corpus.assets import CutoutLibrary
    from failsafe.workload.source import SyntheticSceneSource

    src = SyntheticSceneSource.from_tier(tier, seed, CutoutLibrary())
    conn.send(("ready", src.scene.width, src.scene.height))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        kind = msg[0]
        if kind == "get":
            _, cam, t = msg
            conn.send_bytes(src.frame(cam, float(t)).tobytes())
        elif kind == "stream":
            _, schedule = msg
            for t, cam in schedule:
                conn.send_bytes(src.frame(cam, float(t)).tobytes())
    conn.close()


class RemoteSyntheticSource:
    def __init__(self, tier: str, seed: int):
        self.scene = generate_scene(tier, seed)
        self.cameras = [c.name for c in self.scene.cameras]
        self.critical_camera = CRITICAL_CAMERA
        self.native_fps = self.scene.native_fps
        self.duration_s = self.scene.duration_s
        self.ground_truth = compute_ground_truth(self.scene)
        self.corpus_hash = self.scene.hash

        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_camera_main, args=(child, tier, seed), daemon=True)
        self._proc.start()
        child.close()
        status, self._w, self._h = self._conn.recv()
        assert status == "ready"
        self.pid = self._proc.pid
        self._stream: list[tuple[float, str]] | None = None
        self._stream_pos = 0

    # -- FrameSource -----------------------------------------------------------------------------

    def zone(self, camera: str) -> list[tuple[float, float]]:
        return self.scene.camera(camera).zone

    def _decode(self, b: bytes) -> np.ndarray:
        return np.frombuffer(b, dtype=np.uint8).reshape(self._h, self._w, 3).copy()

    def frame(self, camera: str, scene_t: float) -> np.ndarray:
        if self._stream is not None and self._stream_pos < len(self._stream):
            t, cam = self._stream[self._stream_pos]
            if cam == camera and abs(t - scene_t) < 1e-9:
                self._stream_pos += 1
                return self._decode(self._conn.recv_bytes())
            raise RuntimeError("out-of-order frame request while streaming")
        self._conn.send(("get", camera, float(scene_t)))
        return self._decode(self._conn.recv_bytes())

    def start_stream(self, schedule: list[tuple[float, str]]) -> None:
        self._stream = list(schedule)
        self._stream_pos = 0
        self._conn.send(("stream", self._stream))

    def close(self) -> None:
        try:
            # drain anything the child still pushed
            while self._stream is not None and self._stream_pos < len(self._stream):
                self._conn.recv_bytes()
                self._stream_pos += 1
            self._conn.send(None)
        except (OSError, EOFError, pickle.PicklingError):
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
