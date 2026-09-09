"""Cloud confirmation service.

Normal mode sends the full frame (JPEG) plus the candidate box to a "cloud" service that answers
"is there really a person inside the marked zone?". Phase 1 stand-in: a heavier local model
(YOLOv8m at 640) run on the frame — see DECISIONS D-006. Phase 4 swaps in Nemotron VL via Token
Factory behind the same interface. Both sit behind the NetworkInjector so cloud state, bandwidth
and timeouts apply identically.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from failsafe.corpus.scene import point_in_convex_polygon
from failsafe.workload.zone import Detection

RESPONSE_BYTES = 160  # small JSON payload


@dataclass(frozen=True)
class ConfirmationRequest:
    camera: str
    scene_t: float
    jpeg: bytes
    candidate: Detection
    zone: list[tuple[float, float]]


@dataclass(frozen=True)
class ConfirmationResponse:
    confirmed: bool
    confidence: float
    detail: str = ""


class ConfirmationService(Protocol):
    name: str

    def confirm(self, req: ConfirmationRequest) -> tuple[ConfirmationResponse, int]:
        """Returns (response, response_bytes)."""
        ...


def encode_jpeg(frame: np.ndarray, quality: int = 80) -> bytes:
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not ok:
        raise RuntimeError("jpeg encode failed")
    return buf.tobytes()


class HeavyModelConfirmer:
    """Phase 1 stand-in for a cloud VLM: a heavier detector re-localises the person at full
    resolution and re-checks the foot point against the zone with a tighter box."""

    def __init__(self, weights: str = "yolov8m.pt", device: str | None = None, conf: float = 0.5):
        from ultralytics import YOLO

        from failsafe.workload.detector import _default_device

        self.model = YOLO(weights)
        self.device = device or _default_device()
        self.conf = conf
        self.name = f"stand-in:{weights}"
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        self.model.predict(dummy, imgsz=640, classes=[0], conf=0.25, verbose=False, device=self.device)

    def confirm(self, req: ConfirmationRequest) -> tuple[ConfirmationResponse, int]:
        frame = cv2.imdecode(np.frombuffer(req.jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
        res = self.model.predict(frame, imgsz=640, classes=[0], conf=self.conf, verbose=False, device=self.device)[0]
        best = None
        if res.boxes is not None and len(res.boxes) > 0:
            xyxy = res.boxes.xyxy.cpu().numpy()
            confs = res.boxes.conf.cpu().numpy()
            cx = (req.candidate.x0 + req.candidate.x1) / 2
            for (x0, y0, x1, y1), c in zip(xyxy, confs):
                d = Detection(float(x0), float(y0), float(x1), float(y1), float(c))
                # the heavy model's box that overlaps the candidate horizontally
                if d.x0 - 10 <= cx <= d.x1 + 10:
                    if best is None or d.confidence > best.confidence:
                        best = d
        if best is None:
            return ConfirmationResponse(False, 0.0, "no person re-detected"), RESPONSE_BYTES
        inside = point_in_convex_polygon(best.foot, req.zone)
        return ConfirmationResponse(inside, best.confidence, "foot in zone" if inside else "foot outside zone"), RESPONSE_BYTES


class AlwaysConfirm:
    """Test double: confirms every candidate instantly."""

    name = "mock:always-confirm"

    def confirm(self, req: ConfirmationRequest) -> tuple[ConfirmationResponse, int]:
        return ConfirmationResponse(True, 1.0, "mock"), RESPONSE_BYTES


# ---------------------------------------------------------------------------------------------
# Out-of-process cloud server
# ---------------------------------------------------------------------------------------------


def _server_main(conn, weights: str, device: str | None, conf: float) -> None:  # pragma: no cover
    svc = HeavyModelConfirmer(weights, device, conf)
    conn.send(("ready", svc.name))
    while True:
        msg = conn.recv()
        if msg is None:
            break
        req: ConfirmationRequest = msg
        try:
            resp, nbytes = svc.confirm(req)
            conn.send((resp, nbytes))
        except Exception as e:  # keep serving
            conn.send((ConfirmationResponse(False, 0.0, f"server error: {e}"), RESPONSE_BYTES))
    conn.close()


class CloudServerProcess:
    """Runs the stand-in confirmer in a child process so its compute is *not* charged to the
    edge device's CPU accounting (the ResourceMonitor excludes this PID). Requests are serialised
    through a pipe; the network injector's delays happen on the caller's side, so a slow cloud
    stalls callers, not the server."""

    def __init__(self, weights: str = "yolov8m.pt", device: str | None = None, conf: float = 0.5):
        import multiprocessing as mp
        import threading

        ctx = mp.get_context("spawn")
        self._parent, child = ctx.Pipe()
        self._proc = ctx.Process(target=_server_main, args=(child, weights, device, conf), daemon=True)
        self._proc.start()
        child.close()
        status, self.name = self._parent.recv()
        assert status == "ready"
        self._lock = threading.Lock()
        self.pid = self._proc.pid

    def confirm(self, req: ConfirmationRequest) -> tuple[ConfirmationResponse, int]:
        with self._lock:
            self._parent.send(req)
            return self._parent.recv()

    def close(self) -> None:
        try:
            with self._lock:
                self._parent.send(None)
        except Exception:
            pass
        self._proc.join(timeout=5)
        if self._proc.is_alive():
            self._proc.terminate()
