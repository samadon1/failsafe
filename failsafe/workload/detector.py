"""Person detectors. `YoloDetector` (ultralytics YOLOv8, real) is the default; `HogDetector`
(OpenCV's pretrained pedestrian HOG) is a no-download fallback. Both return Detections in frame
pixel coordinates regardless of the inference resolution."""

from __future__ import annotations

import time
from typing import Protocol

import cv2
import numpy as np

from failsafe.workload.zone import Detection


class Detector(Protocol):
    name: str

    def detect(self, frame: np.ndarray, imgsz: int, conf: float) -> list[Detection]: ...


class YoloDetector:
    def __init__(self, weights: str = "yolov8n.pt", device: str | None = None):
        from ultralytics import YOLO

        self.model = YOLO(weights)
        self.name = f"ultralytics:{weights}"
        self.device = device or _default_device()
        self.infer_ms: list[float] = []
        # warm-up so the first measured frame isn't paying for graph/kernel init
        dummy = np.zeros((360, 640, 3), dtype=np.uint8)
        for sz in (640, 480, 320):
            self.model.predict(dummy, imgsz=sz, classes=[0], conf=0.25, verbose=False, device=self.device)

    def detect(self, frame: np.ndarray, imgsz: int, conf: float) -> list[Detection]:
        t0 = time.perf_counter()
        res = self.model.predict(
            frame, imgsz=imgsz, classes=[0], conf=conf, verbose=False, device=self.device
        )[0]
        self.infer_ms.append((time.perf_counter() - t0) * 1000.0)
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x0, y0, x1, y1), c in zip(xyxy, confs):
            out.append(Detection(float(x0), float(y0), float(x1), float(y1), float(c)))
        return out


class HogDetector:
    """OpenCV default people detector. Weak, but real and dependency-free."""

    def __init__(self):
        self.hog = cv2.HOGDescriptor()
        self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        self.name = "opencv:hog-people"
        self.infer_ms: list[float] = []

    def detect(self, frame: np.ndarray, imgsz: int, conf: float) -> list[Detection]:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]
        scale = imgsz / max(h, w)
        small = cv2.resize(frame, (int(w * scale), int(h * scale))) if scale < 1 else frame
        rects, weights = self.hog.detectMultiScale(small, winStride=(8, 8), padding=(4, 4), scale=1.05)
        self.infer_ms.append((time.perf_counter() - t0) * 1000.0)
        out: list[Detection] = []
        for (x, y, bw, bh), wgt in zip(rects, weights):
            c = float(min(1.0, wgt / 3.0))
            if c < conf:
                continue
            inv = 1.0 / scale if scale < 1 else 1.0
            out.append(Detection(x * inv, y * inv, (x + bw) * inv, (y + bh) * inv, c))
        return out


def _default_device() -> str:
    try:
        import torch

        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda:0"
    except Exception:  # pragma: no cover
        pass
    return "cpu"


def make_detector(kind: str = "yolo", weights: str = "yolov8n.pt", device: str | None = None) -> Detector:
    if kind == "yolo":
        return YoloDetector(weights, device)
    if kind == "hog":
        return HogDetector()
    raise ValueError(kind)
