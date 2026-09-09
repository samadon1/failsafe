"""Person cutout library.

Real person imagery is extracted from ultralytics' own sample images (`bus.jpg`, `zidane.jpg`)
using YOLOv8n-seg, saved as RGBA PNGs and cached under datasets/cutouts/. Composited onto synthetic
backgrounds, they are detectable by the real detector while ground truth stays analytic.

Limitation (documented in DECISIONS D-005): a handful of people only; variety comes from flips,
scales, contrast and brightness. Good enough to make FPS/resolution/occlusion matter; not a claim
about real-world diversity.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)

SAMPLE_IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}
DATASETS_DIR = Path("datasets")
SAMPLES_DIR = DATASETS_DIR / "samples"
CUTOUTS_DIR = DATASETS_DIR / "cutouts"
MIN_CUTOUT_HEIGHT = 80
MIN_ASPECT = 2.0  # height / width; full-body standing people
# bus_0 is a thin, dark, leaning back-view figure that YOLOv8n does not reliably recognise once
# composited (conf 0.06–0.36 even at 96–165 px). Excluded so corpus difficulty comes from the
# controlled knobs (scale, contrast, occlusion, duration), not from one ambiguous asset.
EXCLUDE_CUTOUTS = {"bus_0"}


def _download(url: str, dest: Path) -> None:
    import requests

    dest.parent.mkdir(parents=True, exist_ok=True)
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dest.write_bytes(r.content)


def build_cutouts(force: bool = False) -> list[Path]:
    """Extract person cutouts from the sample images. Returns the cutout paths."""
    existing = sorted(CUTOUTS_DIR.glob("*.png"))
    if existing and not force:
        return existing

    from ultralytics import YOLO

    CUTOUTS_DIR.mkdir(parents=True, exist_ok=True)
    for p in CUTOUTS_DIR.glob("*.png"):
        p.unlink()

    model = YOLO("yolov8n-seg.pt")
    out: list[Path] = []
    for name, url in SAMPLE_IMAGES.items():
        img_path = SAMPLES_DIR / name
        if not img_path.exists():
            log.info("downloading %s", url)
            _download(url, img_path)
        img = cv2.imread(str(img_path))
        res = model.predict(img, classes=[0], conf=0.5, verbose=False)[0]
        if res.masks is None:
            continue
        masks = res.masks.data.cpu().numpy()  # (n, h, w) at model input size
        boxes = res.boxes.xyxy.cpu().numpy()
        for k, (mask, box) in enumerate(zip(masks, boxes)):
            x0, y0, x1, y1 = [int(round(v)) for v in box]
            if y1 - y0 < MIN_CUTOUT_HEIGHT:
                continue
            m = cv2.resize(mask, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
            m = (m > 0.5).astype(np.uint8) * 255
            crop = img[y0:y1, x0:x1]
            alpha = m[y0:y1, x0:x1]
            # trim to the mask's tight bbox so the foot point = bottom of the visible person
            ys, xs = np.where(alpha > 0)
            if len(ys) == 0:
                continue
            crop = crop[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            alpha = alpha[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1]
            # keep full-body standing people only (upper-body crops have no foot point)
            if crop.shape[0] / max(1, crop.shape[1]) < MIN_ASPECT:
                continue
            rgba = np.dstack([crop, alpha])
            if f"{img_path.stem}_{k}" in EXCLUDE_CUTOUTS:
                continue
            dest = CUTOUTS_DIR / f"{img_path.stem}_{k}.png"
            cv2.imwrite(str(dest), rgba)
            out.append(dest)
    if not out:
        raise RuntimeError("no person cutouts extracted; check network / ultralytics install")
    return sorted(out)


class CutoutLibrary:
    def __init__(self, paths: list[Path] | None = None):
        paths = paths or build_cutouts()
        self.images: list[np.ndarray] = []
        for p in paths:
            im = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
            if im is None or im.ndim != 3 or im.shape[2] != 4:
                continue
            self.images.append(im)
        if not self.images:
            raise RuntimeError("cutout library is empty")
        self._cache: dict[tuple[int, int, bool], np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.images)

    def get(self, index: int, height_px: int, flip: bool) -> np.ndarray:
        """RGBA cutout scaled to `height_px` (cached)."""
        key = (index % len(self.images), height_px, flip)
        im = self._cache.get(key)
        if im is None:
            src = self.images[key[0]]
            h, w = src.shape[:2]
            new_w = max(1, int(round(w * height_px / h)))
            im = cv2.resize(src, (new_w, height_px), interpolation=cv2.INTER_AREA if height_px < h else cv2.INTER_LINEAR)
            if flip:
                im = im[:, ::-1].copy()
            self._cache[key] = im
        return im
