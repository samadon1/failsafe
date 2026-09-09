"""Historical indexing: a real background cost.

For every processed frame we compute a perceptual hash (32×32 grey DCT) and a colour histogram and
insert them into a SQLite table. This is what a "search your footage later" feature costs; when
`historical_indexing` is disabled the saving is measurable rather than notional.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import cv2
import numpy as np


class FrameIndexer:
    def __init__(self, db_path: Path):
        db_path.parent.mkdir(parents=True, exist_ok=True)
        if db_path.exists():
            db_path.unlink()
        self._conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self._conn.execute(
            "CREATE TABLE frames (camera TEXT, scene_t REAL, phash BLOB, hist BLOB)"
        )
        self._lock = threading.Lock()
        self.count = 0

    def index(self, camera: str, scene_t: float, frame: np.ndarray) -> None:
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(grey, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
        dct = cv2.dct(small)[:8, :8]
        phash = (dct > np.median(dct)).astype(np.uint8).tobytes()
        hist = cv2.calcHist([frame], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3).flatten()
        hist_b = (hist / max(1.0, hist.sum())).astype(np.float32).tobytes()
        with self._lock:
            self._conn.execute("INSERT INTO frames VALUES (?, ?, ?, ?)", (camera, scene_t, phash, hist_b))
            self.count += 1
            if self.count % 50 == 0:
                self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()
