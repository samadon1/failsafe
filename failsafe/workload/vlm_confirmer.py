"""A real vision-language model as the cloud confirmer (D-049).

The Phase-1 confirmer is a heavier YOLO model standing in for a cloud VLM (D-006). This is the
real thing: an NVIDIA multimodal reasoning model answers "is the person in the marked box actually
inside the outlined zone?" over the frame, with the zone polygon and the candidate box drawn on so
the model has visual grounding.

Endpoint/model-agnostic (OpenAI-compatible `/v1/chat/completions` with `image_url`). It runs on the
hosted `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` today; point `base_url`/`model` at a
self-hosted Cosmos Reason NIM (`nvidia/cosmos3-nano-reasoner` on a GPU node, e.g. Nebius) with no
code change once a GPU host is available (`cosmos-reason2-8b` is hosted but account-gated).

Why it is NOT wired into the timing grid: a remote VLM call is seconds long and non-deterministic,
which would corrupt real-time alert latency (D-005) and reproducibility. So every verdict is cached
by a content hash; a replay reuses the cached verdict while the NetworkInjector still supplies the
(simulated) cloud latency. Use it for the demo and a labelled qualitative set; the deterministic
stand-in remains the default for the measured 200-run grid.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np

from failsafe.corpus.scene import point_in_convex_polygon
from failsafe.workload.cloud import RESPONSE_BYTES, ConfirmationRequest, ConfirmationResponse

DEFAULT_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
CACHE_DIR = Path("artifacts/vlm_cache")

PROMPT = (
    "You are a safety monitor for a restricted zone. The outlined polygon is the zone; the person "
    "under review is in the highlighted box. Decide whether that person's feet are inside the zone. "
    "Reason briefly inside <think></think>, then on the final line write exactly:\n"
    'VERDICT: {"person_in_zone": true|false, "confidence": 0.0-1.0}'
)


def _overlay(jpeg: bytes, candidate, zone: list[tuple[float, float]]) -> bytes:
    """Draw the zone polygon (amber) and the candidate box (green) onto the frame for grounding."""
    frame = cv2.imdecode(np.frombuffer(jpeg, np.uint8), cv2.IMREAD_COLOR)
    if zone:
        pts = np.array([[int(x), int(y)] for x, y in zone], np.int32).reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], True, (10, 150, 230), 3)  # amber (BGR)
    cv2.rectangle(frame, (int(candidate.x0), int(candidate.y0)), (int(candidate.x1), int(candidate.y1)), (0, 200, 0), 3)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


def _parse_verdict(text: str) -> tuple[bool, float]:
    """Pull the JSON verdict from the model's reply; fall back to a yes/no scan."""
    for m in re.finditer(r'\{[^{}]*"person_in_zone"[^{}]*\}', text):
        try:
            obj = json.loads(m.group(0))
            return bool(obj["person_in_zone"]), float(obj.get("confidence", 0.5))
        except Exception:
            continue
    tail = text.lower().rsplit("</think>", 1)[-1]
    if "true" in tail or "yes" in tail:
        return True, 0.5
    return False, 0.5


class VlmConfirmer:
    """Cloud confirmer backed by a real NVIDIA VLM. Non-deterministic calls are cached by content
    hash so experiments and demos replay deterministically."""

    def __init__(
        self,
        model: str | None = None,
        base_url: str | None = None,
        api_key: str | None = None,
        cache_dir: Path | None = CACHE_DIR,
        max_tokens: int = 1024,
        temperature: float = 0.2,
        offline_ok: bool = True,
    ):
        self.model = model or os.environ.get("FAILSAFE_VLM_MODEL") or DEFAULT_MODEL
        self.base_url = base_url or os.environ.get("FAILSAFE_VLM_BASE_URL") or os.environ.get("FAILSAFE_LLM_BASE_URL") or DEFAULT_BASE_URL
        self.api_key = api_key or os.environ.get("FAILSAFE_VLM_API_KEY") or os.environ.get("FAILSAFE_LLM_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.max_tokens, self.temperature, self.offline_ok = max_tokens, temperature, offline_ok
        self.name = f"vlm:{self.model}"
        self._client = None  # lazy: only built when a live call is actually needed
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _key(self, overlay_jpeg: bytes) -> str:
        h = hashlib.sha1()
        h.update(self.model.encode())
        h.update(overlay_jpeg)  # the overlaid frame fully determines the question
        return h.hexdigest()[:16]

    def _client_or_raise(self):
        if self._client is None:
            if not self.api_key:
                raise RuntimeError("no VLM API key (FAILSAFE_VLM_API_KEY / NVIDIA_API_KEY); use the recorded cache or the stand-in confirmer")
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def confirm(self, req: ConfirmationRequest) -> tuple[ConfirmationResponse, int]:
        overlay = _overlay(req.jpeg, req.candidate, req.zone)
        key = self._key(overlay)
        rec = self.cache_dir / f"{key}.json" if self.cache_dir else None

        if rec and rec.exists():
            d = json.loads(rec.read_text())
            return ConfirmationResponse(d["confirmed"], d["confidence"], d.get("detail", "cached")), RESPONSE_BYTES

        b64 = base64.b64encode(overlay).decode()
        t0 = time.perf_counter()
        try:
            r = self._client_or_raise().chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
                max_tokens=self.max_tokens, temperature=self.temperature,
            )
            text = r.choices[0].message.content or ""
            confirmed, confidence = _parse_verdict(text)
            latency_ms = (time.perf_counter() - t0) * 1000.0
            detail = text.rsplit("</think>", 1)[-1].strip()[:200] or "vlm"
            if rec:
                rec.write_text(json.dumps({
                    "model": self.model, "confirmed": confirmed, "confidence": confidence, "detail": detail,
                    "camera": req.camera, "scene_t": round(req.scene_t, 3), "latency_ms": round(latency_ms),
                    "completion_tokens": getattr(r.usage, "completion_tokens", None), "raw_tail": text[-400:],
                }, indent=1))
            return ConfirmationResponse(confirmed, confidence, detail), RESPONSE_BYTES
        except Exception as e:
            # a VLM confirmer must never take down a run; a call failure reads as "unconfirmed"
            # (the local decision still stands via on_cloud_failure), and we say why.
            if not self.offline_ok:
                raise
            return ConfirmationResponse(False, 0.0, f"vlm unavailable: {type(e).__name__}"), RESPONSE_BYTES

    # a ground-truth check for the labelled qualitative comparison (not used in the runtime path)
    def geometric_truth(self, req: ConfirmationRequest) -> bool:
        return point_in_convex_polygon(req.candidate.foot, req.zone)
