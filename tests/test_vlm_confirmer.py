"""VlmConfirmer: parsing, caching, offline safety — no network (the live smoke test is separate)."""

from __future__ import annotations

import numpy as np

from failsafe.workload.cloud import ConfirmationRequest, encode_jpeg
from failsafe.workload.vlm_confirmer import VlmConfirmer, _parse_verdict
from failsafe.workload.zone import Detection

ZONE = [(100.0, 100.0), (500.0, 100.0), (500.0, 400.0), (100.0, 400.0)]


def _req():
    frame = np.full((540, 960, 3), 40, np.uint8)
    return ConfirmationRequest(camera="A", scene_t=1.0, jpeg=encode_jpeg(frame),
                               candidate=Detection(250.0, 150.0, 330.0, 380.0, 0.9), zone=ZONE)


def test_parse_verdict_reads_the_json_line():
    assert _parse_verdict('<think>she is inside</think>\nVERDICT: {"person_in_zone": true, "confidence": 0.9}') == (True, 0.9)
    assert _parse_verdict('...\nVERDICT: {"person_in_zone": false, "confidence": 0.7}') == (False, 0.7)


def test_parse_verdict_falls_back_after_think():
    assert _parse_verdict("<think>weighing it</think> No, the feet are outside.")[0] is False
    assert _parse_verdict("<think>...</think> Yes.")[0] is True


def test_offline_call_is_safe_and_labelled(tmp_path):
    # no api key, no cache hit → returns unconfirmed with a reason, never raises
    c = VlmConfirmer(model="nvidia/x", api_key="", cache_dir=tmp_path)
    resp, nbytes = c.confirm(_req())
    assert resp.confirmed is False and "unavailable" in resp.detail and nbytes > 0


def test_cache_hit_is_served_without_a_client(tmp_path):
    c = VlmConfirmer(model="nvidia/x", api_key="", cache_dir=tmp_path)
    req = _req()
    key = c._key(__import__("failsafe.workload.vlm_confirmer", fromlist=["_overlay"])._overlay(req.jpeg, req.candidate, req.zone))
    (tmp_path / f"{key}.json").write_text('{"confirmed": true, "confidence": 0.88, "detail": "recorded"}')
    resp, _ = c.confirm(req)
    assert resp.confirmed is True and resp.confidence == 0.88 and resp.detail == "recorded"
    assert c._client is None  # a cache hit never constructs a client


def test_matches_the_confirmation_service_protocol():
    from failsafe.workload.cloud import ConfirmationService
    c = VlmConfirmer(api_key="")
    assert hasattr(c, "name") and callable(c.confirm)
    _: ConfirmationService = c  # structural check
