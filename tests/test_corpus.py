import numpy as np
import pytest

from failsafe.corpus.ground_truth import GRACE_S, compute_ground_truth
from failsafe.corpus.scene import (
    FRAME_H,
    FRAME_W,
    TIERS,
    generate_scene,
    point_in_convex_polygon,
)


def test_point_in_convex_polygon():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_convex_polygon((5, 5), sq)
    assert point_in_convex_polygon((0, 5), sq)  # boundary inclusive
    assert not point_in_convex_polygon((11, 5), sq)
    assert not point_in_convex_polygon((5, -0.1), sq)


def test_scene_is_deterministic_from_seed():
    a = generate_scene("smoke", 3)
    b = generate_scene("smoke", 3)
    c = generate_scene("smoke", 4)
    assert a.hash == b.hash
    assert a.hash != c.hash
    assert a.model_dump() == b.model_dump()


def test_unknown_tier():
    with pytest.raises(ValueError):
        generate_scene("nope", 1)


@pytest.mark.parametrize("tier", ["smoke", "quick"])
def test_ground_truth_events_are_contiguous_and_inside(tier):
    scene = generate_scene(tier, 1)
    gt = compute_ground_truth(scene)
    spec = TIERS[tier]
    assert len(gt.events) >= 0.7 * sum(spec.zone_events.values())
    dt = 1.0 / scene.native_fps
    for e in gt.events:
        tr = next(t for t in scene.tracks if t.id == e.track_id)
        poly = scene.camera(e.camera).zone
        # every native frame within [start, end] is inside; the frame before start and after end are not
        f0, f1 = int(round(e.start / dt)), int(round(e.end / dt))
        for f in range(f0, f1 + 1):
            assert point_in_convex_polygon(tr.foot_at(f * dt), poly)
        before, after = tr.foot_at((f0 - 1) * dt), tr.foot_at((f1 + 1) * dt)
        assert before is None or not point_in_convex_polygon(before, poly)
        assert after is None or not point_in_convex_polygon(after, poly)
        assert e.duration_s == pytest.approx(e.end - e.start + dt, abs=1e-4)


def test_zone_events_do_not_overlap_in_zone_on_same_camera():
    scene = generate_scene("quick", 1)
    gt = compute_ground_truth(scene)
    for cam in scene.cameras:
        ev = sorted(gt.for_camera(cam.name), key=lambda e: e.start)
        for a, b in zip(ev, ev[1:]):
            assert b.start > a.end


def test_distractors_never_enter_zone():
    scene = generate_scene("quick", 2)
    gt = compute_ground_truth(scene)
    kinds = {t.kind for t in gt.distractor_tracks}
    assert "near_miss" in kinds and "wanderer" in kinds
    assert all(t.kind != "zone_event" for t in gt.distractor_tracks)
    # every zone_event track produced a GT event
    assert {e.track_id for e in gt.events} == {t.id for t in scene.tracks if t.kind == "zone_event"}


def test_corpus_variation_covers_the_hard_cases():
    scene = generate_scene("quick", 1)
    gt = compute_ground_truth(scene)
    durations = np.array([e.duration_s for e in gt.events])
    heights = np.array([e.height_px for e in gt.events])
    assert (durations < 0.5).sum() >= 5, "need short events so FPS matters"
    assert (durations > 2.0).sum() >= 5
    assert (heights < 65).sum() >= 5, "need small people so resolution matters"
    assert (heights > 120).sum() >= 5
    assert any(e.occlusion > 0 for e in gt.events)


def test_match_alert_window():
    scene = generate_scene("smoke", 1)
    gt = compute_ground_truth(scene)
    e = gt.events[0]
    assert gt.match_alert(e.camera, e.start) is e
    assert gt.match_alert(e.camera, e.start - 0.4) is e  # lead tolerance
    assert gt.match_alert(e.camera, e.start - 0.6) is not e
    assert gt.match_alert(e.camera, e.end + GRACE_S) is e
    assert gt.match_alert(e.camera, e.end + GRACE_S + 0.5) is not e
    assert gt.match_alert("ZZ", e.start) is None


def test_tracks_stay_in_frame():
    scene = generate_scene("quick", 5)
    for tr in scene.tracks:
        for x, y in tr.waypoints:
            assert 0 <= x <= FRAME_W and 0 <= y <= FRAME_H
