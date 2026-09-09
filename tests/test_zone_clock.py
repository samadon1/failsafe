from failsafe.workload.clock import SimulatedClock
from failsafe.workload.zone import Detection, ZoneAlertState, detections_in_zone

SQ = [(100, 100), (200, 100), (200, 200), (100, 200)]


def test_detection_foot_point_rule():
    inside = Detection(140, 20, 160, 150, 0.9)  # foot (150,150) inside
    outside_box_overlaps = Detection(140, 20, 160, 99, 0.9)  # box overlaps region but feet above the zone
    assert detections_in_zone([inside, outside_box_overlaps], SQ) == [inside]


def test_alert_state_first_frame_and_refractory():
    s = ZoneAlertState(confirm_frames=1, refractory_s=1.5)
    assert s.update(0.0, True) is True
    assert s.update(0.1, True) is False  # no burst
    assert s.update(0.2, False) is False
    assert s.update(1.0, True) is False  # still within refractory
    assert s.update(1.6, True) is True  # refractory elapsed → a new intrusion can alert


def test_alert_state_confirm_frames():
    s = ZoneAlertState(confirm_frames=3, refractory_s=1.0)
    assert not s.update(0.0, True)
    assert not s.update(0.1, True)
    assert s.update(0.2, True)
    # a clear frame resets the streak
    assert not s.update(1.5, False)
    assert not s.update(1.6, True)
    assert not s.update(1.7, True)
    assert s.update(1.8, True)


def test_alert_state_reject_rearms():
    s = ZoneAlertState()
    assert s.update(0.0, True)
    s.reject()
    assert s.update(0.1, True)


def test_simulated_clock():
    c = SimulatedClock()
    assert c.now() == 0.0
    c.sleep_until(2.5)
    assert c.now() == 2.5
    c.sleep_until(1.0)  # never goes backwards
    assert c.now() == 2.5
    c.sleep(0.5)
    assert c.now() == 3.0
    assert not c.is_real_time
