"""The robot (JRDB) front-end is code-complete and tested even though the gated data isn't present.

Exercises both JRDB label parsers, the zone geometry, and the scoring core via the script's own
self-test, plus verifies the mission spec loads.
"""
import importlib.util
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]

def _load_robot_holdout():
    spec = importlib.util.spec_from_file_location("robot_holdout", ROOT / "scripts" / "robot_holdout.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod

def test_jrdb_json_parser(tmp_path):
    rh = _load_robot_holdout()
    p = tmp_path / "seq.json"
    p.write_text('{"labels": {"000000.jpg": [{"box": [10, 20, 30, 40], "label_id": "pedestrian:1"},'
                 '{"box": [0,0,5,5], "label_id": "car:2"}]}}')
    out = rh.parse_jrdb_json(p)
    assert out["000000.jpg"] == [[10.0, 20.0, 40.0, 60.0]]  # [x,y,w,h] -> [x0,y0,x1,y1], car dropped

def test_kitti_txt_parser(tmp_path):
    rh = _load_robot_holdout()
    d = tmp_path / "seq"; d.mkdir()
    (d / "000000.txt").write_text("Pedestrian 0 0 -1 -10 5 6 7 8 0 0 0 0 0 0 0\n"
                                  "Car 0 0 -1 -10 1 1 2 2 0 0 0 0 0 0 0\n")
    out = rh.parse_kitti_txt(d)
    assert out["000000"] == [[5.0, 6.0, 7.0, 8.0]]

def test_point_in_poly_and_selftest():
    rh = _load_robot_holdout()
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert rh.point_in_poly((5, 5), sq) is True
    assert rh.point_in_poly((15, 5), sq) is False
    rh.selftest()  # full parser + scoring pipeline; raises on any mismatch

def test_mission_loads():
    d = yaml.safe_load((ROOT / "missions" / "robot-safety-zone.yaml").read_text())
    assert d["name"] == "robot-safety-zone-monitor"
    assert any(inv["metric"] == "critical_event_recall" for inv in d["invariants"])
