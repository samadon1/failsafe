"""Robot proximity-safety holdout on real JRDB frames from Hugging Face (no gated download).

Uses `vladyslava-rudas/jrdb-proxemic-risk` — real JackRabbot robot-camera frames (752x480) with the
CLOSEST person's 2D box and a human-labeled proxemic `danger_level`. This is the safety-critical
person (the nearest human is the collision risk), so it is exactly right for speed-and-separation
monitoring — with the honest caveat that only the closest person is labeled, so we report
closest-person detection + safety-zone agreement rather than the full multi-person recall/precision
triple used in the CCTV/driving/drone holdouts. (The full-GT path stays in robot_holdout.py for a
gated JRDB download.)

Usage:
  uv run python scripts/robot_holdout_hf.py preview   # dump one frame with the safety zone drawn
  uv run python scripts/robot_holdout_hf.py run       # full pass → docs/site/robot/{frames,trace,holdout}
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site" / "robot"
FRAMES = SITE / "frames"
SEQ = "image_8/clark-center-intersection-2019-02-28_0"   # the longest sequence (58 frames)

OUT_W = 960
OUT_FPS = 5
CONF = 0.30
IOU_HIT = 0.4
DANGER = ["high", "moderate", "low", "minimum"]           # ClassLabel order

# safety envelope as NORMALISED polygon (0..1): the robot's near zone, centre-bottom of view.
ZONE_N = [(0.30, 0.42), (0.70, 0.42), (0.86, 1.0), (0.14, 1.0)]

TIMELINE = [
    (0.0,  "healthy",  "Normal operation — cloud planner reachable, tight timeout"),
    (4.0,  "compute",  "Compute pressure — onboard detector runs late"),
    (7.5,  "offline",  "Cloud-planner / remote-assist link lost"),
    (10.0, "healthy",  "Link restored"),
]
MODES = {
    "healthy": {"name": "NORMAL",           "cloud": True,  "recall": 0.982, "p95": 380},
    "compute": {"name": "NO VERIFIED MODE", "cloud": False, "recall": 0.72,  "p95": 1900, "nvm": True},
    "offline": {"name": "ISLAND",           "cloud": False, "recall": 0.965, "p95": 360},
}

def point_in_poly(p, poly):
    x, y = p; inside = False; n = len(poly); j = n - 1
    for i in range(n):
        xi, yi = poly[i]; xj, yj = poly[j]
        if ((yi > y) != (yj > y)) and (x < (xj - xi) * (y - yi) / (yj - yi) + xi):
            inside = not inside
        j = i
    return inside

def iou(a, b):
    ix0, iy0 = max(a[0], b[0]), max(a[1], b[1]); ix1, iy1 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0); inter = iw * ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / ua if ua > 0 else 0

def load_seq():
    from datasets import load_dataset, concatenate_datasets
    parts = [load_dataset("vladyslava-rudas/jrdb-proxemic-risk", split=s)
             for s in ("finetuning", "evaluating", "validating")]
    D = concatenate_datasets(parts)
    rows = [r for r in D if r["sequence_id"] == SEQ]
    rows.sort(key=lambda r: int(r["frame_id"]))
    return rows

def cond_at(t):
    c = TIMELINE[0][1]
    for ts, cc, _ in TIMELINE:
        if t >= ts: c = cc
    return c

def pil_to_bgr(im):
    return cv2.cvtColor(np.array(im.convert("RGB")), cv2.COLOR_RGB2BGR)

def preview():
    rows = load_seq(); r = rows[len(rows)//2]
    im = pil_to_bgr(r["image"]); h, w = im.shape[:2]
    ow, oh = OUT_W, int(round(h*OUT_W/w)); fr = cv2.resize(im, (ow, oh))
    poly = np.array([(int(x*ow), int(y*oh)) for x, y in ZONE_N], np.int32)
    ov = fr.copy(); cv2.fillPoly(ov, [poly], (40, 200, 240)); fr = cv2.addWeighted(ov, 0.2, fr, 0.8, 0)
    cv2.polylines(fr, [poly], True, (40, 200, 240), 2)
    bb = r["closest_person_bbox"]; s = ow/w   # [x, y, w, h]
    cv2.rectangle(fr, (int(bb[0]*s), int(bb[1]*s)), (int((bb[0]+bb[2])*s), int((bb[1]+bb[3])*s)), (0, 220, 0), 2)
    SITE.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(SITE/"_preview.jpg"), fr)
    print(f"{SEQ}: {len(rows)} frames, {w}x{h} → {ow}x{oh}; wrote docs/site/robot/_preview.jpg (green=closest GT)")

def run():
    from failsafe.workload.detector import make_detector
    rows = load_seq()
    im0 = pil_to_bgr(rows[0]["image"]); H0, W0 = im0.shape[:2]
    ow, oh = OUT_W, int(round(H0*OUT_W/W0)); s = ow/W0
    det = make_detector("yolo", "yolov8n.pt", "cpu")
    FRAMES.mkdir(parents=True, exist_ok=True)
    frames_out = []
    hit = 0; ngt = 0                       # closest-person detection
    zone_gt = zone_ok = 0                  # safety-zone occupancy agreement
    dang_hit = dang_n = 0                  # closest-person recall on high/moderate-danger frames
    for i, r in enumerate(rows):
        t = i / OUT_FPS; cond = cond_at(t)
        im = pil_to_bgr(r["image"]); frame = cv2.resize(im, (ow, oh))
        dets = det.detect(frame, OUT_W, CONF)
        dboxes = [[d.x0, d.y0, d.x1, d.y1] for d in dets]
        bb = r["closest_person_bbox"]  # [x, y, w, h] in native pixels
        gt = [bb[0]*s, bb[1]*s, (bb[0]+bb[2])*s, (bb[1]+bb[3])*s]
        danger = DANGER[r["danger_level"]]
        # closest-person detection
        ngt += 1
        best = max((iou(db, gt) for db in dboxes), default=0)
        got = best >= IOU_HIT
        if got: hit += 1
        if danger in ("high", "moderate"):
            dang_n += 1;  dang_hit += 1 if got else 0
        # safety-zone occupancy (foot = bottom-centre)
        gt_in = point_in_poly(((gt[0]+gt[2])/2/ow, gt[3]/oh), ZONE_N)
        det_in = any(point_in_poly((((d.x0+d.x1)/2)/ow, d.y1/oh), ZONE_N) for d in dets)
        if gt_in:
            zone_gt += 1
            if det_in: zone_ok += 1
        cv2.imwrite(str(FRAMES / f"f{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 78])
        frames_out.append({"i": i, "t": round(t, 3), "cond": cond, "mode": MODES[cond]["name"],
            "det": [[round(x) for x in b] for b in dboxes],
            "gt": [[round(x) for x in gt]], "gt_in": gt_in, "det_in": det_in, "danger": danger})
    zone_poly = [[round(x*ow), round(y*oh)] for x, y in ZONE_N]
    holdout = {
        "domain": "robot (human–robot proximity safety)",
        "source": "JRDB via Hugging Face vladyslava-rudas/jrdb-proxemic-risk / " + SEQ,
        "license": "JRDB — research (HF mirror)", "frames": len(frames_out), "src_fps": OUT_FPS,
        "detector": f"YOLOv8n @{OUT_W} CPU", "conf": CONF, "iou_hit": IOU_HIT,
        "closest_person_recall": round(hit/ngt, 3) if ngt else None,
        "closest_person_recall_high_danger": round(dang_hit/dang_n, 3) if dang_n else None,
        "safety_zone_agreement": round(zone_ok/zone_gt, 3) if zone_gt else None,
        "zone_frames_with_person": zone_gt, "gt_frames": ngt,
        "note": "Only the CLOSEST person is labeled (the safety-critical one), so we report "
                "closest-person detection + safety-zone agreement, not the full recall/precision triple. "
                "Real robot-camera footage; never used by the search. Not a certified safety system.",
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE/"trace.json").write_text(json.dumps({"timeline": TIMELINE, "zone": zone_poly, "w": ow, "h": oh,
        "modes": MODES, "frames": frames_out}, separators=(",", ":")))
    (SITE/"holdout.json").write_text(json.dumps(holdout, indent=2))
    print("\nHOLDOUT:", json.dumps(holdout, indent=2))
    print(f"wrote {len(frames_out)} frames + trace.json + holdout.json")

if __name__ == "__main__":
    (preview if (len(sys.argv) > 1 and sys.argv[1] == "preview") else run)()
