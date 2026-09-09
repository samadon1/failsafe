"""Drone (UAV) real-video holdout — the third Physical-AI domain.

Same compiler, aerial front-end: a downward drone camera watches a ground zone; the critical event is
a VRU (pedestrian/people) inside that zone; the failure that matters is a laggy/lost comms link.

Runs the actual Failsafe workload (YOLOv8n person detector + ground-zone rule + scripted fault
timeline) over one real VisDrone MOT sequence, scoring against VisDrone's per-frame 2D VRU boxes.
Aerial people are small, so detection runs at a higher input resolution (1280) than the ground
cameras; web frames are still written at 960.

Dataset: VisDrone MOT (via the Voxel51/visdrone-mot Hugging Face mirror), sequence uav0000137_00458_v.
  Prepared by scripts (manifest + frames) into datasets/holdout/visdrone/ — see scene.json.
  Non-commercial research license (VisDrone).

Usage:
  uv run python scripts/drone_holdout.py preview     # dump one frame with the ground zone drawn
  uv run python scripts/drone_holdout.py run         # full pass → docs/site/drone/{frames,trace,holdout}
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "holdout" / "visdrone"
SCENE_JSON = DS / "scene.json"
SRC = DS / "frames"
SITE = ROOT / "docs" / "site" / "drone"
FRAMES = SITE / "frames"

DET_W = 1280      # detection input width (aerial objects are small)
OUT_W = 960       # web frame width
OUT_FPS = 10      # VisDrone sequences are ~ this rate
CONF = 0.20       # small aerial objects → lower confidence floor
IOU_HIT = 0.3     # aerial boxes are tiny; a slightly looser hit threshold

# --- ground zone as NORMALISED polygon (0..1), resolution-independent; tuned in `preview`. ---
ZONE_N = [(0.30, 0.34), (0.70, 0.34), (0.80, 0.86), (0.20, 0.86)]

TIMELINE = [
    (0.0,  "healthy",  "Normal operation — cloud confirmation on, tight timeout"),
    (8.0,  "compute",  "Compute pressure — onboard detector runs late"),
    (14.0, "offline",  "Comms / RF link lost"),
    (19.0, "healthy",  "Link restored"),
]
MODES = {
    "healthy": {"name": "NORMAL",           "cloud": True,  "recall": 0.982, "p95": 480},
    "compute": {"name": "NO VERIFIED MODE", "cloud": False, "recall": 0.72,  "p95": 1900, "nvm": True},
    "offline": {"name": "ISLAND",           "cloud": False, "recall": 0.965, "p95": 470},
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

def load_scene():
    return json.loads(SCENE_JSON.read_text())

def cond_at(t):
    c = TIMELINE[0][1]
    for ts, cc, _ in TIMELINE:
        if t >= ts: c = cc
    return c

def preview():
    sc = load_scene(); ow = OUT_W; oh = int(round(sc["h"] * OUT_W / sc["w"]))
    p = sorted(SRC.glob("*.jpg"))[len(list(SRC.glob('*.jpg')))//2]
    fr = cv2.resize(cv2.imread(str(p)), (ow, oh))
    poly = np.array([(int(x*ow), int(y*oh)) for x, y in ZONE_N], np.int32)
    ov = fr.copy(); cv2.fillPoly(ov, [poly], (40, 200, 240)); fr = cv2.addWeighted(ov, 0.2, fr, 0.8, 0)
    cv2.polylines(fr, [poly], True, (40, 200, 240), 2)
    SITE.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(SITE/"_preview.jpg"), fr)
    print(f"scene {sc['scene']} {sc['w']}x{sc['h']} → out {ow}x{oh}; wrote docs/site/drone/_preview.jpg")

def run():
    from failsafe.workload.detector import make_detector
    sc = load_scene(); W0, H0 = sc["w"], sc["h"]
    ow, oh = OUT_W, int(round(H0 * OUT_W / W0))
    dw, dh = DET_W, int(round(H0 * DET_W / W0))
    FRAMES.mkdir(parents=True, exist_ok=True)
    det = make_detector("yolo", "yolov8n.pt", "cpu")
    frames_out = []
    tp = fp = gtp = 0
    zone_gt_frames = zone_agree = 0
    for m in sc["frames"]:
        i = m["i"]; t = i / OUT_FPS
        cond = cond_at(t); mode = MODES[cond]
        img = cv2.imread(str(SRC / f"{i:05d}.jpg"))
        if img is None: continue
        det_frame = cv2.resize(img, (dw, dh))
        dets = det.detect(det_frame, DET_W, CONF)                       # boxes in DET space
        gboxes_d = [[b[0]*dw, b[1]*dh, (b[0]+b[2])*dw, (b[1]+b[3])*dh] for b in (x["bb"] for x in m["vru"])]
        # detection recall / precision in DET space
        gtp += len(gboxes_d); matched = set()
        for d in dets:
            db = [d.x0, d.y0, d.x1, d.y1]; best = -1; bi = -1
            for k, gb in enumerate(gboxes_d):
                if k in matched: continue
                v = iou(db, gb)
                if v > best: best = v; bi = k
            if best >= IOU_HIT: tp += 1; matched.add(bi)
            else: fp += 1
        # ground-zone occupancy in NORMALISED space (foot = bottom-centre)
        gt_in = any(point_in_poly(((b[0]+b[2]/2), (b[1]+b[3])), ZONE_N) for b in (x["bb"] for x in m["vru"]))
        det_in = any(point_in_poly((((d.x0+d.x1)/2)/dw, d.y1/dh), ZONE_N) for d in dets)
        if gt_in:
            zone_gt_frames += 1
            if det_in: zone_agree += 1
        # web frame + trace (OUT space)
        out_frame = cv2.resize(img, (ow, oh))
        cv2.imwrite(str(FRAMES / f"f{i:05d}.jpg"), out_frame, [cv2.IMWRITE_JPEG_QUALITY, 72])
        sx, sy = ow/dw, oh/dh
        dbox_o = [[round(d.x0*sx), round(d.y0*sy), round(d.x1*sx), round(d.y1*sy)] for d in dets]
        gbox_o = [[round(b[0]*ow), round(b[1]*oh), round((b[0]+b[2])*ow), round((b[1]+b[3])*oh)] for b in (x["bb"] for x in m["vru"])]
        frames_out.append({"i": i, "t": round(t, 3), "cond": cond, "mode": mode["name"],
            "det": dbox_o, "gt": gbox_o, "gt_in": gt_in, "det_in": det_in})
        if i % 40 == 0: print(f"  frame {i}/{len(sc['frames'])} (t={t:.1f}s, {cond})", flush=True)
    recall = tp/gtp if gtp else 0; prec = tp/(tp+fp) if (tp+fp) else 0
    zone_poly_out = [[round(x*ow), round(y*oh)] for x, y in ZONE_N]
    holdout = {
        "domain": "drone (UAV aerial)", "source": "VisDrone MOT / " + sc["scene"] +
        " (via Voxel51/visdrone-mot HF mirror)", "license": "VisDrone — non-commercial research",
        "frames": len(frames_out), "src_fps": OUT_FPS, "detector": f"YOLOv8n @{DET_W} CPU",
        "conf": CONF, "iou_hit": IOU_HIT,
        "vru_detection_recall": round(recall, 3), "vru_detection_precision": round(prec, 3),
        "gt_vru_boxes": gtp, "zone_frames_with_vru": zone_gt_frames,
        "zone_agreement": round(zone_agree/zone_gt_frames, 3) if zone_gt_frames else None,
        "note": "Real aerial footage; VRU = pedestrian+people. Detection at 1280 (aerial objects are "
                "small); recall is a floor for the cheapest edge config. Never used by the search.",
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE/"trace.json").write_text(json.dumps({"timeline": TIMELINE, "zone": zone_poly_out, "w": ow, "h": oh,
        "modes": MODES, "frames": frames_out}, separators=(",", ":")))
    (SITE/"holdout.json").write_text(json.dumps(holdout, indent=2))
    print("\nHOLDOUT:", json.dumps(holdout, indent=2))
    print(f"wrote {len(frames_out)} frames + trace.json + holdout.json")

if __name__ == "__main__":
    (preview if (len(sys.argv) > 1 and sys.argv[1] == "preview") else run)()
