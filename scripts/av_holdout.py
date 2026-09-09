"""Self-driving (AV) real-video holdout — the second Physical-AI domain.

Runs the ACTUAL Failsafe workload (YOLOv8n person detector + an ego-path zone rule + the scripted
fault timeline) over one real KITTI tracking dashcam sequence, scoring detections against KITTI's
2D ground-truth boxes for vulnerable road users (Pedestrian + Cyclist).

Same compiler, different front-end: a dashcam replaces the warehouse camera, the ego path replaces
the restricted zone, VRU recall replaces person recall. This is a real holdout — the sequence is
never used by any search.

Dataset: KITTI multi-object tracking, sequence 0016 (dense urban crossing), CC-BY-NC-SA 3.0.
  images: datasets/holdout/kitti/training/image_02/0016/*.png
  labels: datasets/holdout/kitti/training/label_02/0016.txt   (columns:
          frame track_id type truncated occluded alpha  bbox_l bbox_t bbox_r bbox_b  h w l  x y z  ry)

Usage:
  uv run python scripts/av_holdout.py preview     # dump one frame with the ego-path drawn (tune ZONE)
  uv run python scripts/av_holdout.py run         # full pass → docs/site/av/{frames,trace.json,holdout.json}
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
KITTI = ROOT / "datasets" / "holdout" / "kitti" / "training"
SEQ = "0016"
IMG_DIR = KITTI / "image_02" / SEQ
LBL = KITTI / "label_02" / f"{SEQ}.txt"
SITE = ROOT / "docs" / "site" / "av"
FRAMES = SITE / "frames"

VRU_TYPES = {"Pedestrian", "Cyclist"}
SRC_FPS = 10                     # KITTI tracking capture rate
OUT_FPS = 10                     # process every frame
OUT_W = 960                      # web frame width
CONF = 0.30                      # a moving dashcam frame; VRUs are small/distant
IOU_HIT = 0.4

# --- ego-path zone (polygon in OUTPUT pixel space; tuned in `preview`). Trapezoid: near lane ahead. ---
# output height is set from the real image aspect at run time; these y's assume ~960x291 (KITTI 1242x375).
ZONE = [(392, 150), (556, 150), (860, 289), (150, 289)]

# --- scripted fault timeline (seconds within the clip) ---
TIMELINE = [
    (0.0,  "healthy",  "Normal operation — cloud confirmation on, tight timeout"),
    (7.0,  "compute",  "Compute pressure — perception outputs late"),
    (13.0, "offline",  "Remote-assist / cloud link lost"),
    (18.0, "healthy",  "Link restored"),
]
MODES = {
    "healthy": {"name": "NORMAL",           "cloud": True,  "recall": 0.982, "p95": 480},
    "compute": {"name": "NO VERIFIED MODE", "cloud": False, "recall": 0.72,  "p95": 1900, "nvm": True},
    "offline": {"name": "ISLAND",           "cloud": False, "recall": 0.965, "p95": 470},
}

def point_in_poly(p, poly):
    x, y = p; inside = False; n = len(poly)
    j = n - 1
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

def load_gt(scale):
    """frame -> list of VRU boxes [x0,y0,x1,y1] in OUTPUT pixel space."""
    out = {}
    for line in LBL.read_text().splitlines():
        p = line.split()
        if len(p) < 10: continue
        fr = int(p[0]); typ = p[2]
        if typ not in VRU_TYPES: continue
        l, t, r, b = (float(p[6]) * scale, float(p[7]) * scale, float(p[8]) * scale, float(p[9]) * scale)
        out.setdefault(fr, []).append([l, t, r, b])
    return out

def cond_at(t):
    c = TIMELINE[0][1]
    for ts, cc, _ in TIMELINE:
        if t >= ts: c = cc
    return c

def frame_paths():
    return sorted(IMG_DIR.glob("*.png"))

def out_size():
    fs = frame_paths()
    if not fs: raise SystemExit(f"no frames in {IMG_DIR} — extract KITTI seq {SEQ} first")
    h, w = cv2.imread(str(fs[0])).shape[:2]
    return w, h, OUT_W, int(round(h * OUT_W / w)), OUT_W / w

def preview():
    fs = frame_paths(); w, h, ow, oh, scale = out_size()
    fr = cv2.resize(cv2.imread(str(fs[len(fs)//2])), (ow, oh))
    poly = np.array(ZONE, np.int32)
    ov = fr.copy(); cv2.fillPoly(ov, [poly], (40, 200, 240)); fr = cv2.addWeighted(ov, 0.20, fr, 0.80, 0)
    cv2.polylines(fr, [poly], True, (40, 200, 240), 2)
    SITE.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SITE / "_preview.jpg"), fr)
    print(f"image {w}x{h} → output {ow}x{oh} (scale {scale:.3f}); wrote docs/site/av/_preview.jpg — tune ZONE")

def run():
    from failsafe.workload.detector import make_detector
    fs = frame_paths(); w, h, ow, oh, scale = out_size()
    FRAMES.mkdir(parents=True, exist_ok=True)
    gt = load_gt(scale)
    det = make_detector("yolo", "yolov8n.pt", "cpu")
    frames_out = []
    tp = fp = gtp = 0
    path_gt_frames = path_agree = 0
    idx = 0
    for f, p in enumerate(fs):
        t = f / SRC_FPS
        cond = cond_at(t); mode = MODES[cond]
        small = cv2.resize(cv2.imread(str(p)), (ow, oh))
        dets = det.detect(small, 640, CONF)
        dboxes = [[d.x0, d.y0, d.x1, d.y1] for d in dets]
        gboxes = gt.get(f, [])
        # detection recall / precision vs VRU GT
        gtp += len(gboxes); matched = set()
        for db in dboxes:
            best = -1; bi = -1
            for i, gb in enumerate(gboxes):
                if i in matched: continue
                v = iou(db, gb)
                if v > best: best = v; bi = i
            if best >= IOU_HIT: tp += 1; matched.add(bi)
            else: fp += 1
        # ego-path occupancy (foot point = bottom-center)
        gt_in = any(point_in_poly(((g[0]+g[2])/2, g[3]), ZONE) for g in gboxes)
        det_in = any(point_in_poly(((d.x0+d.x1)/2, d.y1), ZONE) for d in dets)
        if gt_in:
            path_gt_frames += 1
            if det_in: path_agree += 1
        name = f"f{idx:05d}.jpg"
        cv2.imwrite(str(FRAMES / name), small, [cv2.IMWRITE_JPEG_QUALITY, 72])
        frames_out.append({"i": idx, "t": round(t, 3), "cond": cond, "mode": mode["name"],
            "det": [[round(x) for x in b] for b in dboxes],
            "gt": [[round(x) for x in b] for b in gboxes],
            "gt_in": gt_in, "det_in": det_in})
        idx += 1
        if idx % 40 == 0: print(f"  frame {idx}/{len(fs)} (t={t:.1f}s, {cond})", flush=True)
    recall = tp/gtp if gtp else 0; prec = tp/(tp+fp) if (tp+fp) else 0
    holdout = {
        "domain": "self-driving (dashcam)",
        "source": "KITTI multi-object tracking / training / sequence 0016 (urban crossing)",
        "license": "CC-BY-NC-SA 3.0", "frames": idx, "src_fps": SRC_FPS,
        "detector": "YOLOv8n @640 CPU", "conf": CONF, "iou_hit": IOU_HIT,
        "vru_detection_recall": round(recall, 3), "vru_detection_precision": round(prec, 3),
        "gt_vru_boxes": gtp,
        "egopath_frames_with_vru": path_gt_frames,
        "egopath_agreement": round(path_agree / path_gt_frames, 3) if path_gt_frames else None,
        "note": "Real dashcam footage (Pedestrian+Cyclist GT). VRU = person/rider; YOLO detects persons, "
                "so cyclist IoU is conservative. Never used by the search. Not a certified system.",
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE/"trace.json").write_text(json.dumps({"timeline": TIMELINE, "zone": ZONE, "w": ow, "h": oh,
        "modes": MODES, "frames": frames_out}, separators=(",", ":")))
    (SITE/"holdout.json").write_text(json.dumps(holdout, indent=2))
    print("\nHOLDOUT:", json.dumps(holdout, indent=2))
    print(f"wrote {idx} frames + trace.json + holdout.json")

if __name__ == "__main__":
    (preview if (len(sys.argv) > 1 and sys.argv[1] == "preview") else run)()
