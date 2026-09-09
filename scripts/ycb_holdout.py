"""Robot manipulation / object-detection holdout — the fifth Physical-AI domain.

Same compiler, object-detection front-end: a manipulation robot's camera must recognise the graspable
objects on the table (not people), and keep doing so when the cloud planner / VLM goes slow or drops.
Runs the actual detector + a workspace zone + the scripted fault timeline over one real YCB-Video
sequence, scoring class-aware against YCB's per-frame 2D object boxes.

Honest scope: the edge detector is generic YOLOv8n (COCO classes), so it can only recognise the YCB
objects that are COCO classes — bottle / bowl / cup(mug) / banana / scissors. We score recall/precision
on those known classes; the domain-specific objects (cracker box, drill, clamp…) are outside its
vocabulary — exactly the capability gap Failsafe surfaces (the mission would need a fine-tuned
detector). Frames are pulled a subsample at a time via HTTP range requests (no 265 GB download).

Dataset: YCB-Video via Linpeng502502/YCB_Video_Dataset (per-sequence zips). BSD/research license.

Usage:
  uv run python scripts/ycb_holdout.py fetch     # pull a subsample of one sequence (color + boxes)
  uv run python scripts/ycb_holdout.py preview    # dump one frame with the workspace zone + GT
  uv run python scripts/ycb_holdout.py run        # full pass → docs/site/manip/{frames,trace,holdout}
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEQ = os.environ.get("YCB_SEQ", "0000")
DS = ROOT / "datasets" / "holdout" / "ycb" / SEQ
SITE = ROOT / "docs" / "site" / "manip"
FRAMES = SITE / "frames"
ZIP_URL = f"https://huggingface.co/datasets/Linpeng502502/YCB_Video_Dataset/resolve/main/data/{SEQ}.zip"

EVERY = 10               # subsample: keep every Nth frame
OUT_W = 800              # YCB color is 640x480; upscale a touch for the web
CONF = 0.30
IOU_HIT = 0.4

# YCB object name -> COCO class the generic detector knows
YCB2COCO = {"006_mustard_bottle": "bottle", "021_bleach_cleanser": "bottle", "011_banana": "banana",
            "024_bowl": "bowl", "025_mug": "cup", "037_scissors": "scissors"}
TARGETS = set(YCB2COCO.values())

# workspace zone (normalised): the graspable area in front of the arm — centre of the table.
ZONE_N = [(0.22, 0.30), (0.78, 0.30), (0.90, 0.95), (0.10, 0.95)]

TIMELINE = [
    (0.0, "healthy",  "Normal operation — cloud VLM reachable, tight timeout"),
    (3.5, "compute",  "Compute pressure — onboard detector runs late"),
    (6.5, "offline",  "Cloud planner / VLM link lost"),
    (9.0, "healthy",  "Link restored"),
]
MODES = {
    "healthy": {"name": "NORMAL", "cloud": True}, "compute": {"name": "NO VERIFIED MODE", "cloud": False},
    "offline": {"name": "ISLAND", "cloud": False},
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

def fetch():
    from remotezip import RemoteZip
    DS.mkdir(parents=True, exist_ok=True)
    with RemoteZip(ZIP_URL) as z:
        color = sorted(n for n in z.namelist() if n.endswith("-color.png"))
        keep = color[::EVERY]
        print(f"seq {SEQ}: {len(color)} frames → keeping {len(keep)}")
        for i, c in enumerate(keep):
            stem = c[:-len("-color.png")]                     # e.g. 0000/000123
            (DS / f"{i:05d}.png").write_bytes(z.read(c))
            (DS / f"{i:05d}.txt").write_bytes(z.read(stem + "-box.txt"))
            if i % 20 == 0: print("  ", i, flush=True)
    print("fetched →", DS)

def load_gt(idx):
    """box.txt: '<ycb_class> x1 y1 x2 y2' → list of (coco_class, [x1,y1,x2,y2]) for COCO-known objects."""
    p = DS / f"{idx:05d}.txt"
    out = []
    for line in p.read_text().splitlines():
        t = line.split()
        if not t or t[0] not in YCB2COCO: continue
        out.append((YCB2COCO[t[0]], [float(t[1]), float(t[2]), float(t[3]), float(t[4])]))
    return out

def frames():
    return sorted(DS.glob("*.png"))

def preview():
    fs = frames()
    if not fs: raise SystemExit("no frames — run `fetch` first")
    im = cv2.imread(str(fs[len(fs)//2])); h, w = im.shape[:2]
    ow, oh = OUT_W, int(round(h*OUT_W/w)); fr = cv2.resize(im, (ow, oh)); s = ow/w
    poly = np.array([(int(x*ow), int(y*oh)) for x, y in ZONE_N], np.int32)
    ov = fr.copy(); cv2.fillPoly(ov, [poly], (40, 200, 240)); fr = cv2.addWeighted(ov, 0.2, fr, 0.8, 0)
    cv2.polylines(fr, [poly], True, (40, 200, 240), 2)
    for cls, b in load_gt(len(fs)//2):
        cv2.rectangle(fr, (int(b[0]*s), int(b[1]*s)), (int(b[2]*s), int(b[3]*s)), (0, 220, 0), 2)
        cv2.putText(fr, cls, (int(b[0]*s), int(b[1]*s)-4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    SITE.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(SITE/"_preview.jpg"), fr)
    print(f"{SEQ} {w}x{h} → {ow}x{oh}; wrote docs/site/manip/_preview.jpg (green=COCO-known GT)")

def cond_at(t):
    c = TIMELINE[0][1]
    for ts, cc, _ in TIMELINE:
        if t >= ts: c = cc
    return c

def run():
    from ultralytics import YOLO
    fs = frames()
    if not fs: raise SystemExit("no frames — run `fetch` first")
    im0 = cv2.imread(str(fs[0])); H0, W0 = im0.shape[:2]
    ow, oh = OUT_W, int(round(H0*OUT_W/W0)); s = ow/W0
    model = YOLO("yolov8n.pt")
    FRAMES.mkdir(parents=True, exist_ok=True)
    OUT_FPS = 7
    tp = fp = gtp = 0; zone_gt = zone_ok = 0; frames_out = []
    for i, p in enumerate(fs):
        t = i / OUT_FPS; cond = cond_at(t)
        im = cv2.imread(str(p)); frame = cv2.resize(im, (ow, oh))
        res = model.predict(frame, imgsz=OUT_W, conf=CONF, verbose=False)[0]
        dets = []  # (coco_class, [x0,y0,x1,y1]) in OUT space
        for b, c in zip(res.boxes.xyxy.tolist(), res.boxes.cls.tolist()):
            name = model.names[int(c)]
            if name in TARGETS: dets.append((name, b))
        gt = [(cls, [b[0]*s, b[1]*s, b[2]*s, b[3]*s]) for cls, b in load_gt(i)]  # OUT space
        gtp += len(gt); matched = set()
        for dn, db in dets:
            best = -1; bi = -1
            for k, (gn, gb) in enumerate(gt):
                if k in matched or gn != dn: continue
                v = iou(db, gb)
                if v > best: best = v; bi = k
            if best >= IOU_HIT: tp += 1; matched.add(bi)
            else: fp += 1
        gt_in = any(point_in_poly(((b[0]+b[2])/2/ow, (b[1]+b[3])/2/oh), ZONE_N) for _, b in gt)
        det_in = any(point_in_poly(((b[0]+b[2])/2/ow, (b[1]+b[3])/2/oh), ZONE_N) for _, b in dets)
        if gt_in:
            zone_gt += 1
            if det_in: zone_ok += 1
        cv2.imwrite(str(FRAMES / f"f{i:05d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
        frames_out.append({"i": i, "t": round(t, 3), "cond": cond, "mode": MODES[cond]["name"],
            "det": [[round(x) for x in b] for _, b in dets],
            "gt": [[round(x) for x in b] for _, b in gt], "gt_in": gt_in, "det_in": det_in,
            "labels": [n for n, _ in dets]})
    recall = tp/gtp if gtp else 0; prec = tp/(tp+fp) if (tp+fp) else 0
    zone_poly = [[round(x*ow), round(y*oh)] for x, y in ZONE_N]
    holdout = {"domain": "robot manipulation (object detection)",
        "source": f"YCB-Video / seq {SEQ} (Linpeng502502/YCB_Video_Dataset)", "license": "YCB-Video — research",
        "frames": len(frames_out), "src_fps": OUT_FPS, "detector": f"YOLOv8n @{OUT_W} CPU (COCO classes)",
        "conf": CONF, "iou_hit": IOU_HIT, "scored_classes": sorted(TARGETS),
        "object_detection_recall": round(recall, 3), "object_detection_precision": round(prec, 3),
        "gt_object_boxes": gtp, "workspace_frames_with_object": zone_gt,
        "workspace_agreement": round(zone_ok/zone_gt, 3) if zone_gt else None,
        "note": "Class-aware recall/precision on the COCO-known YCB objects only (bottle/bowl/cup/"
                "banana/scissors); domain-specific objects are outside the generic detector's vocabulary "
                "— a capability gap Failsafe surfaces. Never used by the search. Not certified."}
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE/"trace.json").write_text(json.dumps({"timeline": TIMELINE, "zone": zone_poly, "w": ow, "h": oh,
        "modes": MODES, "frames": frames_out}, separators=(",", ":")))
    (SITE/"holdout.json").write_text(json.dumps(holdout, indent=2))
    print("\nHOLDOUT:", json.dumps(holdout, indent=2))

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "fetch"
    {"fetch": fetch, "preview": preview, "run": run}.get(cmd, fetch)()
