"""Real-video holdout + release-demo trace.

Runs the Failsafe detector stage (YOLOv8n person detector + restricted-zone logic) over one camera
of NVIDIA's PhysicalAI-SmartSpaces warehouse dataset, scoring detections against the dataset's
ground-truth person boxes, under a scripted fault timeline. What is measured here is the detector on
real footage. The per-condition mode labels and their recall / p95 (MODES below) are copied from the
compiled policy's synthetic-corpus measurements for the release demo; this script does not run the
policy runtime, and the real clip contains one zone event, so it does not validate the policy.

Outputs:
  docs/site/frames/f######.jpg   sampled frames (web-sized)
  docs/site/trace.json           per-frame boxes + zone status + active mode + events + rolling metrics
  docs/site/holdout.json         overall REAL recall / precision on the clip (the §7 validation)

This is the real-data holdout (discovery corpus was synthetic; this clip is never used by search).
Usage:
  uv run python scripts/real_holdout.py preview      # dump one frame with the candidate zone drawn
  uv run python scripts/real_holdout.py run          # full pass
"""
from __future__ import annotations

import json, sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DS = ROOT / "datasets" / "holdout" / "warehouse15"
VID = DS / "Camera.mp4"
GT = DS / "ground_truth.json"
CAM = "Camera"
SITE = ROOT / "docs" / "site"
FRAMES = SITE / "frames"

# --- segment + sampling ---
START_S, END_S = 60.0, 105.0     # 45 s window
SRC_FPS = 30
OUT_FPS = 6                       # detection + playback rate
OUT_W = 960                      # web frame width (source 1920 → /2)
SCALE = OUT_W / 1920.0
CONF = 0.35

# --- restricted zone (polygon in OUTPUT pixel space, 960x540). Tuned in `preview`. ---
ZONE = [(300, 250), (620, 250), (700, 470), (250, 470)]

# --- scripted fault timeline (seconds within the segment) ---
TIMELINE = [
    (0.0,  "healthy",  "Normal operation — cloud confirmation on, 1 s timeout"),
    (14.0, "offline",  "WAN cut — cloud unreachable"),
    (30.0, "healthy",  "WAN restored"),
    (36.0, "compute",  "Compute pressure — detector 3× slower"),
]

# per-condition behaviour drawn from measured modes (recall/p95 used only for the naive contrast)
MODES = {
    "healthy": {"name": "NORMAL",   "cloud": True,  "recall": 0.982, "p95": 902},
    "offline": {"name": "ISLAND",   "cloud": False, "recall": 0.965, "p95": 970},
    "compute": {"name": "NO VERIFIED MODE", "cloud": False, "recall": 0.72, "p95": 1900, "nvm": True},
}

def point_in_poly(p, poly):
    x, y = p; s = 0
    for i in range(len(poly)):
        a, b = poly[i], poly[(i+1) % len(poly)]
        cr = (b[0]-a[0])*(y-a[1]) - (b[1]-a[1])*(x-a[0])
        if abs(cr) < 1e-9: continue
        sg = 1 if cr > 0 else -1
        if s == 0: s = sg
        elif sg != s: return False
    return True

def iou(a, b):
    ix0,iy0 = max(a[0],b[0]), max(a[1],b[1]); ix1,iy1 = min(a[2],b[2]), min(a[3],b[3])
    iw,ih = max(0,ix1-ix0), max(0,iy1-iy0); inter = iw*ih
    ua = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter/ua if ua > 0 else 0

def load_gt():
    print("loading ground_truth.json (305 MB, one-time)…", flush=True)
    g = json.loads(GT.read_text())
    out = {}
    for fid, objs in g.items():
        fr = int(fid); people = []
        for o in objs:
            if o.get("object type") != "Person": continue
            box = o.get("2d bounding box visible", {}).get(CAM)
            if not box: continue
            x0,y0,x1,y1 = [v*SCALE for v in box]
            people.append([x0,y0,x1,y1])
        if people: out[fr] = people
    print(f"  GT persons on {CAM}: {sum(len(v) for v in out.values())} boxes across {len(out)} frames")
    return out

def cond_at(t):
    c = TIMELINE[0][1]
    for ts, cc, _ in TIMELINE:
        if t >= ts: c = cc
    return c

def preview():
    cap = cv2.VideoCapture(str(VID)); cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_S*SRC_FPS)+150)
    ok, fr = cap.read(); cap.release()
    fr = cv2.resize(fr, (OUT_W, int(1080*SCALE)))
    poly = np.array(ZONE, np.int32)
    cv2.polylines(fr, [poly], True, (40,200,240), 2)
    ov = fr.copy(); cv2.fillPoly(ov, [poly], (40,200,240)); fr = cv2.addWeighted(ov,0.18,fr,0.82,0)
    SITE.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(SITE/"_preview.jpg"), fr); print("wrote docs/site/_preview.jpg  (check the zone, then edit ZONE)")

def run():
    from failsafe.workload.detector import make_detector
    from failsafe.workload.zone import Detection
    FRAMES.mkdir(parents=True, exist_ok=True)
    gt = load_gt()
    det = make_detector("yolo", "yolov8n.pt", "cpu")
    cap = cv2.VideoCapture(str(VID))
    step = int(round(SRC_FPS / OUT_FPS))
    f0, f1 = int(START_S*SRC_FPS), int(END_S*SRC_FPS)
    frames_out = []
    # real detection scoring (against GT persons), independent of the fault story
    tp = fp = gtp = 0
    # zone-event scoring
    zone_gt_events = 0; zone_detected = 0
    prev_gt_in = False
    idx = 0
    for f in range(f0, f1, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f); ok, fr = cap.read()
        if not ok: break
        t = (f - f0)/SRC_FPS
        cond = cond_at(t); mode = MODES[cond]
        small = cv2.resize(fr, (OUT_W, int(1080*SCALE)))
        dets = det.detect(small, 640, CONF)
        dboxes = [[d.x0,d.y0,d.x1,d.y1] for d in dets]
        gboxes = gt.get(f, [])
        # detection recall/precision vs GT
        gtp += len(gboxes)
        matched=set()
        for db in dboxes:
            best=-1; bi=-1
            for i,gb in enumerate(gboxes):
                if i in matched: continue
                v=iou(db,gb)
                if v>best: best=v; bi=i
            if best>=0.4: tp+=1; matched.add(bi)
            else: fp+=1
        # zone status (foot point = bottom-center)
        gt_in = any(point_in_poly(((g[0]+g[2])/2, g[3]), ZONE) for g in gboxes)
        det_in = any(point_in_poly(((d.x0+d.x1)/2, d.y1), ZONE) for d in dets)
        if gt_in and not prev_gt_in:
            zone_gt_events += 1
            # detected if the active mode would alert (cloud/local); NVM/naive may miss
            if det_in: zone_detected += 1
        prev_gt_in = gt_in
        # write frame
        name=f"f{idx:05d}.jpg"; cv2.imwrite(str(FRAMES/name), small, [cv2.IMWRITE_JPEG_QUALITY,72])
        frames_out.append({
            "i": idx, "t": round(t,3), "cond": cond, "mode": mode["name"],
            "det": [[round(x) for x in b] for b in dboxes],
            "gt":  [[round(x) for x in b] for b in gboxes],
            "gt_in": gt_in, "det_in": det_in,
        })
        idx += 1
        if idx % 20 == 0: print(f"  frame {idx} (t={t:.1f}s, {cond})", flush=True)
    cap.release()
    recall = tp/gtp if gtp else 0; prec = tp/(tp+fp) if (tp+fp) else 0
    holdout = {
        "source":"NVIDIA PhysicalAI-SmartSpaces / MTMC_Tracking_2025 / val / Warehouse_015 / Camera",
        "license":"CC-BY 4.0","frames":idx,"segment_s":[START_S,END_S],"out_fps":OUT_FPS,
        "detector":"YOLOv8n @640 CPU","conf":CONF,
        "person_detection_recall":round(recall,3),"person_detection_precision":round(prec,3),
        "gt_person_boxes":gtp,"zone_gt_events":zone_gt_events,
        "note":"Real detections on real (Omniverse-generated) footage, scored against dataset ground truth. Never used by the search.",
    }
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE/"trace.json").write_text(json.dumps({"timeline":TIMELINE,"zone":ZONE,"w":OUT_W,"h":int(1080*SCALE),
        "modes":MODES,"frames":frames_out}, separators=(",",":")))
    (SITE/"holdout.json").write_text(json.dumps(holdout, indent=2))
    print("\nHOLDOUT:", json.dumps(holdout, indent=2))
    print(f"wrote {idx} frames + trace.json + holdout.json")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "preview"
    (preview if cmd == "preview" else run)()
