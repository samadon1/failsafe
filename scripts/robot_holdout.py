"""Robot (human–robot proximity safety) holdout — the fourth Physical-AI domain.

Same compiler, robot front-end: a robot's camera watches for a human entering its safety envelope
(speed-and-separation monitoring); the failure that matters is a lost cloud-planner / remote-assist
link. Runs the actual Failsafe workload (YOLOv8n person detector + safety-zone rule + scripted fault
timeline) over one real JRDB sequence, scoring against JRDB's per-frame 2D pedestrian boxes.

JRDB (Stanford JackRabbot) is a social-navigation robot dataset — a robot's own cameras among people,
with 2D pedestrian ground truth. It is gated (registration/licence), so this script is code-complete
and unit-tested against JRDB's documented formats; you provide the download, then it produces real
numbers exactly like the CCTV / self-driving / drone holdouts.

Expected layout after you download JRDB (point JRDB_DIR at your copy):
  $JRDB_DIR/images/image_stitched/<seq>/<frame>.jpg     (stitched panorama; or an individual camera)
  $JRDB_DIR/labels/<seq>.json                           (JRDB labels_2d JSON), OR
  $JRDB_DIR/labels_kitti/<seq>/<frame>.txt              (KITTI-style rows; JRDB toolkit converter)

The label parser auto-detects JSON vs KITTI-txt. Both are unit-tested in tests/test_robot_holdout.py.

Usage:
  uv run python scripts/robot_holdout.py selftest    # fabricate a tiny dataset, run the whole pipeline
  uv run python scripts/robot_holdout.py preview      # dump one frame with the safety zone drawn
  uv run python scripts/robot_holdout.py run          # full pass → docs/site/robot/{frames,trace,holdout}
"""
from __future__ import annotations

import json, os, sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
JRDB_DIR = Path(os.environ.get("JRDB_DIR", ROOT / "datasets" / "holdout" / "jrdb"))
SEQ = os.environ.get("JRDB_SEQ", "packard-poster-session-2019-03-20_0")
SITE = ROOT / "docs" / "site" / "robot"
FRAMES = SITE / "frames"

OUT_W = 960
DET_W = 1280           # panoramas are wide; people can be small
OUT_FPS = 7
CONF = 0.30
IOU_HIT = 0.4

# --- safety zone as NORMALISED polygon (0..1): the robot's near envelope, centre-bottom of view. ---
ZONE_N = [(0.34, 0.45), (0.66, 0.45), (0.82, 1.0), (0.18, 1.0)]

TIMELINE = [
    (0.0,  "healthy",  "Normal operation — cloud planner reachable, tight timeout"),
    (6.0,  "compute",  "Compute pressure — onboard detector runs late"),
    (12.0, "offline",  "Cloud-planner / remote-assist link lost"),
    (17.0, "healthy",  "Link restored"),
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

# ---- label parsing (both JRDB formats), returns frame_key -> list of [x0,y0,x1,y1] pedestrian boxes ----
def parse_jrdb_json(path: Path):
    """JRDB labels_2d JSON: {"labels": {"<img>.jpg": [{"box":[x,y,w,h], "label_id":"pedestrian:N", ...}]}}."""
    d = json.loads(Path(path).read_text())
    labels = d.get("labels", d)
    out = {}
    for img, boxes in labels.items():
        rows = []
        for b in boxes:
            lid = str(b.get("label_id", b.get("label", "pedestrian")))
            if "pedestrian" not in lid.lower() and "person" not in lid.lower():
                continue
            x, y, w, h = b["box"]
            rows.append([float(x), float(y), float(x) + float(w), float(y) + float(h)])
        out[img] = rows
    return out

def parse_kitti_txt(seq_dir: Path):
    """KITTI-style rows: type truncated occluded num_points alpha  bbox_l bbox_t bbox_r bbox_b  ..."""
    out = {}
    for f in sorted(Path(seq_dir).glob("*.txt")):
        rows = []
        for line in f.read_text().splitlines():
            p = line.split()
            if len(p) < 9 or p[0].lower() not in ("pedestrian", "person"):
                continue
            l, t, r, b = map(float, p[5:9])
            rows.append([l, t, r, b])
        out[f.stem] = rows
    return out

def load_labels():
    j = JRDB_DIR / "labels" / f"{SEQ}.json"
    if j.exists():
        return parse_jrdb_json(j), "json"
    k = JRDB_DIR / "labels_kitti" / SEQ
    if k.exists():
        return parse_kitti_txt(k), "kitti"
    return None, None

def image_dir():
    for cand in [JRDB_DIR/"images"/"image_stitched"/SEQ, JRDB_DIR/"images"/SEQ, JRDB_DIR/SEQ]:
        if cand.exists(): return cand
    return None

def cond_at(t):
    c = TIMELINE[0][1]
    for ts, cc, _ in TIMELINE:
        if t >= ts: c = cc
    return c

def _require_data():
    imgd = image_dir(); labels, kind = load_labels()
    if imgd is None or labels is None:
        print("JRDB data not found. Download JRDB (gated: https://jrdb.erc.monash.edu), then set:\n"
              f"  export JRDB_DIR=/path/to/jrdb   JRDB_SEQ={SEQ}\n"
              "expecting images/image_stitched/<seq>/*.jpg and labels/<seq>.json (or labels_kitti/<seq>/*.txt).\n"
              "Meanwhile: `uv run python scripts/robot_holdout.py selftest` proves the pipeline on a fixture.")
        raise SystemExit(1)
    return imgd, labels, kind

def _score(frames, labels_for, det_boxes_for, W0, H0):
    """Pure scoring core — shared by run() and selftest(). Returns holdout dict + trace frames."""
    import cv2  # noqa
    dw, dh = DET_W, int(round(H0 * DET_W / W0)); ow, oh = OUT_W, int(round(H0 * OUT_W / W0))
    tp = fp = gtp = 0; zone_gt = zone_ok = 0; frames_out = []
    for fr in frames:
        i = fr["i"]; t = i / OUT_FPS; cond = cond_at(t)
        g = labels_for(fr)                         # GT boxes in native pixels
        d = det_boxes_for(fr, dw, dh)              # detections in DET space
        g_d = [[b[0]*dw/W0, b[1]*dh/H0, b[2]*dw/W0, b[3]*dh/H0] for b in g]
        gtp += len(g_d); matched = set()
        for db in d:
            best = -1; bi = -1
            for k, gb in enumerate(g_d):
                if k in matched: continue
                v = iou(db, gb)
                if v > best: best = v; bi = k
            if best >= IOU_HIT: tp += 1; matched.add(bi)
            else: fp += 1
        gt_in = any(point_in_poly(((b[0]+b[2])/2/W0, b[3]/H0), ZONE_N) for b in g)
        det_in = any(point_in_poly((((b[0]+b[2])/2)/dw, b[3]/dh), ZONE_N) for b in d)
        if gt_in:
            zone_gt += 1
            if det_in: zone_ok += 1
        sx, sy = ow/dw, oh/dh
        frames_out.append({"i": i, "t": round(t, 3), "cond": cond, "mode": MODES[cond]["name"],
            "det": [[round(b[0]*sx), round(b[1]*sy), round(b[2]*sx), round(b[3]*sy)] for b in d],
            "gt": [[round(b[0]*ow/W0), round(b[1]*oh/H0), round(b[2]*ow/W0), round(b[3]*oh/H0)] for b in g],
            "gt_in": gt_in, "det_in": det_in})
    recall = tp/gtp if gtp else 0; prec = tp/(tp+fp) if (tp+fp) else 0
    return {"vru_detection_recall": round(recall, 3), "vru_detection_precision": round(prec, 3),
            "gt_vru_boxes": gtp, "zone_frames_with_vru": zone_gt,
            "zone_agreement": round(zone_ok/zone_gt, 3) if zone_gt else None}, frames_out, (ow, oh)

def run():
    import cv2
    from failsafe.workload.detector import make_detector
    imgd, labels, kind = _require_data()
    imgs = sorted(imgd.glob("*.jpg")) or sorted(imgd.glob("*.png"))
    im0 = cv2.imread(str(imgs[0])); H0, W0 = im0.shape[:2]
    det = make_detector("yolo", "yolov8n.pt", "cpu")
    FRAMES.mkdir(parents=True, exist_ok=True)
    cache = {}
    def labels_for(fr):
        key = fr["key"]; return labels.get(key, labels.get(Path(key).stem, []))
    def det_for(fr, dw, dh):
        img = cv2.imread(str(fr["path"])); df = cv2.resize(img, (dw, dh))
        return [[d.x0, d.y0, d.x1, d.y1] for d in det.detect(df, DET_W, CONF)]
    frames = [{"i": i, "key": p.name, "path": p} for i, p in enumerate(imgs)]
    # write web frames
    ow = OUT_W; oh = int(round(H0 * OUT_W / W0))
    for fr in frames:
        cv2.imwrite(str(FRAMES / f"f{fr['i']:05d}.jpg"), cv2.resize(cv2.imread(str(fr['path'])), (ow, oh)),
                    [cv2.IMWRITE_JPEG_QUALITY, 72])
        if fr["i"] % 40 == 0: print(f"  frame {fr['i']}/{len(frames)}", flush=True)
    metrics, frames_out, (ow, oh) = _score(frames, labels_for, det_for, W0, H0)
    zone_out = [[round(x*ow), round(y*oh)] for x, y in ZONE_N]
    holdout = {"domain": "robot (human–robot proximity safety)",
        "source": f"JRDB / {SEQ} ({kind} labels)", "license": "JRDB — research, gated",
        "frames": len(frames_out), "detector": f"YOLOv8n @{DET_W} CPU", "conf": CONF, "iou_hit": IOU_HIT,
        **metrics, "note": "Real robot-camera footage among people; humans-in-safety-zone events. "
        "Never used by the search. Not a certified safety system."}
    SITE.mkdir(parents=True, exist_ok=True)
    (SITE/"trace.json").write_text(json.dumps({"timeline": TIMELINE, "zone": zone_out, "w": ow, "h": oh,
        "modes": MODES, "frames": frames_out}, separators=(",", ":")))
    (SITE/"holdout.json").write_text(json.dumps(holdout, indent=2))
    print("\nHOLDOUT:", json.dumps(holdout, indent=2))

def preview():
    import cv2
    imgd, _, _ = _require_data()
    imgs = sorted(imgd.glob("*.jpg")) or sorted(imgd.glob("*.png"))
    im = cv2.imread(str(imgs[len(imgs)//2])); H0, W0 = im.shape[:2]
    ow = OUT_W; oh = int(round(H0*OUT_W/W0)); fr = cv2.resize(im, (ow, oh))
    poly = np.array([(int(x*ow), int(y*oh)) for x, y in ZONE_N], np.int32)
    ov = fr.copy(); cv2.fillPoly(ov, [poly], (40, 200, 240)); fr = cv2.addWeighted(ov, 0.2, fr, 0.8, 0)
    cv2.polylines(fr, [poly], True, (40, 200, 240), 2)
    SITE.mkdir(parents=True, exist_ok=True); cv2.imwrite(str(SITE/"_preview.jpg"), fr)
    print(f"{SEQ} {W0}x{H0}; wrote docs/site/robot/_preview.jpg — tune ZONE_N")

def selftest():
    """Prove the parser + scoring pipeline end-to-end on a fabricated fixture (no JRDB, no detector)."""
    W0, H0 = 1000, 500
    # fabricated JRDB JSON with two frames, one pedestrian each (one inside the zone, one outside)
    fixture = {"labels": {
        "000000.jpg": [{"box": [480, 300, 40, 120], "label_id": "pedestrian:1"}],   # centre → in zone
        "000001.jpg": [{"box": [30, 60, 30, 90], "label_id": "pedestrian:2"}],       # top-left → outside
    }}
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        jp = Path(td)/"seq.json"; jp.write_text(json.dumps(fixture))
        labels = parse_jrdb_json(jp)
        assert set(labels) == {"000000.jpg", "000001.jpg"}, labels
        assert labels["000000.jpg"][0] == [480, 300, 520, 420], labels["000000.jpg"]
        # KITTI-txt parser round-trip
        kd = Path(td)/"seq"; kd.mkdir()
        (kd/"000000.txt").write_text("Pedestrian 0 0 -1 -10 480 300 520 420 0 0 0 0 0 0 0\n"
                                     "Car 0 0 -1 -10 10 10 50 50 0 0 0 0 0 0 0\n")
        kl = parse_kitti_txt(kd)
        assert kl["000000"] == [[480, 300, 520, 420]], kl
        # scoring core with a "perfect detector" (returns GT scaled into DET space)
        frames = [{"i": i, "key": k} for i, k in enumerate(["000000.jpg", "000001.jpg"])]
        def labels_for(fr): return labels[fr["key"]]
        def det_for(fr, dw, dh):
            return [[b[0]*dw/W0, b[1]*dh/H0, b[2]*dw/W0, b[3]*dh/H0] for b in labels[fr["key"]]]
        m, fout, _ = _score(frames, labels_for, det_for, W0, H0)
    assert m["vru_detection_recall"] == 1.0 and m["vru_detection_precision"] == 1.0, m
    assert m["gt_vru_boxes"] == 2, m
    assert fout[0]["gt_in"] is True and fout[1]["gt_in"] is False, [f["gt_in"] for f in fout]
    assert m["zone_frames_with_vru"] == 1 and m["zone_agreement"] == 1.0, m
    print("SELFTEST OK — JRDB JSON + KITTI parsers, zone geometry, and scoring pipeline all pass.")
    print("Pipeline is code-complete; provide JRDB data (JRDB_DIR) and run `... robot_holdout.py run`.")

if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "selftest"
    {"selftest": selftest, "preview": preview, "run": run}.get(cmd, selftest)()
