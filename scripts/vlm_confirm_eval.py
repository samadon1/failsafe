"""Qualitative check: does the real NVIDIA VLM confirmer agree with the geometric zone rule on real
footage? (D-049)

Reuses the release demo's trace for the warehouse holdout (docs/site/trace.json): it already carries
the real YOLOv8n detections and the monitored-zone polygon in the same coordinates as the frames in
docs/site/frames/. For a balanced sample of detected people (some with a foot inside the zone by the
geometric rule, some outside), we draw the zone + that person's box on the frame, ask the VLM whether
the feet are inside, and compare its verdict to the geometric rule. Every call is cached
(artifacts/vlm_cache), so this is re-runnable and deterministic.

  uv run python scripts/vlm_confirm_eval.py            # 30 cases, balanced
  uv run python scripts/vlm_confirm_eval.py 40 cctv
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

from failsafe.corpus.scene import point_in_convex_polygon
from failsafe.workload.cloud import ConfirmationRequest
from failsafe.workload.vlm_confirmer import VlmConfirmer
from failsafe.workload.zone import Detection

ROOT = Path(__file__).resolve().parents[1]
SCENES = {"cctv": ROOT / "docs" / "site"}  # warehouse holdout; others could be added
N = int(sys.argv[1]) if len(sys.argv) > 1 else 30
SCENE = sys.argv[2] if len(sys.argv) > 2 else "cctv"


def foot_in_zone(box, zone) -> bool:
    x0, y0, x1, y1 = box
    return point_in_convex_polygon(((x0 + x1) / 2.0, y1), zone)


def main() -> None:
    d = SCENES[SCENE]
    tr = json.loads((d / "trace.json").read_text())
    zone = [(float(x), float(y)) for x, y in tr["zone"]]
    frames_dir = d / "frames"

    # collect (frame_index, box, geometric_truth) for every detected person
    cases = []
    for i, f in enumerate(tr["frames"]):
        for box in f.get("det", []):
            cases.append((i, box, foot_in_zone(box, zone)))
    inside = [c for c in cases if c[2]]
    outside = [c for c in cases if not c[2]]
    # balance and spread across frames
    def spread(xs, k):
        return xs[:: max(1, len(xs) // k)][:k] if xs else []
    half = N // 2
    sample = spread(inside, half) + spread(outside, N - half)
    print(f"{SCENE}: {len(cases)} detected people across {len(tr['frames'])} frames "
          f"({len(inside)} inside / {len(outside)} outside by the geometric rule); testing {len(sample)}")

    vlm = VlmConfirmer(cache_dir=ROOT / "artifacts" / "vlm_cache")
    print(f"model: {vlm.name}\n")
    rows, latencies, agree = [], [], 0
    for n, (i, box, truth) in enumerate(sample, 1):
        jpeg = (frames_dir / f"f{i:05d}.jpg").read_bytes()
        cand = Detection(float(box[0]), float(box[1]), float(box[2]), float(box[3]), 0.9)
        req = ConfirmationRequest(camera=SCENE, scene_t=float(tr["frames"][i]["t"]), jpeg=jpeg, candidate=cand, zone=zone)
        resp, _ = vlm.confirm(req)
        ok = resp.confirmed == truth
        agree += ok
        rows.append({"frame": i, "geometric": truth, "vlm": resp.confirmed, "confidence": resp.confidence,
                     "agree": ok, "detail": resp.detail})
        # cached rows have no fresh latency; only count live ones
        rec = (ROOT / "artifacts" / "vlm_cache" / f"{vlm._key(__import__('failsafe.workload.vlm_confirmer', fromlist=['_overlay'])._overlay(jpeg, cand, zone))}.json")
        if rec.exists():
            lm = json.loads(rec.read_text()).get("latency_ms")
            if lm:
                latencies.append(lm)
        mark = "ok " if ok else "XX "
        print(f"  {mark}[{n:2}/{len(sample)}] frame {i:3}  geometric={'IN ' if truth else 'OUT'}  vlm={'IN ' if resp.confirmed else 'OUT'} ({resp.confidence:.2f})")

    rate = agree / len(sample) if sample else 0.0
    tp = sum(1 for r in rows if r["geometric"] and r["vlm"]); tn = sum(1 for r in rows if not r["geometric"] and not r["vlm"])
    fp = sum(1 for r in rows if not r["geometric"] and r["vlm"]); fn = sum(1 for r in rows if r["geometric"] and not r["vlm"])
    out = {"scene": SCENE, "model": vlm.name, "n": len(sample), "agreement": round(rate, 3),
           "confusion": {"both_in": tp, "both_out": tn, "vlm_in_geo_out": fp, "vlm_out_geo_in": fn},
           "median_latency_ms": round(statistics.median(latencies)) if latencies else None,
           "rows": rows}
    (ROOT / "artifacts" / "vlm_eval.json").write_text(json.dumps(out, indent=1))
    print(f"\nagreement with the geometric zone rule: {agree}/{len(sample)} = {rate:.0%}")
    print(f"  both say inside {tp} · both say outside {tn} · VLM-in/geo-out {fp} · VLM-out/geo-in {fn}")
    if latencies:
        print(f"  live call latency: median {statistics.median(latencies):.0f} ms over {len(latencies)} fresh calls")
    print("  wrote artifacts/vlm_eval.json")


if __name__ == "__main__":
    main()
