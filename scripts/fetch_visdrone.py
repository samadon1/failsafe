"""Fetch one VisDrone MOT sequence for the drone holdout (scripts/drone_holdout.py).

Pulls a single sequence's frames + per-frame VRU annotations from the Voxel51/visdrone-mot Hugging
Face mirror (no login, HTTP) into datasets/holdout/visdrone/. Only the chosen sequence's frames are
downloaded, not the whole split. VisDrone is non-commercial research data.

  uv run python scripts/fetch_visdrone.py            # default sequence uav0000137_00458_v
  uv run python scripts/fetch_visdrone.py <scene_id> # a different sequence
"""
from __future__ import annotations

import json, shutil, sys
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO = "Voxel51/visdrone-mot"
VRU = {"pedestrian", "people"}
OUT = Path(__file__).resolve().parents[1] / "datasets" / "holdout" / "visdrone"

def main(scene: str = "uav0000137_00458_v"):
    frames_dir = OUT / "frames"; frames_dir.mkdir(parents=True, exist_ok=True)
    sp = hf_hub_download(REPO, "samples.json", repo_type="dataset")
    S = json.loads(Path(sp).read_text())["samples"]
    rows = sorted((s for s in S if s["scene_id"] == scene), key=lambda s: s["frame_number"])
    if not rows:
        raise SystemExit(f"scene {scene} not found; scenes: "
                         f"{sorted({s['scene_id'] for s in S})}")
    w = rows[0].get("metadata", {}).get("width"); h = rows[0].get("metadata", {}).get("height")
    print(f"{scene}: {len(rows)} frames, {w}x{h}")
    manifest = []
    for i, s in enumerate(rows):
        src = s["filepath"].split("/", 1)[1]
        fp = hf_hub_download(REPO, "data/" + src, repo_type="dataset")
        shutil.copy(fp, frames_dir / f"{i:05d}.jpg")
        vru = [{"l": d["label"], "bb": d["bounding_box"]} for d in s.get("detections", []) if d["label"] in VRU]
        manifest.append({"i": i, "fn": s["frame_number"], "vru": vru})
        if i % 50 == 0: print("  ", i, flush=True)
    (OUT / "scene.json").write_text(json.dumps({"scene": scene, "w": w, "h": h, "frames": manifest}))
    print(f"done: {len(manifest)} frames, {sum(len(m['vru']) for m in manifest)} VRU boxes → {OUT}")

if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "uav0000137_00458_v")
