"""Per-alert latency distribution for the 'slow != dead' finding.

Each dot is one real alert (measured, real-time replay), for the SAME cloud-confirmed configuration,
grouped by cloud condition. A dead (offline) cloud produces fast alerts; a slow-but-alive (zombie)
cloud pushes every alert past the 2-second limit. Renders docs/diagrams/c6_latency_dist.png.
"""
from __future__ import annotations

import glob, json, statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "diagrams"
INK = "#1a2430"; SUB = "#5b6670"; MUT = "#95a1ae"; GREEN = "#76b900"; GREEN_D = "#4e7a00"; RED = "#cf3b31"; GRID = "#e6eaee"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
                     "text.color": INK, "figure.dpi": 150})

def cfg_match(c):
    return (c["cloud_confirmation"] and c["cloud_timeout_ms"] == 3000 and c["historical_indexing"]
            and c["critical_fps"] == 15 and c["background_fps"] == 2 and c["detector_resolution"] == 640)

# scenario -> plain label (top to bottom)
ROWS = [("wan_offline", "Cloud dead\n(offline)"), ("wan_severely_slow", "Cloud very slow"),
        ("wan_zombie", "Cloud slow\nbut alive")]

def collect():
    by = {}
    for p in glob.glob(str(ROOT / "artifacts" / "experiments" / "**" / "*.json"), recursive=True):
        try: d = json.load(open(p))
        except Exception: continue
        e = d["experiment"]; sc = e["scenario"]["name"]
        if not cfg_match(e["config"]): continue
        if (d.get("metrics", {}).get("qc_suspect_contention", {}) or {}).get("value"): continue
        lats = [a["latency_ms"] for a in d.get("alerts", []) if isinstance(a, dict) and a.get("latency_ms") is not None]
        if lats: by.setdefault(sc, []).extend(lats)
    return by

def main():
    by = collect()
    fig, ax = plt.subplots(figsize=(11.6, 5.0))
    rng = np.random.default_rng(0)
    yt, yl = [], []
    for i, (sc, label) in enumerate(ROWS):
        y0 = len(ROWS) - 1 - i
        lat = np.array(by.get(sc, [])) / 1000.0
        if len(lat) == 0: continue
        jitter = (rng.random(len(lat)) - 0.5) * 0.62
        colors = [GREEN if v <= 2 else RED for v in lat]
        ax.scatter(lat, np.full(len(lat), y0) + jitter, s=46, c=colors, alpha=0.55, edgecolors="none", zorder=3)
        med = statistics.median(lat)
        ax.plot([med, med], [y0 - 0.36, y0 + 0.36], color=INK, lw=2.4, zorder=4)
        ax.text(med, y0 + 0.46, f"median {med:.1f} s", ha="center", fontsize=10.5, fontweight="bold", color=INK)
        yt.append(y0); yl.append(label + f"\nn={len(lat)}")
    ax.axvline(2.0, color=INK, lw=1.6, ls=(0, (5, 3)), zorder=2)
    ax.text(2.0, len(ROWS) - 0.35, " 2-second limit", ha="left", va="center", fontsize=11, fontweight="bold", color=INK)
    # zones
    ax.axvspan(0, 2, color=GREEN, alpha=0.05, zorder=0); ax.axvspan(2, 100, color=RED, alpha=0.05, zorder=0)
    ax.set_yticks(yt); ax.set_yticklabels(yl, fontsize=12)
    ax.set_ylim(-0.7, len(ROWS) - 0.15); ax.set_xlim(0, max(6.2, max(max(v) for v in by.values()) / 1000 * 1.05))
    ax.set_xlabel("alert latency (seconds)  ·  each dot is one real alert", fontsize=12, color=SUB)
    ax.grid(axis="x", color=GRID, lw=1); ax.set_axisbelow(True)
    for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
    ax.tick_params(left=False)
    ax.text(0.5, -0.5, "safe", color=GREEN_D, fontsize=11, fontweight="bold")
    ax.text(4.2, -0.5, "too late, the person is missed", color=RED, fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(OUT / "c6_latency_dist.png", bbox_inches="tight", facecolor="white")
    print("wrote docs/diagrams/c6_latency_dist.png")
    for sc, label in ROWS:
        v = by.get(sc, [])
        if v: print(f"  {sc}: n={len(v)} median={statistics.median(v):.0f}ms  <2s:{sum(x<=2000 for x in v)}/{len(v)}")

if __name__ == "__main__":
    main()
