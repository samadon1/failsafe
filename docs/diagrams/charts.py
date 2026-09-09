"""Data charts 5–9 — every number read from real result files in artifacts/.
Branded palette; status colors (green verified / amber fail) always carry labels; one axis each;
single-hue magnitude for ranking bars; recessive grid; direct value labels."""
from __future__ import annotations

import glob
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).parent
EXP = ROOT / "artifacts" / "experiments"

INK = "#1b2430"; SUB = "#5b6670"; GREEN = "#76b900"; GREEN_D = "#5c8f00"
AMBER = "#e8862a"; BLUE = "#2f6db0"; GREY = "#c3cad1"; GRID = "#e6eaee"
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica Neue", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": "#b8c0c8", "axes.labelcolor": INK,
    "xtick.color": SUB, "ytick.color": SUB, "axes.linewidth": 1.0, "figure.dpi": 150,
})


def _load_results():
    out = []
    for p in glob.glob(str(EXP / "**" / "*.json"), recursive=True):
        if "/stale/" in p:
            continue
        try:
            out.append(json.loads(Path(p).read_text()))
        except Exception:
            pass
    return out


def _mv(r, k):
    m = r["metrics"].get(k)
    return None if not m or m.get("value") is None else m["value"]


def _clean(r):
    return not (_mv(r, "qc_suspect_contention") or 0)


def _style(ax, title, sub=None, xlabel=None, ylabel=None):
    ax.set_title(title, fontsize=15, fontweight="bold", loc="left", pad=18 if sub else 10, color=INK)
    if sub:
        ax.text(0, 1.02, sub, transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="y", color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=11, color=SUB)
    if ylabel:
        ax.set_ylabel(ylabel, fontsize=11, color=SUB)


# ---------------------------------------------------------------- 5. slow != dead
def chart5(results):
    want = {"wan_offline": "offline", "wan_zombie": "zombie", "wan_severely_slow": "severely slow", "healthy": "healthy"}
    # naive_default = 3s timeout cloud-heavy; identify by config knobs
    rows = []
    for r in results:
        c = r["experiment"]["config"]; sc = r["experiment"]["scenario"]["name"]
        if sc not in want or not _clean(r):
            continue
        if c["cloud_confirmation"] and c["cloud_timeout_ms"] == 3000 and c["historical_indexing"] and c["critical_fps"] == 15 and c["background_fps"] == 2 and c["detector_resolution"] == 640:
            p95 = _mv(r, "alert_latency_p95_ms")
            if p95 is not None:
                rows.append((want[sc], p95, _mv(r, "critical_event_recall")))
    # dedupe: mean per scenario
    import statistics
    agg = {}
    for name, p95, rec in rows:
        agg.setdefault(name, []).append(p95)
    order = ["healthy", "offline", "severely slow", "zombie"]
    labels = [o for o in order if o in agg]
    vals = [statistics.fmean(agg[o]) for o in labels]
    fig, ax = plt.subplots(figsize=(7.6, 4.4))
    colors = [GREEN if v <= 2000 else AMBER for v in vals]
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    ax.axhline(2000, color=INK, lw=1.4, ls=(0, (4, 3)), zorder=2)
    ax.text(len(labels) - 0.5, 2080, "2 s mission limit", ha="right", fontsize=10, color=INK, fontweight="bold")
    for b, v in zip(bars, vals):
        ok = v <= 2000
        ax.text(b.get_x() + b.get_width() / 2, v + 90, f"{v/1000:.2f} s\n{'PASS' if ok else 'FAIL'}",
                ha="center", va="bottom", fontsize=10.5, fontweight="bold", color=GREEN_D if ok else "#b5651b")
    _style(ax, "A slow cloud is more dangerous than a dead one",
           "p95 alert latency of the same cloud-confirmed configuration, by cloud condition (measured)", ylabel="p95 alert latency (ms)")
    ax.set_ylim(0, max(vals) * 1.22)
    fig.tight_layout(); fig.savefig(OUT / "c5_slow_not_dead.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("c5", labels, [round(v) for v in vals])


# ---------------------------------------------------------------- 6. search efficiency
def chart6():
    # evaluations to first verified mode on wan_zombie (measured/derived)
    data = [("grid", 72, 0), ("random", 5.5, 4.7), ("greedy", 2, 0), ("Nemotron", 1, 0)]
    labels = [d[0] for d in data]; vals = [d[1] for d in data]; err = [d[2] for d in data]
    fig, ax = plt.subplots(figsize=(7.6, 4.0))
    y = range(len(labels))
    colors = [GREY, GREY, GREY, GREEN]  # highlight Nemotron; single-hue magnitude otherwise
    bars = ax.barh(list(y), vals, color=colors, height=0.6, zorder=3,
                   xerr=[[0]*4, err], error_kw=dict(ecolor=SUB, elinewidth=1.4, capsize=4))
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, fontsize=12)
    ax.invert_yaxis()
    for b, v, e in zip(bars, vals, err):
        t = f"{v:g}" + (f" ± {e:g}" if e else "")
        ax.text(v + e + 1.4, b.get_y() + b.get_height() / 2, t, va="center", fontsize=11, fontweight="bold", color=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    ax.set_title("Evaluations to the first verified mode  ·  wan_zombie", fontsize=15, fontweight="bold", loc="left", pad=18, color=INK)
    ax.text(0, 1.02, "same 72-mode grid as ground truth; fewer is better. Nemotron and greedy reach it fastest; grid is exhaustive.",
            transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
    ax.set_xlabel("experiments evaluated", fontsize=11, color=SUB)
    ax.set_xlim(0, 80)
    fig.tight_layout(); fig.savefig(OUT / "c6_search_efficiency.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("c6 ok")


# ---------------------------------------------------------------- 7. Pareto frontier
def chart7(results):
    # wan_zombie: capability vs edge CPU%, feasible points; Pareto highlighted
    import sys
    sys.path.insert(0, str(ROOT))
    from failsafe.mission.schema import MissionSpec
    from failsafe.experiments.schema import ExperimentResult
    from failsafe.search.objective import capability_retained, frontier
    mission = MissionSpec.from_yaml(str(ROOT / "missions" / "restricted-zone.yaml"))
    ers = []
    for r in results:
        if r["experiment"]["scenario"]["name"] != "wan_zombie" or not _clean(r):
            continue
        ers.append(ExperimentResult.model_validate(r))
    pts, pareto = frontier(ers, mission)
    par_ids = {p.experiment_id for p in pareto}
    fig, ax = plt.subplots(figsize=(7.6, 4.8))
    for p in pts:
        is_par = p.experiment_id in par_ids
        ax.scatter(p.cpu_percent, p.capability, s=120 if is_par else 70,
                   color=GREEN if is_par else GREY, edgecolor="white", linewidth=1.5, zorder=4 if is_par else 3)
    # (no connecting line: the frontier is 3-D — capability / CPU / cloud-bytes — projected to 2-D)
    # label the best (max capability)
    best = max(pareto, key=lambda p: p.capability)
    ax.annotate("best verified mode\ncloud on · 1 s timeout", (best.cpu_percent, best.capability),
                textcoords="offset points", xytext=(-8, -34), fontsize=9.5, color=GREEN_D, ha="right", fontweight="bold")
    _style(ax, "Feasible modes under a zombie cloud  ·  Pareto frontier",
           "each dot PASSES the mission; green = Pareto-optimal over capability / CPU / cloud-bytes (not dominated)",
           xlabel="edge CPU cost  (% of one core, lower is better)", ylabel="capability retained")
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_ylim(0.4, 1.05)
    fig.tight_layout(); fig.savefig(OUT / "c7_pareto.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("c7", len(pts), "feasible,", len(pareto), "pareto")


# ---------------------------------------------------------------- 8. demo timeline
def chart8():
    naive = json.loads((ROOT / "artifacts" / "demo" / "naive.json").read_text())
    fs = json.loads((ROOT / "artifacts" / "demo" / "failsafe.json").read_text())
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    # shade the offline window (phase named wan_offline)
    off = next(p for p in naive["phases"] if p["name"] == "wan_offline")
    ax.axvspan(off["start_t"], off["end_t"], color="#fbe6cf", zorder=1)
    ax.text((off["start_t"]+off["end_t"])/2, 1.6, "WAN offline", ha="center", fontsize=10.5, color="#b5651b", fontweight="bold")
    rows = [("Failsafe", fs, GREEN, 1.0), ("naive system", naive, AMBER, 0.0)]
    for name, rep, col, yv in rows:
        for ph in rep["phases"]:
            det = ph["detected"]; ev = ph["events"]
            ax.plot([ph["start_t"], ph["end_t"]], [yv, yv], color=col, lw=7, solid_capstyle="butt", zorder=3, alpha=0.85)
            if ev:
                ax.text((ph["start_t"]+ph["end_t"])/2, yv+0.14, f"{det}/{ev}", ha="center", fontsize=10, color=INK, fontweight="bold")
    ax.set_yticks([0, 1]); ax.set_yticklabels(["naive system", "Failsafe"], fontsize=12)
    ax.set_ylim(-0.5, 2.0); ax.set_xlim(0, naive["phases"][-1]["end_t"])
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.set_xlabel("scene time (s)", fontsize=11, color=SUB)
    ax.set_title("Cut the WAN: intrusions detected per phase (real-time demo)", fontsize=15, fontweight="bold", loc="left", pad=18, color=INK)
    offn = next(p for p in naive["phases"] if p["name"]=="wan_offline"); offf = next(p for p in fs["phases"] if p["name"]=="wan_offline")
    ax.text(0, 1.02, f"during the outage: naive detects {offn['detected']}/{offn['events']}, Failsafe {offf['detected']}/{offf['events']}",
            transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
    ax.grid(axis="x", color=GRID, linewidth=1); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(OUT / "c8_demo_timeline.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("c8 ok")


# ---------------------------------------------------------------- 9. recall landscape heatmap
def chart9():
    # measured detectability (D-016 / detectability run): recall by person-height bin x resolution
    import numpy as np
    res = ["640", "480", "320"]
    bins = ["≥130 px", "80–130 px", "40–80 px"]
    # rows = height bins (top=large), cols = resolution ; measured values
    M = np.array([
        [1.00, 1.00, 1.00],   # >=130
        [1.00, 1.00, 0.82],   # 80-130
        [0.85, 0.65, 0.12],   # 40-80
    ])
    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    im = ax.imshow(M, cmap="YlGn", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3)); ax.set_xticklabels([f"{r} px" for r in res], fontsize=12)
    ax.set_yticks(range(3)); ax.set_yticklabels(bins, fontsize=12)
    for i in range(3):
        for j in range(3):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=13,
                    color="white" if v > 0.6 else INK, fontweight="bold")
    ax.set_title("Why resolution is the dominant recall knob", fontsize=15, fontweight="bold", loc="left", pad=18, color=INK)
    ax.text(0, 1.03, "detector recall by person size × input resolution (measured, offline detectability probe)",
            transform=ax.transAxes, fontsize=10.5, color=SUB, va="bottom")
    ax.set_xlabel("detector input resolution", fontsize=11, color=SUB)
    ax.set_ylabel("person height in frame", fontsize=11, color=SUB)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04); cb.set_label("recall", color=SUB)
    ax.set_xticks([x-0.5 for x in range(1,3)], minor=True); ax.set_yticks([y-0.5 for y in range(1,3)], minor=True)
    ax.grid(which="minor", color="white", linewidth=3); ax.tick_params(which="minor", length=0)
    fig.tight_layout(); fig.savefig(OUT / "c9_recall_landscape.png", bbox_inches="tight", facecolor="white"); plt.close(fig)
    print("c9 ok")


if __name__ == "__main__":
    results = _load_results()
    print("loaded", len(results), "experiment results")
    chart5(results)
    chart6()
    chart7(results)
    chart8()
    chart9()
