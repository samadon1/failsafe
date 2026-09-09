"""Multi-seed generalization: does each headline finding hold on seeds 1/2/3?
Reads artifacts/experiments (clean runs only), prints a markdown table + writes c10_generalization.png."""
from __future__ import annotations

import glob, json, statistics
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
EXP = ROOT / "artifacts" / "experiments"
OUT = Path(__file__).parent
INK="#1b2430"; SUB="#5b6670"; GREEN="#76b900"; GREEN_D="#5c8f00"; AMBER="#e8862a"; BLUE="#2f6db0"; GRID="#e6eaee"
plt.rcParams.update({"font.family":"sans-serif","font.sans-serif":["Helvetica Neue","Arial","DejaVu Sans"],
    "text.color":INK,"axes.edgecolor":"#b8c0c8","xtick.color":SUB,"ytick.color":SUB,"figure.dpi":150})

def load():
    out=[]
    for p in glob.glob(str(EXP/"**"/"*.json"),recursive=True):
        if "/stale/" in p: continue
        try: out.append(json.loads(Path(p).read_text()))
        except: pass
    return out
def mv(r,k):
    m=r["metrics"].get(k); return None if not m or m.get("value") is None else m["value"]
def clean(r): return not (mv(r,"qc_suspect_contention") or 0)

def match(r, *, cf=None,bg=None,res=None,cloud=None,to=None,idx=None):
    c=r["experiment"]["config"]
    if cf is not None and c["critical_fps"]!=cf: return False
    if bg is not None and c["background_fps"]!=bg: return False
    if res is not None and c["detector_resolution"]!=res: return False
    if cloud is not None and c["cloud_confirmation"]!=cloud: return False
    if to is not None and c["cloud_timeout_ms"]!=to: return False
    if idx is not None and c["historical_indexing"]!=idx: return False
    return True

def val(results, scenario, metric, seeds=(1,2,3), **knobs):
    by=defaultdict(list)
    for r in results:
        e=r["experiment"]
        if e["scenario"]["name"]!=scenario or e["corpus"]["kind"]!="synthetic": continue
        if not clean(r) or not match(r,**knobs): continue
        v=mv(r,metric)
        if v is not None: by[e["corpus"]["seed"]].append(v)
    return {s:(statistics.fmean(by[s]) if by.get(s) else None) for s in seeds}

def fmt(d,scale=1,unit=""):
    return " · ".join(f"s{s}: {('—' if v is None else f'{v/scale:.3f}{unit}')}" for s,v in d.items())

if __name__=="__main__":
    R=load(); seeds=(1,2,3)
    print("loaded",len(R),"results")
    rows=[]
    # 3.1 slow != dead: naive_default (3s) p95 offline vs zombie
    off = val(R,"wan_offline","alert_latency_p95_ms",res=640,cloud=True)
    zom = val(R,"wan_zombie","alert_latency_p95_ms",cf=15,bg=2,res=640,cloud=True,to=3000,idx=True)
    rows.append(("Slow ≠ dead — offline p95 (ms)", off, "≤2000 PASS"))
    rows.append(("Slow ≠ dead — zombie p95 (ms)", zom, ">2000 FAIL"))
    # 3.2 timeout fix: normal (1s) zombie p95
    fix = val(R,"wan_zombie","alert_latency_p95_ms",cf=15,bg=2,res=640,cloud=True,to=1000,idx=True)
    rows.append(("Timeout fix — normal zombie p95 (ms)", fix, "≤2000 PASS"))
    # 3.3 resolution knob: island recall 640 vs 320 healthy
    r640=val(R,"healthy","critical_event_recall",cf=15,bg=1,res=640,cloud=False,idx=False)
    r320=val(R,"healthy","critical_event_recall",cf=15,bg=1,res=320,cloud=False,idx=False)
    rows.append(("Resolution — island recall @640", r640, "≥0.95"))
    rows.append(("Resolution — island recall @320", r320, "<0.95"))
    # 3.5 no verified: island recall compute_severe
    cs=val(R,"compute_severe","critical_event_recall",cf=15,bg=1,res=640,cloud=False,idx=False)
    rows.append(("No-verified — island recall (compute)", cs, "<0.95"))

    print("\n## 5. Generalization across seeds\n")
    print("| finding (metric) | per-seed values | expected |")
    print("|---|---|---|")
    for name,d,exp in rows:
        scale = 1000 if "p95" in name else 1
        # format inline
        cells=" · ".join(f"s{s}: {'—' if v is None else (f'{v:.0f}' if 'p95' in name else f'{v:.3f}')}" for s,v in d.items())
        print(f"| {name} | {cells} | {exp} |")

    # chart: slow!=dead p95 across seeds (grouped) + resolution recall across seeds
    fig,(a,b)=plt.subplots(1,2,figsize=(11,4.2))
    xs=list(seeds); import numpy as np
    def bars(ax,series,title,ylab,thr=None,thrlab=""):
        w=0.38; x=np.arange(len(xs))
        for i,(lab,d,col) in enumerate(series):
            vals=[d.get(s) or 0 for s in xs]
            ax.bar(x+(i-(len(series)-1)/2)*w, vals, w, label=lab, color=col, zorder=3)
        if thr: ax.axhline(thr,color=INK,lw=1.3,ls=(0,(4,3))); ax.text(len(xs)-1.4,thr*1.02,thrlab,fontsize=9,fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels([f"seed {s}" for s in xs]); ax.set_title(title,fontsize=13,fontweight="bold",loc="left")
        ax.set_ylabel(ylab,fontsize=10,color=SUB); ax.grid(axis="y",color=GRID); ax.set_axisbelow(True)
        for s in ("top","right"): ax.spines[s].set_visible(False)
        ax.legend(fontsize=9,frameon=False)
    bars(a,[("offline",off,GREEN),("zombie",zom,AMBER)],"Slow ≠ dead: p95 latency","p95 (ms)",2000,"2 s limit")
    bars(b,[("640 px",r640,GREEN),("320 px",r320,AMBER)],"Resolution knob: island recall","recall",0.95,"0.95")
    b.set_ylim(0,1.05)
    fig.suptitle("Findings hold across independently seeded corpora", fontsize=15, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=(0,0,1,0.95)); fig.savefig(OUT/"c10_generalization.png",bbox_inches="tight",facecolor="white")
    print("\nwrote c10_generalization.png")
