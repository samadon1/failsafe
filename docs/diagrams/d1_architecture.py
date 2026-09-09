"""Diagram 1 — Failsafe core loop (hero). Icon-led / numbered / narrative style
(adopted from clean cloud-reference diagrams) with NVIDIA branding."""
from pathlib import Path

from kit import (AMBER, BLUE, GREEN, GREEN_D, INK, LINE, SLATE, SUB, SVG,
                 TINT_GREEN, TINT_SLATE, VIOLET, render)

W, H = 1460, 1120
d = SVG(W, H)

# ---- title + narrative intro ----
d.text(48, 52, "How Failsafe compiles and runs resilience", 25, INK, anchor="start", weight="800")
intro = ["Given an app and its mission invariants, Failsafe searches how the system should degrade when the cloud, network or compute fail,",
         "verifies every candidate with a deterministic evaluator, and compiles only the verified modes into a runtime policy. The runtime is a",
         "plain lookup — no model runs while the system is failing — and if no verified mode fits the moment, it fails closed and says so."]
for i, ln in enumerate(intro):
    d.text(48, 80 + i * 19, ln, 12.5, SUB, anchor="start")

# ---- operator + inputs (top band) ----
d.persona(96, 176, "Operator")
d.inode(200, 158, 300, 60, "target", "Mission invariants", "recall ≥ 95% · p95 ≤ 2 s · survive WAN loss", accent=SLATE)
d.inode(520, 158, 300, 60, "film", "App + labelled corpus", "restricted-zone monitor · 57 events", accent=SLATE)
# cloud external (top-right)
d.inode(1000, 158, 412, 60, "cloud", "Cloud confirmer", "Nemotron VL · Token Factory / NVIDIA Build", accent=VIOLET, wordmark=True)
d.arrow(140, 176, 200, 176)

# ============================ DISCOVERY panel ============================
dx, dy, dw, dh = 48, 250, W - 96, 210
d.panel(dx, dy, dw, dh, "DISCOVERY", tint=TINT_GREEN, title_color=GREEN_D,
        subtitle="offline · AI-assisted · run once per mission")
ny = dy + 70
nh = 84
nw = 236
gap = (dw - 60 - 5 * nw) / 4
xs = [dx + 30 + i * (nw + gap) for i in range(5)]
d.inode(xs[0], ny, nw, nh, "compass", "Search strategies", "grid · random · greedy · Nemotron", accent=GREEN, step=1)
d.inode(xs[1], ny, nw, nh, "zap", "Fault injector", "cloud · bandwidth · compute", accent=AMBER, step=2)
d.inode(xs[2], ny, nw, nh, "activity", "Experiment runner", "real-time replay · measured", accent=BLUE, step=3)
d.inode(xs[3], ny, nw, nh, "shield", "Deterministic evaluator", "invariants → PASS / FAIL · no LLM", accent=GREEN, hero=True, step=4)
d.inode(xs[4], ny, nw, nh, "db", "resilience-policy.yaml", "verified modes + NO-VERIFIED", accent=GREEN, step=5)
for i in range(4):
    d.arrow(xs[i] + nw, ny + nh / 2, xs[i + 1], ny + nh / 2, gap=5)
# FAIL feedback (dashed amber, below the row)
fy = ny + nh + 24
d.el.append(f'<polyline points="{xs[3]+nw/2},{ny+nh} {xs[3]+nw/2},{fy} {xs[0]+nw/2},{fy} {xs[0]+nw/2},{ny+nh}" '
            f'fill="none" stroke="{AMBER}" stroke-width="2" stroke-dasharray="5 5" marker-end="url(#aha)"/>')
d.text((xs[0] + xs[3] + nw) / 2, fy + 15, "FAIL  →  propose another configuration", 11.5, AMBER, weight="700")
# cloud confirmer feeds the runner (dashed violet elbow, labelled)
_cc = 1206  # cloud node center-x
d.el.append(f'<polyline points="{_cc},218 {_cc},234 {xs[2]+nw/2},234 {xs[2]+nw/2},{ny-5}" fill="none" stroke="{VIOLET}" stroke-width="1.8" stroke-dasharray="4 4" marker-end="url(#ah)"/>')
d.text(xs[2]+nw/2+10, 230, "cloud calls", 10.5, VIOLET, anchor="start", weight="600")

# ============================ deploy ============================
yaml_cx = xs[4] + nw / 2
by = 560
d.arrow(yaml_cx, dy + dh, yaml_cx, by, color=GREEN_D, dash="6 5", sw=2.6, head="ahg")
d.text(yaml_cx + 14, (dy + dh + by) / 2 + 4, "deploy", 12.5, GREEN_D, anchor="start", weight="800")

# ============================ RUNTIME panel ============================
rx0, rw, rh = 48, W - 96, 200
d.panel(rx0, by, rw, rh, "RUNTIME", tint=TINT_SLATE, title_color=SLATE,
        subtitle="on-device · deterministic · no LLM · fails closed")
ry = by + 70
rw_n = 250
d.inode(rx0 + 30, ry, rw_n, 84, "radio", "Observe", "cloud RTT · reachability · slowdown", accent=BLUE, step=6)
d.inode(rx0 + 30 + rw_n + 48, ry, 210, 84, "branch", "Classify", "→ operating condition", accent=SLATE)
lx = rx0 + 30 + rw_n + 48 + 210 + 48
d.inode(lx, ry, 230, 84, "search", "Policy lookup", "match a verified mode", accent=SLATE)
# outcomes stacked
ox = lx + 230 + 60
ow = rx0 + rw - 30 - ox
d.inode(ox, ry, ow, 40, "check", "Verified mode → reconfigure", None, accent=GREEN)
d.inode(ox, ry + 48, ow, 40, "alert", "NO VERIFIED MODE → fail closed + escalate", None, accent=AMBER)
d.arrow(rx0 + 30 + rw_n, ry + 42, rx0 + 30 + rw_n + 48, ry + 42, gap=5)
d.arrow(rx0 + 30 + rw_n + 48 + 210, ry + 42, lx, ry + 42, gap=5)
# fork: lookup → both outcomes
_fk = lx + 230 + 26
d.el.append(f'<polyline points="{lx+230+5},{ry+42} {_fk},{ry+42} {_fk},{ry+20}" fill="none" stroke="{LINE}" stroke-width="2.2"/>')
d.el.append(f'<polyline points="{_fk},{ry+42} {_fk},{ry+68}" fill="none" stroke="{LINE}" stroke-width="2.2"/>')
d.arrow(_fk, ry+20, ox, ry+20, gap=2)
d.arrow(_fk, ry+68, ox, ry+68, gap=2, color=AMBER, head="aha")
d.badge((_fk+ox)/2, ry+68, 7, AMBER)

# ============================ mono caption ============================
my = by + rh + 26
mh = d.mono(48, my, 640, [
    "modes:",
    "  island:            # verified for wan_offline",
    "    cloud_confirmation: false     recall: 0.965  p95_ms: 970",
    "  zombie-cloud:      # verified for wan_zombie",
    "    cloud_timeout_ms: 1000        recall: 0.982  p95_ms: 1583",
    "unverified: [compute_severe]      # → fail closed",
], title="resilience-policy.yaml  (excerpt, compiled from measured runs)")

# ============================ numbered legend ============================
ly = my
d.text(720, ly + 14, "How the loop runs", 14, INK, anchor="start", weight="800")
d.legend(720, ly + 30, W - 720 - 48, [
    (1, "Strategies propose candidate operating configurations from a bounded space.", SUB),
    (2, "Each is run under injected faults — slow/zombie/offline cloud, bandwidth, compute.", SUB),
    (3, "The runner replays the corpus in real time and measures recall, latency, drops, CPU.", SUB),
    (4, "The deterministic evaluator checks the measured metrics against the invariants.", SUB),
    (5, "Verified modes are compiled into the policy, each linked to its experiment.", SUB),
    (6, "At runtime the device observes conditions and looks up a verified mode.", SUB),
    (7, "If none fits, it fails closed — NO VERIFIED MODE — and escalates. Never a guess.", AMBER),
], cols=1)

# footnote
d.text(48, H - 26, "NVIDIA Nemotron via Nebius Token Factory / NVIDIA Build; NVIDIA marks used only to identify sponsor technology.",
       11, "#9aa2ab", anchor="start")

render(d.out(), "d1_architecture", Path(__file__).parent, W, H)
print("d1 done")
