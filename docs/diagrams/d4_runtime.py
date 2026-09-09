"""Diagram 4 — runtime state machine, icon-led."""
from pathlib import Path

from kit import (AMBER, BLUE, BORD, CARD, GREEN, GREEN_D, INK, SLATE, SUB, SVG,
                 TINT_AMBER, TINT_GREEN, TINT_SLATE, VIOLET, render)

W, H = 1300, 620
d = SVG(W, H)
d.text(48, 50, "The runtime state machine", 25, INK, anchor="start", weight="800")
for i, ln in enumerate([
    "On the device, a plain deterministic loop moves between verified modes as conditions change — degrading fast, recovering",
    "slowly, and never activating anything that was not verified for the current condition."]):
    d.text(48, 78 + i * 19, ln, 12.5, SUB, anchor="start")

def state(cx, cy, name, sub, icon, accent=GREEN, w=210, h=78):
    d.rect(cx - w/2, cy - h/2, w, h, CARD, rx=14, stroke=accent, sw=2.2, shadow=True)
    d.rect(cx - w/2 + 14, cy - 19, 38, 38, d._tint(accent), rx=10)
    d.icon(cx - w/2 + 21, cy - 12, 24, icon, accent, sw=2.0)
    d.text(cx - w/2 + 62, cy - 3, name, 15, INK, anchor="start", weight="800")
    d.text(cx - w/2 + 62, cy + 16, sub, 10.8, SUB, anchor="start")

ys = 230
cxs = [220, 520, 820]
state(cxs[0], ys, "NORMAL", "cloud on · 1 s timeout", "shield", GREEN)
state(cxs[1], ys, "ISLAND", "local only · bg 1 fps", "cpu", GREEN)
state(cxs[2], ys, "SURVIVAL", "critical cam · 5 fps", "activity", GREEN)
state(1080, ys, "NO VERIFIED MODE", "fail closed · escalate", "alert", AMBER, w=210)

# entry
d.el.append(f'<circle cx="80" cy="{ys}" r="9" fill="{INK}"/>')
d.arrow(90, ys, cxs[0] - 105, ys)

def arc(x1, x2, y, label, up=True, color=BLUE, dash=None, head="ah"):
    yy = y - 62 if up else y + 62
    d.el.append(f'<path d="M{x1},{y + (-39 if up else 39)} C {x1},{yy} {x2},{yy} {x2},{y + (-39 if up else 39)}" '
                f'fill="none" stroke="{color}" stroke-width="2.2" {"stroke-dasharray=\'5 5\'" if dash else ""} marker-end="url(#{head})"/>')
    d.text((x1 + x2) / 2, yy + (-8 if up else 18), label, 11.5, color, weight="700")

# degrade (top, →)
arc(cxs[0] + 105, cxs[1] - 105, ys, "cloud slow/zombie · WAN loss", up=True, color=BLUE)
arc(cxs[1] + 105, cxs[2] - 105, ys, "compute pressure", up=True, color=BLUE)
arc(cxs[2] + 105, 1080 - 105, ys, "recall ceiling < 0.95", up=True, color=AMBER, head="aha")
# recover (bottom, dashed ←)
arc(cxs[1] - 105, cxs[0] + 105, ys, "cloud healthy (5 probes)", up=False, color=GREEN_D, dash=True, head="ahg")
arc(cxs[2] - 105, cxs[1] + 105, ys, "pressure clears", up=False, color=GREEN_D, dash=True, head="ahg")

# legend panel
ly = 400
d.rect(48, ly, W - 96, 150, TINT_SLATE, rx=16)
d.text(72, ly + 32, "How a transition fires", 14.5, INK, anchor="start", weight="800")
items = [
    "A health ping + the detector's own timing produce an observed condition about every 2 seconds.",
    "A change must persist for N consistent probes before it takes effect — degrade N=3, recover N=5 (no flapping).",
    "The mode is a compiled-policy lookup; if none matches, the runtime reports NO VERIFIED MODE and holds the conservative fallback.",
]
for i, t in enumerate(items):
    d.badge(84, ly + 62 + i * 30, i + 1, SUB if i < 2 else AMBER)
    d.text(104, ly + 66 + i * 30, t, 12, INK, anchor="start")
d.pill(W - 340, ly + 20, 280, "solid → degrade (fast)   ·   dashed ← recover (slow)", fill="#e7f0fb", tx=BLUE, h=26, size=11)

render(d.out(), "d4_runtime", Path(__file__).parent, W, H)
print("d4 done")
