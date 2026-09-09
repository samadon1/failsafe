"""Diagram 2 — the trust boundary: AI proposes · evaluator verifies · runtime decides.
Layered-stack idiom (NVIDIA NIM-operator style) making the defensible separation explicit."""
from pathlib import Path

from kit import AMBER, GREEN, GREEN_D, GREEN_PALE, INK, SUB, SVG, VIOLET, render

W, H = 1100, 620
d = SVG(W, H)
d.text(W / 2, 52, "Why you can trust the agent", 24, INK, weight="800")
d.text(W / 2, 80, "the LLM never decides what a failing system does — measurement does", 14.5, SUB)

x, w = 150, W - 300
bh, gap = 96, 26
y = 130

# Band 1 — AI proposes (pale green, LLM allowed)
d.rect(x, y, w, bh, GREEN_PALE, rx=12, stroke="#9fce4e", sw=2)
d.text(x + 24, y + 38, "AI  ·  proposes", 17, GREEN_D, anchor="start", weight="800")
d.text(x + 24, y + 62, "Nemotron compiles missions and proposes candidate degraded modes; grid / random / greedy also propose",
       12.5, SUB, anchor="start")
d.pill(x + w - 150, y + 34, 128, "LLM allowed", fill="#eef7dd", tx=GREEN_D, border=GREEN, h=30)

# divider line 1
y2 = y + bh + gap
d.el.append(f'<line x1="{x-30}" y1="{y2-gap/2}" x2="{x+w+30}" y2="{y2-gap/2}" stroke="#c9d3dd" stroke-width="1.5" stroke-dasharray="2 4"/>')

# Band 2 — evaluator verifies (green hero)
d.rect(x, y2, w, bh, GREEN, rx=12, shadow=True)
d.text(x + 24, y2 + 38, "Deterministic evaluator  ·  verifies", 17, "#ffffff", anchor="start", weight="800")
d.text(x + 24, y2 + 62, "measured metrics checked against the mission invariants  →  PASS / FAIL. A missing or unavailable metric FAILS.",
       12.5, "#eaf5d3", anchor="start")

# hard line: no LLM below
y3 = y2 + bh + gap
d.el.append(f'<line x1="{x-30}" y1="{y3-gap/2}" x2="{x+w+30}" y2="{y3-gap/2}" stroke="{AMBER}" stroke-width="2.4"/>')
d.text(x + w + 34, y3 - gap / 2 + 4, "no LLM", 12, AMBER, anchor="start", weight="800")
d.text(x + w + 34, y3 - gap / 2 + 19, "below", 12, AMBER, anchor="start", weight="800")

# Band 3 — runtime decides (dark)
d.rect(x, y3, w, bh, INK, rx=12, shadow=True)
d.text(x + 24, y3 + 38, "Deterministic runtime  ·  decides", 17, "#ffffff", anchor="start", weight="800")
d.text(x + 24, y3 + 62, "compiled-policy lookup only; activates a verified mode, or reports NO VERIFIED MODE and fails closed",
       12.5, "#c7ced6", anchor="start")
d.pill(x + w - 168, y3 + 34, 146, "on-device · offline-safe", fill="#2b3644", tx="#9fce4e", h=30, size=11.5)

# vertical labels
d.text(x - 60, y2 + bh / 2, "compile-time", 12, SUB, weight="700")
d.el[-1] = d.el[-1].replace("<text", f'<text transform="rotate(-90 {x-60} {y2+bh/2})"')
d.text(x - 60, y3 + bh / 2, "run-time", 12, SUB, weight="700")
d.el[-1] = d.el[-1].replace("<text", f'<text transform="rotate(-90 {x-60} {y3+bh/2})"')

render(d.out(), "d2_trust", Path(__file__).parent, W, H)
print("d2 done")
