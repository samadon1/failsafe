"""Diagram 3 — the degradation ladder, read from the compiled policy.

Every rung is a mode in artifacts/resilience-policy.yaml (D-037): the condition it is verified
for, capability retained, what it keeps and gives up, and the worst run across its clean runs.
The floor lists the conditions the policy refuses to serve, with the compiler's reason. Nothing
in this file types a result number; regenerate after `failsafe compile-policy`.
"""
from pathlib import Path

from kit import (AMBER, BORD, CARD, GREEN, GREEN_D, INK, SLATE, SUB, SVG,
                 TINT_AMBER, TINT_GREEN, render)

from failsafe.policy.schema import ResiliencePolicy

ROOT = Path(__file__).resolve().parents[2]
policy = ResiliencePolicy.from_yaml(ROOT / "artifacts" / "resilience-policy.yaml")

CLOUD = {"healthy": "cloud fine", "slow": "cloud slow", "severely_slow": "cloud very slow",
         "zombie": "cloud slow but alive", "timeout": "cloud timing out", "offline": "cloud dead"}
COMPUTE = {"normal": "", "moderate": "device busy", "severe": "device overloaded", "critical": "device saturated"}
REASON = {"untested": "untested", "insufficient_evidence": "not enough evidence yet",
          "marginal": "passes, but inside the noise floor", "refuted": "every candidate failed"}


def condition(c) -> str:
    parts = [CLOUD.get(s.value, s.value) for s in c.cloud_state]
    parts += [COMPUTE.get(s.value, s.value) for s in c.compute_pressure if COMPUTE.get(s.value, s.value)]
    if c.bandwidth_mbps_max is not None:
        parts.append(f"link ≤ {c.bandwidth_mbps_max:g} Mbps")
    return " · ".join(parts)


def keeps(cfg) -> str:
    out = [f"cloud confirmation ({cfg.cloud_timeout_ms / 1000:g} s timeout)" if cfg.cloud_confirmation else "local decisions only"]
    if cfg.historical_indexing:
        out.append("indexing")
    out.append("all cameras" if cfg.background_fps > 0 and cfg.drop_background_streams == "none" else "background trimmed")
    out.append(f"{cfg.critical_fps} fps @ {cfg.detector_resolution} px on the critical camera")
    return " · ".join(out)


def icon_for(cfg) -> str:
    if cfg.cloud_confirmation:
        return "shield"
    if cfg.critical_fps < 15 or cfg.detector_resolution < 640:
        return "activity"
    return "cpu" if not cfg.historical_indexing else "check"


modes = sorted(policy.modes, key=lambda m: -m.capability_retained)
RH, GAP, TOP = 104, 14, 118
floor_lines = len(policy.unverified)
FLOOR_H = 44 + 22 * max(1, floor_lines)
W = 1300
H = TOP + len(modes) * (RH + GAP) + FLOOR_H + 40
d = SVG(W, H)
d.text(48, 50, "The degradation ladder", 25, INK, anchor="start", weight="800")
for i, ln in enumerate([
    "Each rung is a mode in the compiled policy: the condition it is verified for, what it keeps, what it gives up, and the numbers that passed.",
    f"The green panel is the worst run across every clean run (admission: {policy.admission.min_clean_runs} clean runs, all must pass). Corpus {policy.corpus}."]):
    d.text(48, 78 + i * 19, ln, 12.5, SUB, anchor="start")

x = 48; y = TOP
for i, m in enumerate(modes):
    cfg, v = m.config, m.verification
    ind = i * 30
    rx = x + ind; rw = W - 96 - ind
    d.rect(rx, y, rw, RH, CARD, rx=14, stroke=BORD, sw=1.6, shadow=True)
    d.rect(rx + 14, y + RH / 2 - 20, 40, 40, TINT_GREEN, rx=10)
    d.icon(rx + 21, y + RH / 2 - 13, 26, icon_for(cfg), GREEN, sw=2.0)
    d.text(rx + 66, y + 28, m.name.upper(), 16, INK, anchor="start", weight="800")
    d.text(rx + 66, y + 45, condition(m.conditions), 11, SLATE, anchor="start")
    d.text(rx + 66, y + 62, f"capability {m.capability_retained:.2f}", 11, SUB, anchor="start")
    d.rect(rx + 66, y + 70, 150, 8, "#e3e8ec", rx=4)
    d.rect(rx + 66, y + 70, 150 * m.capability_retained, 8, GREEN, rx=4)
    kx = rx + 330
    d.text(kx, y + 34, "keeps", 11, GREEN_D, anchor="start", weight="800")
    d.text(kx + 62, y + 34, keeps(cfg), 12, INK, anchor="start")
    d.text(kx, y + 62, "gives up", 11, AMBER, anchor="start", weight="800")
    d.text(kx + 62, y + 62, " · ".join(m.sacrifices) if m.sacrifices else "nothing", 12, SUB, anchor="start")
    vw = 236; vx = rx + rw - vw - 20
    d.rect(vx, y + 22, vw, RH - 44, TINT_GREEN, rx=10, stroke=GREEN, sw=1.5)
    d.icon(vx + 14, y + RH / 2 - 12, 22, "check", GREEN_D, sw=2.2)
    d.text(vx + 44, y + RH / 2 - 3, f"recall {v.recall:.3f}  ·  p95 {v.p95_latency_ms:.0f} ms", 12.5, GREEN_D, anchor="start", weight="800")
    d.text(vx + 44, y + RH / 2 + 15, f"{v.tier.upper()} · worst of {v.clean_runs} clean runs", 11, SUB, anchor="start")
    if i < len(modes) - 1:
        d.arrow(rx + 34, y + RH, rx + 34 + 30, y + RH + GAP, color=SUB, sw=2)
    y += RH + GAP

# floor: what the policy refuses to serve, and why
ind = len(modes) * 30
d.rect(x + ind, y, W - 96 - ind, FLOOR_H, TINT_AMBER, rx=14, stroke=AMBER, sw=1.8)
d.rect(x + ind + 14, y + 12, 38, 38, "#fbe3c8", rx=10)
d.icon(x + ind + 21, y + 18, 24, "alert", AMBER, sw=2.2)
fb = f"fail closed to `{policy.fallback.name}` + escalate" if policy.fallback else "halt + escalate"
d.text(x + ind + 64, y + 27, f"NO VERIFIED MODE  →  {fb}", 15, "#9a5a12", anchor="start", weight="800")
for j, u in enumerate(policy.unverified):
    why = REASON.get(u.reason, u.reason) + (" (whole space tried)" if u.exhaustive else "")
    tail = f"; next: repeat {u.best_candidate_config}" if u.best_candidate_config else ""
    tested = f" · {u.candidates_tested} candidates" if u.candidates_tested and not u.exhaustive else ""
    d.text(x + ind + 64, y + 49 + j * 22, f"{condition(u.conditions)}: {why}{tested}{tail}", 12, "#9a5a12", anchor="start")

render(d.out(), "d3_ladder", Path(__file__).parent, W, H)
print(f"d3 done: {len(modes)} rungs, {floor_lines} unverified, H={H}")
