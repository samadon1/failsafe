"""Write the compiled policy's coverage table into docs/RESEARCH.md, between markers, from
docs/site/policy.json. Standard library only, so it runs anywhere; scripts/export_policy.py
calls it after every export, and scripts/build_report.py picks the result up.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "docs" / "site" / "policy.json"
MD = ROOT / "docs" / "RESEARCH.md"
START, END = "<!-- policy-coverage:start -->", "<!-- policy-coverage:end -->"

ORDER = ["healthy", "wan_slow", "wan_severely_slow", "wan_zombie", "wan_timeout", "wan_offline",
         "bandwidth_2mbps", "compute_severe", "offline_compute_severe"]
PLAIN = {"healthy": "Cloud fine", "wan_slow": "Cloud slow (about half a second)", "wan_severely_slow": "Cloud very slow (about 1.5 s)",
         "wan_zombie": "Cloud slow but alive (5 to 10 s)", "wan_timeout": "Cloud timing out", "wan_offline": "Cloud dead",
         "bandwidth_2mbps": "Link down to 2 Mbps", "compute_severe": "Device overloaded",
         "offline_compute_severe": "Cloud dead and device overloaded"}
WHY = {"untested": "Never tested yet.", "insufficient_evidence": "A candidate passed, but on fewer clean runs than the bar requires.",
       "marginal": "A candidate passed every clean run, but inside the noise floor.",
       "refuted": "Every candidate tried failed at least one clean run."}


def out_of_100(x) -> str:
    return "?" if x is None else f"{int(x * 100)}/100"


def table(p: dict) -> str:
    by: dict[str, tuple[str, dict]] = {}
    for m in p.get("modes", []):
        by[m["verification"]["scenario"]] = ("mode", m)
    for u in p.get("unverified", []):
        by.setdefault(u["scenario"], ("unverified", u))
    rows = ["| Condition | Verdict | What the experiments showed |", "|---|---|---|"]
    for s in ORDER:
        if s not in by:
            continue
        kind, e = by[s]
        name = f"**{PLAIN.get(s, s)}** (`{s}`)"
        if kind == "mode":
            v = e["verification"]
            seeds = f", across corpus seeds {', '.join(map(str, v['seeds']))}" if len(v.get("seeds", [])) > 1 else ""
            rows.append(f"| {name} | verified, {v['tier']} | `{e['name']}`: catches {out_of_100(v['recall'])}, alerts within "
                        f"{v['p95_latency_ms'] / 1000:.1f} s. Worst of {v['clean_runs']} clean runs{seeds}. |")
        else:
            verdict = "untested" if e["reason"] == "untested" else "no verified mode"
            extra = (" The whole 72-configuration space was tried." if e.get("exhaustive")
                     else (f" {e['candidates_tested']} candidate{'' if e['candidates_tested'] == 1 else 's'} tried." if e.get("candidates_tested") else ""))
            nxt = f" Next: repeat `{e['best_candidate_config']}`." if e.get("best_candidate_config") else ""
            rows.append(f"| {name} | {verdict} | {WHY.get(e['reason'], e['reason'])}{extra}{nxt} |")
    fb = p.get("fallback")
    rule = p.get("admission", {})
    rows += ["", f"Admission bar: {rule.get('min_clean_runs', '?')} clean runs across {rule.get('min_clean_seeds', '?')} corpus seeds, every one passing. Corpus {p.get('corpus', '?')}."
             + (f" Fallback when nothing is verified: `{fb['name']}`." if fb else "")
             + f" Compiled {str(p.get('compiled_at', ''))[:19]} UTC."]
    return "\n".join(rows)


def main() -> None:
    p = json.loads(POLICY.read_text())
    md = MD.read_text()
    if START not in md or END not in md:
        sys.exit("coverage markers missing in docs/RESEARCH.md")
    head, tail = md[: md.index(START) + len(START)], md[md.index(END):]
    MD.write_text(head + "\n" + table(p) + "\n" + tail)
    print(f"RESEARCH.md coverage table updated: {len(p.get('modes', []))} verified, {len(p.get('unverified', []))} unverified")


if __name__ == "__main__":
    main()
