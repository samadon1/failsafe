"""Grace-window sensitivity: how much do recall, precision and the pass/fail verdicts depend on the
post-event matching grace? D-021 justified the 0.5 s lead with a measured incident; the 2.0 s
grace had no justification on record, and on a corpus whose median event lasts 1.2 s it can be
longer than the event it is attributed to.

Read-only. Every stored synthetic result is rescored in memory at each grace value; nothing on
disk is touched. Prints a markdown table (for docs/EXPERIMENTS.md) and lists every result whose
verdict flips between windows, so the decision about the window is made on evidence.

    uv run python scripts/grace_sensitivity.py            # graces 0.5 1.0 2.0
    uv run python scripts/grace_sensitivity.py 0.25 1 2 3
"""
from __future__ import annotations

import statistics
import sys
from collections import defaultdict
from pathlib import Path

from failsafe.experiments.rescore import StaleCorpus, rescore
from failsafe.experiments.runner import result_files
from failsafe.experiments.schema import ExperimentResult

GRACES = [float(x) for x in (sys.argv[1:] or ["0.5", "1.0", "2.0"])]


def _mean(xs: list[float | None]) -> str:
    v = [x for x in xs if x is not None]
    return f"{statistics.fmean(v):.3f}" if v else "n/a"


def main() -> None:
    per_file: dict[str, dict[float, tuple[bool, float | None, float | None, str, str]]] = defaultdict(dict)
    skipped = 0
    for p in result_files(Path("artifacts/experiments")):
        r = ExperimentResult.from_json(p.read_text())
        if r.experiment.corpus.kind != "synthetic":
            continue
        try:
            for g in GRACES:
                a = rescore(r, grace_s=g)
                per_file[p.name][g] = (a.passed, a.metric_value("critical_event_recall"), a.metric_value("alert_precision"),
                                       r.experiment.config.name, r.experiment.scenario.name)
        except StaleCorpus:
            skipped += 1

    agg: dict[tuple[str, float], list[tuple[bool, float | None, float | None]]] = defaultdict(list)
    for d in per_file.values():
        for g, (ok, rec, prec, _cfg, scn) in d.items():
            agg[(scn, g)].append((ok, rec, prec))
    scenarios = sorted({s for s, _ in agg})

    head = " | ".join(f"grace {g:g} s: pass / recall / precision" for g in GRACES)
    print(f"| scenario | runs | {head} |")
    print("|---|---|" + "---|" * len(GRACES))
    for s in scenarios:
        n = len(agg[(s, GRACES[0])])
        cells = [f"{sum(o for o, _, _ in agg[(s, g)])} / {_mean([x for _, x, _ in agg[(s, g)]])} / {_mean([x for _, _, x in agg[(s, g)]])}" for g in GRACES]
        print(f"| {s} | {n} | " + " | ".join(cells) + " |")

    flips = {fn: d for fn, d in per_file.items() if len({d[g][0] for g in GRACES}) > 1}
    print(f"\nverdict flips between windows: {len(flips)} of {len(per_file)} results" + (f"; {skipped} stale results skipped" if skipped else ""))
    for fn in sorted(flips):
        d = flips[fn]
        cfg, scn = d[GRACES[-1]][3], d[GRACES[-1]][4]
        print(f"  {cfg} × {scn}: " + ", ".join(f"@{g:g}s {'PASS' if d[g][0] else 'FAIL'} recall {d[g][1] if d[g][1] is None else round(d[g][1], 3)}" for g in GRACES) + f"  [{fn}]")


if __name__ == "__main__":
    main()
