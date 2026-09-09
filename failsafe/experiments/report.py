"""Assemble stored experiment results into a Phase 1 report (markdown). Numbers come only from
result files; nothing is typed in by hand."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

from failsafe.experiments.catalog import config_display_name
from failsafe.experiments.rescore import RULE_VERSION
from failsafe.experiments.schema import ExperimentResult


def load_results(directory: Path) -> list[ExperimentResult]:
    from failsafe.experiments.runner import result_files

    return [ExperimentResult.from_json(p.read_text()) for p in result_files(directory)]


def cname(r: ExperimentResult) -> str:
    return config_display_name(r.experiment.config)


def _v(r: ExperimentResult, k: str, fmt: str = "{:.3f}") -> str:
    m = r.metrics.get(k)
    return "n/a" if m is None or m.value is None else fmt.format(m.value)


def _note(r: ExperimentResult, k: str) -> str:
    m = r.metrics.get(k)
    return "" if m is None else m.note


def matrix_table(results: list[ExperimentResult]) -> str:
    rows = [
        "| scenario | config | recall | precision | p95 latency | max latency | frames dropped | cloud req / timeouts / unavail | cloud bytes | det ms | pressure× | edge cpu % | result |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    order = sorted(results, key=lambda r: (r.experiment.scenario.name, cname(r)))
    for r in order:
        rec = _v(r, "critical_event_recall") + " (" + _note(r, "critical_event_recall").split(" ")[0] + ")"
        cloud = f"{_v(r,'cloud_requests','{:.0f}')} / {_v(r,'cloud_timeouts','{:.0f}')} / {_v(r,'cloud_unavailable','{:.0f}')}"
        rows.append(
            f"| {r.experiment.scenario.name} | {cname(r)} | {rec} | {_v(r,'alert_precision')} | "
            f"{_v(r,'alert_latency_p95_ms','{:.0f} ms')} | {_v(r,'alert_latency_max_ms','{:.0f} ms')} | {_v(r,'frames_dropped','{:.0f}')} | {cloud} | "
            f"{_v(r,'cloud_bytes_total','{:,.0f}')} | {_v(r,'detector_ms_mean','{:.1f}')} | {_v(r,'compute_pressure_observed','{:.2f}')} | "
            f"{_v(r,'cpu_percent_mean','{:.0f}')} | {'**PASS**' if r.passed else 'FAIL'}{' †' if (r.metric_value('qc_suspect_contention') or 0) else ''} |"
        )
    rows.append("")
    rows.append("† run flagged as suspect: detector slowed > 1.25× its unpressured baseline without injected pressure (external contention, D-028)")
    return "\n".join(rows)


def provenance_block(results: list[ExperimentResult]) -> str:
    shas = sorted({r.provenance.git_sha or "?" for r in results})
    corpora = sorted({f"{r.experiment.corpus.tier}/seed{r.experiment.corpus.seed}/{r.provenance.corpus_hash}" for r in results})
    dets = sorted({r.provenance.detector_model for r in results})
    conf = sorted({r.provenance.confirmer_model for r in results if r.provenance.confirmer_model})
    hosts = sorted({r.provenance.platform for r in results})
    rules = sorted({r.provenance.extra.get("scoring_rule", "original") for r in results})
    return (
        f"- experiments: {len(results)} · backend: local · replay: real time (time_scale=1.0)\n"
        f"- corpus: {', '.join(corpora)} (synthetic discovery corpus; no real holdout yet)\n"
        f"- detector: {', '.join(dets)} on CPU · cloud stand-in: {', '.join(conf) or '—'} (child process, D-006/D-014)\n"
        f"- git: {', '.join(shas)} · platform: {', '.join(hosts)}\n"
        f"- scoring rule: {', '.join(rules)} (current: {RULE_VERSION})\n"
    )


def knob_tradeoffs(results: list[ExperimentResult]) -> str:
    """Pairs of experiments that differ in exactly one knob, same scenario → what that knob costs."""
    by_scn: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        by_scn[r.experiment.scenario.name].append(r)
    lines = ["| scenario | A | B | knob | recall A→B | precision A→B | p95 A→B | dropped A→B | cloud bytes A→B |", "|---|---|---|---|---|---|---|---|---|"]
    for scn, rs in sorted(by_scn.items()):
        for i, a in enumerate(rs):
            for b in rs[i + 1 :]:
                da = a.experiment.config.model_dump(exclude={"name", "description"})
                db = b.experiment.config.model_dump(exclude={"name", "description"})
                diff = [k for k in da if da[k] != db[k]]
                if len(diff) != 1:
                    continue
                k = diff[0]
                lines.append(
                    f"| {scn} | {cname(a)} | {cname(b)} | {k}: {da[k]}→{db[k]} | "
                    f"{_v(a,'critical_event_recall')}→{_v(b,'critical_event_recall')} | {_v(a,'alert_precision')}→{_v(b,'alert_precision')} | "
                    f"{_v(a,'alert_latency_p95_ms','{:.0f}')}→{_v(b,'alert_latency_p95_ms','{:.0f}')} | {_v(a,'frames_dropped','{:.0f}')}→{_v(b,'frames_dropped','{:.0f}')} | "
                    f"{_v(a,'cloud_bytes_total','{:,.0f}')}→{_v(b,'cloud_bytes_total','{:,.0f}')} |"
                )
    return "\n".join(lines) if len(lines) > 2 else "(no single-knob pairs)"


def scenario_effect(results: list[ExperimentResult]) -> str:
    """Same config across scenarios → what the failure condition costs."""
    by_cfg: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        by_cfg[cname(r)].append(r)
    lines = ["| config | scenario | recall | precision | p95 latency | cloud timeouts | result |", "|---|---|---|---|---|---|---|"]
    for cfg, rs in sorted(by_cfg.items()):
        if len(rs) < 2:
            continue
        for r in sorted(rs, key=lambda x: x.experiment.scenario.name):
            lines.append(
                f"| {cfg} | {r.experiment.scenario.name} | {_v(r,'critical_event_recall')} | {_v(r,'alert_precision')} | "
                f"{_v(r,'alert_latency_p95_ms','{:.0f} ms')} | {_v(r,'cloud_timeouts','{:.0f}')} | {'**PASS**' if r.passed else 'FAIL'} |"
            )
    return "\n".join(lines) if len(lines) > 2 else "(no config measured under more than one scenario)"


def build_report(directory: Path) -> str:
    results = load_results(directory)
    passed = [r for r in results if r.passed]
    return (
        "# Phase 1 results — restricted-zone workload\n\n"
        "All numbers are MEASURED under wall-clock real-time replay unless marked n/a. "
        "`recall (x/y)` = detected / total ground-truth events. Generated by `failsafe report`; nothing hand-typed.\n\n"
        "## Provenance\n\n" + provenance_block(results) + "\n"
        f"## Matrix ({len(passed)} PASS / {len(results) - len(passed)} FAIL)\n\n" + matrix_table(results) + "\n\n"
        "## Single-knob tradeoffs (same scenario, configs differing in exactly one knob)\n\n" + knob_tradeoffs(results) + "\n\n"
        "## Failure-condition effect (same config across scenarios)\n\n" + scenario_effect(results) + "\n"
    )


def repeatability_table(results: list[ExperimentResult]) -> str:
    """Mean ± sd over repeated runs of the same experiment id (≥ 2 runs)."""
    import statistics

    groups: dict[str, list[ExperimentResult]] = defaultdict(list)
    for r in results:
        groups[r.experiment.id].append(r)
    rows = ["| scenario | config | runs | recall | p95 latency (ms) | frames dropped | edge CPU % | pass rate |", "|---|---|---|---|---|---|---|---|"]
    def ms(vals, fmt):
        vals = [v for v in vals if v is not None]
        if not vals:
            return "n/a"
        m = statistics.fmean(vals)
        sd = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        return f"{fmt.format(m)} ± {fmt.format(sd)}"
    any_rows = False
    for eid, rs in sorted(groups.items(), key=lambda kv: (kv[1][0].experiment.scenario.name, cname(kv[1][0]))):
        if len(rs) < 2:
            continue
        any_rows = True
        r0 = rs[0]
        rows.append(
            f"| {r0.experiment.scenario.name} | {cname(r0)} | {len(rs)} | {ms([r.metric_value('critical_event_recall') for r in rs], '{:.3f}')} | "
            f"{ms([r.metric_value('alert_latency_p95_ms') for r in rs], '{:.0f}')} | {ms([r.metric_value('frames_dropped') for r in rs], '{:.0f}')} | "
            f"{ms([r.metric_value('cpu_percent_mean') for r in rs], '{:.0f}')} | {sum(1 for r in rs if r.passed)}/{len(rs)} |"
        )
    return "\n".join(rows) if any_rows else "(no experiment has more than one run yet)"
