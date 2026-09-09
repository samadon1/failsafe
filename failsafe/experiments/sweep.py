"""Sequential local sweep over configs × scenarios. Writes results JSON + a markdown table.
Sequential on purpose: concurrent experiments on one laptop would contend for CPU and corrupt each
other's latency and pressure measurements."""

from __future__ import annotations

import json
import time
from datetime import UTC, datetime
from pathlib import Path

from rich.console import Console

from failsafe.experiments.catalog import CONFIGS, load_scenarios
from failsafe.experiments.runner import HostBusy, LocalRunner, git_sha, result_files
from failsafe.experiments.schema import CorpusRef, Experiment, ExperimentResult, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec


def _fmt(res: ExperimentResult, key: str, fmt: str = "{:.3f}") -> str:
    m = res.metrics.get(key)
    if m is None or m.value is None:
        return "n/a"
    return fmt.format(m.value)


def results_table(results: list[ExperimentResult]) -> str:
    rows = [
        "| config | scenario | recall | precision | p95 latency | cloud bytes | timeouts | frames dropped | det ms | pressure× | cpu % | result |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        rows.append(
            "| {c} | {s} | {rec} | {prec} | {lat} | {bytes} | {to} | {drop} | {det} | {px} | {cpu} | {res} |".format(
                c=r.experiment.config.name,
                s=r.experiment.scenario.name,
                rec=_fmt(r, "critical_event_recall") + f" ({r.metrics['critical_event_recall'].note.split(' ')[0]})",
                prec=_fmt(r, "alert_precision"),
                lat=_fmt(r, "alert_latency_p95_ms", "{:.0f} ms"),
                bytes=_fmt(r, "cloud_bytes_total", "{:,.0f}"),
                to=_fmt(r, "cloud_timeouts", "{:.0f}"),
                drop=_fmt(r, "frames_dropped", "{:.0f}"),
                det=_fmt(r, "detector_ms_mean", "{:.1f}"),
                px=_fmt(r, "compute_pressure_observed", "{:.2f}"),
                cpu=_fmt(r, "cpu_percent_mean", "{:.0f}"),
                res="**PASS**" if r.passed else "FAIL",
            )
        )
    return "\n".join(rows)


def pressure_last(scenarios: list[Scenario]) -> list[Scenario]:
    """Order scenarios so that compute-pressure ones run last (stable otherwise): a pressure run
    pins every core for minutes and the runs that follow it read slow on a passively cooled
    machine (D-043)."""
    return sorted(scenarios, key=lambda s: s.compute_pressure.value != "normal")


def run_sweep(
    mission_path: Path,
    configs: str | list[OperatingConfig],
    scenarios: str | list[Scenario],
    tier: str,
    seed: int,
    out: Path,
    name: str,
    device: str,
    console: Console,
    skip_existing: bool = True,
    repeats: int = 1,
    cooldown_s: float = 90.0,
    busy_policy: str = "abort",
) -> list[ExperimentResult]:
    mission = MissionSpec.from_yaml(mission_path)
    if isinstance(configs, str):
        cfg_list = list(CONFIGS.values()) if configs == "all" else [CONFIGS[c.strip()] for c in configs.split(",")]
    else:
        cfg_list = list(configs)
    if isinstance(scenarios, str):
        scen = load_scenarios()
        sc_list = list(scen.values()) if scenarios == "all" else [scen[s.strip()] for s in scenarios.split(",")]
    else:
        sc_list = list(scenarios)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    results_path = out / f"{name}_{stamp}.json"
    table_path = out / f"{name}_{stamp}.md"

    runner = LocalRunner(detector_device=device, busy_policy=busy_policy)
    results: list[ExperimentResult] = []
    total = len(cfg_list) * len(sc_list) * repeats
    exp_dir = runner.artifacts_dir / "experiments"
    sc_list = pressure_last(sc_list)  # a pressure run heats the machine; the next runs read slow (D-043)
    try:
        k = 0
        for sc in sc_list:
            for cfg in cfg_list:
                exp = Experiment(mission=mission, scenario=sc, config=cfg, corpus=CorpusRef(tier=tier, seed=seed), proposed_by="manual")
                have = len(result_files(exp_dir, exp.id))
                for rep in range(repeats):
                    k += 1
                    if skip_existing and have >= repeats:
                        console.print(f"[{k}/{total}] {cfg.name} × {sc.name} ({exp.id}) — {have} result(s) exist, skipping")
                        continue
                    if skip_existing and rep < have:
                        continue
                    console.print(f"[{k}/{total}] {cfg.name} × {sc.name} ({exp.id}) run {rep + 1}/{repeats} …")
                    try:
                        res = runner.run(exp)
                    except HostBusy as e:
                        console.print(f"      STOPPED before {cfg.name} × {sc.name}: {e}", markup=False)
                        console.print("      remaining experiments not run; results so far are stored. Fix the host, then re-run.", markup=False)
                        raise SystemExit(3)
                    results.append(res)
                    if sc.compute_pressure.value != "normal" and cooldown_s > 0:
                        console.print(f"      cooling down {cooldown_s:.0f} s after a compute-pressure run (D-043)")
                        time.sleep(cooldown_s)
                    console.print(
                        f"      recall={_fmt(res, 'critical_event_recall')} precision={_fmt(res, 'alert_precision')} "
                        f"p95={_fmt(res, 'alert_latency_p95_ms', '{:.0f}ms')} dropped={_fmt(res, 'frames_dropped', '{:.0f}')} "
                        f"→ {'PASS' if res.passed else 'FAIL'}"
                    )
                    # persist incrementally so a crash keeps what was measured
                    results_path.write_text(json.dumps([json.loads(r.to_json()) for r in results], indent=1))
                    table_path.write_text(_header(name, tier, seed, mission) + results_table(results) + "\n")
    finally:
        runner.close()
    console.print(f"wrote {results_path}\nwrote {table_path}")
    return results


def _header(name: str, tier: str, seed: int, mission: MissionSpec) -> str:
    return (
        f"# Sweep `{name}`\n\n"
        f"corpus: synthetic tier=`{tier}` seed={seed} · mission `{mission.name}` ({mission.hash}) · git {git_sha()} · "
        f"replay: real time (time_scale=1.0) · backend: local\n\n"
        "Recall column shows detected/total GT events. Latency is UNAVAILABLE (n/a) if no GT event was detected.\n\n"
    )
