"""Failsafe CLI. CLI before UI (CLAUDE.md #9)."""

from __future__ import annotations

import json
import logging
import statistics
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Failsafe — resilience compiler for edge / Physical-AI systems")
corpus_app = typer.Typer(no_args_is_help=True, help="synthetic corpus + real holdout tooling")
app.add_typer(corpus_app, name="corpus")
console = Console()


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader (no dependency): KEY=VALUE lines, # comments, does not overwrite
    variables already set in the environment."""
    import os
    from pathlib import Path as _P

    p = _P(path)
    if not p.exists():
        return
    for line in p.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip().strip('"').strip("'")
        if k and k not in os.environ:
            os.environ[k] = v


_load_dotenv()


@app.callback()
def _setup(verbose: bool = typer.Option(False, "--verbose", "-v")):
    logging.basicConfig(level=logging.INFO if verbose else logging.WARNING, format="%(levelname)s %(name)s: %(message)s")


# ------------------------------------------------------------------------------------------------
# corpus
# ------------------------------------------------------------------------------------------------


@corpus_app.command("build-cutouts")
def corpus_build_cutouts(force: bool = False):
    """Extract person cutouts from ultralytics' sample images (needs network on first run)."""
    from failsafe.corpus.assets import build_cutouts

    paths = build_cutouts(force=force)
    console.print(f"[green]{len(paths)} cutouts[/green] in datasets/cutouts/")


@corpus_app.command("stats")
def corpus_stats(tier: str = "quick", seed: int = 1, as_json: bool = typer.Option(False, "--json")):
    """Ground-truth statistics of a synthetic scene (the corpus-statistics deliverable)."""
    from failsafe.corpus.ground_truth import compute_ground_truth
    from failsafe.corpus.scene import generate_scene

    scene = generate_scene(tier, seed)
    gt = compute_ground_truth(scene)
    ev = gt.events
    stats = {
        "tier": tier,
        "seed": seed,
        "scene_hash": scene.hash,
        "duration_s": scene.duration_s,
        "cameras": {c.name: {"role": c.role, "events": len(gt.for_camera(c.name))} for c in scene.cameras},
        "gt_events": len(ev),
        "distractor_tracks": len(gt.distractor_tracks),
        "near_miss_tracks": sum(1 for t in gt.distractor_tracks if t.kind == "near_miss"),
        "wanderer_tracks": sum(1 for t in gt.distractor_tracks if t.kind == "wanderer"),
        "event_duration_s": _dist([e.duration_s for e in ev]),
        "person_height_px": _dist([e.height_px for e in ev]),
        "speed_px_s": _dist([e.speed_px_s for e in ev]),
        "occluded_events": sum(1 for e in ev if e.occlusion > 0),
        "low_contrast_events(alpha<0.7)": sum(1 for e in ev if e.alpha < 0.7),
        "events_shorter_than_0.5s": sum(1 for e in ev if e.duration_s < 0.5),
        "events_shorter_than_0.2s": sum(1 for e in ev if e.duration_s < 0.2),
        "events_height_below_40px": sum(1 for e in ev if e.height_px < 40),
    }
    if as_json:
        print(json.dumps(stats, indent=2))
        return
    t = Table(title=f"corpus {tier} seed={seed} hash={scene.hash}")
    t.add_column("stat")
    t.add_column("value")
    for k, v in stats.items():
        t.add_row(k, json.dumps(v) if isinstance(v, (dict, list)) else str(v))
    console.print(t)


def _dist(xs: list[float]) -> dict[str, float]:
    if not xs:
        return {}
    s = sorted(xs)
    return {
        "min": round(s[0], 3),
        "p25": round(s[len(s) // 4], 3),
        "median": round(statistics.median(s), 3),
        "p75": round(s[(3 * len(s)) // 4], 3),
        "max": round(s[-1], 3),
    }


@corpus_app.command("export")
def corpus_export(
    tier: str = "smoke",
    seed: int = 1,
    camera: str = "A",
    out: Path = Path("artifacts/corpus"),
    t0: float = 0.0,
    t1: float | None = None,
):
    """Render one camera of a synthetic scene to MP4 for inspection / demo."""
    from failsafe.corpus.assets import CutoutLibrary
    from failsafe.corpus.export import export_camera_mp4
    from failsafe.corpus.render import SceneRenderer
    from failsafe.corpus.scene import generate_scene

    cut = CutoutLibrary()
    scene = generate_scene(tier, seed)
    dest = export_camera_mp4(SceneRenderer(scene, cut), camera, out / f"{tier}_s{seed}_{camera}.mp4", t0, t1)
    console.print(f"wrote {dest}")


@corpus_app.command("detectability")
def corpus_detectability(tier: str = "quick", seed: int = 1, conf: float = 0.4, device: str = "cpu", weights: str = "yolov8n.pt"):
    """Offline per-event detectability by resolution (recall landscape; no latency)."""
    from failsafe.corpus.assets import CutoutLibrary
    from failsafe.corpus.detectability import analyse, near_miss_false_positives, summarise
    from failsafe.workload.detector import make_detector
    from failsafe.workload.source import SyntheticSceneSource

    src = SyntheticSceneSource.from_tier(tier, seed, CutoutLibrary())
    det = make_detector("yolo", weights, device)
    rows = analyse(src, det, conf=conf)
    summary = summarise(rows)
    summary["near_miss_frames_with_in_zone_detection"] = {
        str(r): f"{c}/{n}" for r, (c, n) in near_miss_false_positives(src, det, conf=conf).items()
    }
    summary["detector"] = det.name
    summary["conf"] = conf
    print(json.dumps(summary, indent=2))
    missed = [x for x in rows if x.hits[640] == 0]
    if missed:
        console.print(f"[yellow]{len(missed)} events never detected at 640:[/yellow]")
        for x in missed:
            e = x.event
            console.print(f"  {e.id} h={e.height_px:.0f}px dur={e.duration_s:.2f}s occ={e.occlusion} alpha={e.alpha} speed={e.speed_px_s:.0f}")


@corpus_app.command("precision-probe")
def corpus_precision_probe(tier: str = "quick", seed: int = 1, device: str = "cpu", with_confirmer: bool = True):
    """Offline: do near-miss distractors create false in-zone candidates, and does the confirmer reject them?"""
    from failsafe.corpus.assets import CutoutLibrary
    from failsafe.corpus.precision_probe import hallucination_probe, probe
    from failsafe.workload.cloud import HeavyModelConfirmer
    from failsafe.workload.detector import make_detector
    from failsafe.workload.source import SyntheticSceneSource

    src = SyntheticSceneSource.from_tier(tier, seed, CutoutLibrary())
    det = make_detector("yolo", "yolov8n.pt", device)
    conf = HeavyModelConfirmer() if with_confirmer else None
    rows = probe(src, det, conf)
    t = Table(title="near-miss frames (no real intrusion) → false in-zone candidates")
    for col in ("local thr", "frames", "false cand.", "rate", "rejected by confirmer", "<6px", "6-12px", ">12px"):
        t.add_column(col)
    for r in rows:
        t.add_row(str(r.threshold), str(r.frames), str(r.false_candidates), f"{r.false_candidates / max(1, r.frames):.3f}",
                  f"{r.rejected_by_confirmer}/{r.false_candidates}" if conf else "—",
                  *[f"{v[0]}/{v[1]}" for v in r.by_margin.values()])
    console.print(t)
    h = hallucination_probe(src, det)
    console.print("empty-frame in-zone hallucinations: " + ", ".join(f"thr {k}: {v[0]}/{v[1]}" for k, v in h.items()))


@corpus_app.command("label")
def corpus_label(clip: Path, zone: str = typer.Option(..., help='JSON list of [x,y] points'), events: str = typer.Option("[]", help="JSON list of [start_s,end_s]")):
    """Write <clip>.labels.json for a real holdout clip."""
    from failsafe.corpus.export import write_labels

    dest = write_labels(clip, [tuple(p) for p in json.loads(zone)], [tuple(e) for e in json.loads(events)])
    console.print(f"wrote {dest}")


# ------------------------------------------------------------------------------------------------
# experiments
# ------------------------------------------------------------------------------------------------


@app.command("configs")
def list_configs():
    from failsafe.experiments.catalog import CONFIGS

    t = Table(title="operating configurations")
    for col in ("name", "crit fps", "bg fps", "res", "cloud", "timeout", "on fail", "index", "drop bg", "backlog"):
        t.add_column(col)
    for c in CONFIGS.values():
        t.add_row(c.name, str(c.critical_fps), str(c.background_fps), str(c.detector_resolution), str(c.cloud_confirmation),
                  str(c.cloud_timeout_ms), c.on_cloud_failure, str(c.historical_indexing), c.drop_background_streams, c.backlog_policy)
    console.print(t)


@app.command("run")
def run_experiment(
    mission: Path = Path("missions/restricted-zone.yaml"),
    config: str = "normal",
    scenario: str = "healthy",
    tier: str = "quick",
    seed: int = 1,
    time_scale: float = 1.0,
    scenarios_file: Path = Path("scenarios/phase1.yaml"),
    device: str = "cpu",
):
    """Run one experiment locally (real-time replay) and print metrics + verification."""
    from failsafe.experiments.catalog import CONFIGS, load_scenarios
    from failsafe.experiments.runner import LocalRunner
    from failsafe.experiments.schema import CorpusRef, Experiment
    from failsafe.mission.schema import MissionSpec

    exp = Experiment(
        mission=MissionSpec.from_yaml(mission),
        scenario=load_scenarios(scenarios_file)[scenario],
        config=CONFIGS[config],
        corpus=CorpusRef(tier=tier, seed=seed),
        time_scale=time_scale,
    )
    runner = LocalRunner(detector_device=device)
    try:
        console.print(f"experiment {exp.id}: config={config} scenario={scenario} tier={tier} seed={seed} (≈{exp.corpus.tier} replay in real time)")
        res = runner.run(exp)
    finally:
        runner.close()
    print_result(res)


def print_result(res) -> None:
    t = Table(title=f"{res.experiment.config.name} × {res.experiment.scenario.name} — {'PASS' if res.passed else 'FAIL'}")
    t.add_column("metric")
    t.add_column("value", justify="right")
    t.add_column("kind")
    t.add_column("note")
    for k, m in res.metrics.items():
        v = "—" if m.value is None else (f"{m.value:.3f}" if abs(m.value) < 100 else f"{m.value:,.0f}")
        t.add_row(k, f"{v} {m.unit}".strip(), m.kind.value, m.note)
    console.print(t)
    if res.verification:
        for c in res.verification.checks:
            mark = "[green]✓[/green]" if c.passed else "[red]✗[/red]"
            val = "—" if c.value is None else f"{c.value:.3f}"
            console.print(f"  {mark} {c.metric} = {val} {c.operator} {c.threshold} {c.note}")
    for n in res.notes:
        console.print(f"  [dim]note: {n}[/dim]")
    console.print(f"  saved artifacts/experiments/{res.experiment.id}.json")


@app.command("grid")
def grid_cmd(
    scenario: str = "wan_zombie",
    mission: Path = Path("missions/restricted-zone.yaml"),
    tier: str = "quick",
    seed: int = 1,
    device: str = "cpu",
    limit: int | None = typer.Option(None, help="only the first N grid configs (for smoke testing)"),
):
    """Run the bounded search space (72 configs) under one scenario; skips configs with stored results."""
    from failsafe.experiments.catalog import grid_configs, load_scenarios
    from failsafe.experiments.sweep import run_sweep

    cfgs = grid_configs()[: limit or None]
    run_sweep(mission, cfgs, [load_scenarios()[scenario]], tier, seed, Path("artifacts/sweeps"), f"grid_{scenario}", device, console)


@app.command("repeat")
def repeat_cmd(
    configs: str = "island,normal,naive_default",
    scenarios: str = "healthy,wan_zombie,compute_severe",
    n: int = 3,
    mission: Path = Path("missions/restricted-zone.yaml"),
    tier: str = "quick",
    seed: int = 1,
    device: str = "cpu",
    cooldown_s: float = typer.Option(90.0, help="idle seconds after a compute-pressure run before the next replay (D-043)"),
    on_busy: str = typer.Option("abort", help="if the host stays busy past the idle gate's wait budget: abort (exit 3) | run (flagged) (D-044)"),
):
    """Repeatability: make sure each config × scenario has at least N stored runs."""
    from failsafe.experiments.sweep import run_sweep

    run_sweep(mission, configs, scenarios, tier, seed, Path("artifacts/sweeps"), "repeat", device, console,
              repeats=n, cooldown_s=cooldown_s, busy_policy=on_busy)


@app.command("calibrate")
def calibrate_cmd(resolutions: str = "640,480,320", n: int = 60, device: str = "cpu",
                  reset: bool = typer.Option(False, help="accept a slower reading as the host's new idle speed")):
    """Measure this host's isolated detector speed on an IDLE, COOL machine (D-041, D-043). Each
    resolution is read twice; the two must agree within 15%, and must not be more than 1.25x slower
    than the host's recorded idle speed, or the command refuses to write anything."""
    from failsafe.experiments.runner import LocalRunner

    r = LocalRunner(detector_device=device)
    try:
        b = r.calibrate_host(tuple(int(x) for x in resolutions.split(",")), n=n, reset=reset)
    except RuntimeError as e:
        console.print(str(e), style="red", markup=False)
        raise typer.Exit(1)
    finally:
        r.close()
    p = r.save_host_calibration(b, reset=reset)
    for res, ms in b.items():
        console.print(f"  {res} px: {ms:.1f} ms/frame")
    console.print(f"wrote {p}", markup=False)


@app.command("search")
def search_cmd(
    scenario: str = "wan_zombie",
    strategy: str = typer.Option("greedy", help="grid | random | greedy | llm | all"),
    provider: str = typer.Option("mock", help="for --strategy llm: mock | nemotron (needs FAILSAFE_LLM_API_KEY, NVIDIA_API_KEY or NEBIUS_API_KEY; D-035)"),
    live: bool = typer.Option(False, help="run experiments that are not in the cache (real time!)"),
    seed: int = 0,
    mission: Path = Path("missions/restricted-zone.yaml"),
    tier: str = "quick",
    corpus_seed: int = 1,
    device: str = "cpu",
):
    """Derive a verified operating mode for a scenario (or conclude NONE) with one search strategy."""
    from failsafe.experiments.catalog import load_scenarios
    from failsafe.experiments.runner import LocalRunner
    from failsafe.experiments.schema import CorpusRef
    from failsafe.mission.schema import MissionSpec
    from failsafe.search.objective import NoVerifiedMode
    from failsafe.search.strategies import STRATEGIES, Evaluator, MissingResult, ResultCache, default_space

    m = MissionSpec.from_yaml(mission)
    sc = load_scenarios()[scenario]
    runner = LocalRunner(detector_device=device) if live else None
    ev = Evaluator(ResultCache(), m, sc, CorpusRef(tier=tier, seed=corpus_seed), runner=runner, allow_live=live)
    names = list(STRATEGIES) + ["llm"] if strategy == "all" else [strategy]
    try:
        for name in names:
            if name == "llm":
                from failsafe.reasoning.planner import LLMSearch

                if provider == "nemotron":
                    from failsafe.reasoning.nemotron import NemotronProvider

                    prov = NemotronProvider()  # credentials from FAILSAFE_LLM_* / NVIDIA_API_KEY / NEBIUS_API_KEY (D-035)
                else:
                    from failsafe.reasoning.mock import MockReasoningProvider

                    prov = MockReasoningProvider()
                strat = LLMSearch(prov)
            else:
                strat = STRATEGIES[name](seed=seed) if name == "random" else STRATEGIES[name]()
            traj = strat.run(ev, default_space())
            path = traj.save()
            console.print(f"[bold]{traj.summary()}[/bold]  → {path}")
            for s in traj.steps:
                mark = "[green]PASS[/green]" if s.passed else "[red]FAIL[/red]"
                rec = "n/a" if s.recall is None else f"{s.recall:.3f}"
                p95 = "n/a" if s.p95_ms is None else f"{s.p95_ms:.0f}ms"
                console.print(f"  {s.index:2d}. {s.config.name:34} recall={rec} p95={p95} cap={s.capability:.2f} {mark} ({s.source}) — {s.reason}")
            if traj.no_verified_mode:
                console.print("[red]" + NoVerifiedMode(**traj.no_verified_mode).render() + "[/red]")
            elif traj.best_config:
                b = traj.best_config
                console.print(f"  best verified mode: {b.name} — crit {b.critical_fps} fps, bg {b.background_fps} fps, {b.detector_resolution}px, "
                              f"cloud {'on/' + str(b.cloud_timeout_ms) + 'ms' if b.cloud_confirmation else 'off'}, indexing {'on' if b.historical_indexing else 'off'}; "
                              f"capability {traj.best_capability:.2f}; pareto-optimal: {len(traj.pareto)}")
    except MissingResult as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    finally:
        if runner is not None:
            runner.close()


@app.command("compare")
def compare_cmd(
    scenario: str = "wan_zombie",
    mission: Path = Path("missions/restricted-zone.yaml"),
    tier: str = "quick",
    corpus_seed: int = 1,
    random_seeds: int = 1000,
):
    """Compare grid / random / greedy by replay against the cached grid for one scenario."""
    from failsafe.experiments.catalog import load_scenarios
    from failsafe.experiments.schema import CorpusRef
    from failsafe.mission.schema import MissionSpec
    from failsafe.search.compare import compare_strategies

    m = MissionSpec.from_yaml(mission)
    sc = load_scenarios()[scenario]
    console.print(compare_strategies(m, sc, CorpusRef(tier=tier, seed=corpus_seed), random_seeds))


@app.command("repeatability")
def repeatability_cmd(directory: Path = Path("artifacts/experiments")):
    """Run-to-run variation: mean ± sd over repeated runs of the same experiment."""
    from failsafe.experiments.report import load_results, repeatability_table

    console.print(repeatability_table(load_results(directory)))


@app.command("compile-policy")
def compile_policy_cmd(
    mission: Path = Path("missions/restricted-zone.yaml"),
    scenarios: str = typer.Option("all", help="comma-separated scenario names or 'all'"),
    out: Path = Path("artifacts/resilience-policy.yaml"),
    require_robust: bool = typer.Option(False, help="treat marginal passes as unverified"),
    min_clean_runs: int = typer.Option(2, help="un-contended runs an experiment needs before it counts as verified (D-037)"),
    min_clean_seeds: int = typer.Option(2, help="distinct corpus seeds those clean runs must span (D-048)"),
):
    """Compile stored verified results into a deployable resilience policy (no LLM involved)."""
    from failsafe.experiments.catalog import grid_configs, load_scenarios
    from failsafe.mission.schema import MissionSpec
    from failsafe.policy.compiler import compile_to_file

    scen = load_scenarios()
    sc_list = list(scen.values()) if scenarios == "all" else [scen[s.strip()] for s in scenarios.split(",")]
    names = {"healthy": "normal", "wan_offline": "offline-cloud", "wan_zombie": "zombie-cloud", "wan_slow": "slow-cloud",
             "wan_severely_slow": "severely-slow-cloud", "wan_timeout": "dead-cloud", "bandwidth_2mbps": "low-bandwidth",
             "compute_severe": "compute-pressure", "offline_compute_severe": "offline-compute-pressure"}
    policy, path = compile_to_file(MissionSpec.from_yaml(mission), sc_list, out, require_robust=require_robust, mode_names=names,
                                   min_clean_runs=min_clean_runs, min_clean_seeds=min_clean_seeds, space_size=len(grid_configs()))
    console.print(policy.ladder(), markup=False, highlight=False)  # the ladder's [conditions] are not Rich tags
    for n in policy.notes:
        console.print(f"note: {n}", markup=False, style="dim")
    console.print(f"wrote {path}")


@app.command("runtime")
def runtime_cmd(
    policy: Path = Path("artifacts/resilience-policy.yaml"),
    script: str = typer.Option("healthy:4,offline:6,healthy:8,zombie:5,severe:5", help="sequence of observed conditions with probe counts"),
):
    """Simulate the deterministic runtime against a compiled policy (no video; condition probes only)."""
    from failsafe.policy.runtime import Observed, PolicyRuntime
    from failsafe.policy.schema import ResiliencePolicy

    presets = {
        "healthy": Observed(cloud_reachable=True, cloud_rtt_p95_ms=150),
        "slow": Observed(cloud_reachable=True, cloud_rtt_p95_ms=600),
        "zombie": Observed(cloud_reachable=True, cloud_rtt_p95_ms=6000, cloud_timeout_rate=0.9),
        "offline": Observed(cloud_reachable=False),
        "severe": Observed(cloud_reachable=True, cloud_rtt_p95_ms=150, detector_slowdown=2.8),
    }
    rt = PolicyRuntime(ResiliencePolicy.from_yaml(policy))
    for item in script.split(","):
        name, n = item.split(":")
        for i in range(int(n)):
            d = rt.step(presets[name])
            if d.changed or i == 0:
                mode = d.mode.name if d.mode else "HALT"
                tag = "[green]verified[/green]" if d.verified else "[red]UNVERIFIED[/red]"
                console.print(f"probe {name:8} → {d.condition.label():28} mode={mode:24} {tag}  {d.reason if d.changed else '(' + d.reason + ')'}")
    console.print(f"{len(rt.transitions)} transitions")


@app.command("demo")
def demo_cmd(
    policy: Path = Path("artifacts/resilience-policy.yaml"),
    tier: str = "quick",
    seed: int = 1,
    timeline: str = typer.Option("healthy:0,wan_offline:50,healthy:120", help="scenario:start_s,... (scene seconds)"),
    naive: str = typer.Option("naive_drop_on_fail", help="static config for the no-policy run"),
    mission: Path = Path("missions/restricted-zone.yaml"),
    device: str = "cpu",
    skip_naive: bool = False,
):
    """Real-time end-to-end demo: same corpus and timeline, once without a policy and once with it."""
    from failsafe.demo.timeline import Phase, TimelineDemo, save_report
    from failsafe.experiments.catalog import CONFIGS, load_scenarios
    from failsafe.experiments.runner import calibrate_detector
    from failsafe.mission.schema import MissionSpec
    from failsafe.policy.schema import ResiliencePolicy
    from failsafe.workload.clock import RealTimeClock
    from failsafe.workload.cloud import CloudServerProcess
    from failsafe.workload.detector import make_detector

    m = MissionSpec.from_yaml(mission)
    scen = load_scenarios()
    phases = []
    for item in timeline.split(","):
        name, start = item.split(":")
        phases.append(Phase(float(start), scen[name]))
    pol = ResiliencePolicy.from_yaml(policy)
    det = make_detector("yolo", "yolov8n.pt", device)
    from failsafe.workload.remote_source import RemoteSyntheticSource

    src = RemoteSyntheticSource(tier, seed)  # cameras render in a child process (D-018)
    cloud = CloudServerProcess()  # started before calibration so its idle side-load is in the baseline
    baseline = calibrate_detector(det, src, 640, 0.4)
    console.print(f"detector baseline {baseline:.1f} ms/frame @640; corpus {src.corpus_hash}; timeline {timeline}")
    try:
        runs = [] if skip_naive else [("naive", None, CONFIGS[naive])]
        runs.append(("failsafe", pol, pol.modes[0].config))
        for label, p, cfg in runs:
            console.print(f"\n[bold]running {label} ({cfg.name}) in real time — {src.duration_s:.0f} s[/bold]")
            demo = TimelineDemo(src, m, phases, cfg, det, RealTimeClock(), cloud, policy=p, baseline_ms=baseline)
            rep = demo.run(label)
            console.print(rep.render())
            console.print(f"saved {save_report(rep)}")
    finally:
        cloud.close()
        src.close()


@app.command("rescore")
def rescore_cmd(directory: Path = Path("artifacts/experiments"), dry_run: bool = False):
    """Re-derive alert-based metrics of stored results under the current scoring rule."""
    from failsafe.experiments.rescore import RULE_VERSION, rescore_dir

    console.print(f"scoring rule: {RULE_VERSION}", markup=False)  # the rule contains [brackets]
    for line in rescore_dir(directory, write=not dry_run):
        console.print("  " + line, markup=False, highlight=False)


@app.command("report")
def report_cmd(directory: Path = Path("artifacts/experiments"), out: Path = Path("artifacts/phase1_report.md")):
    """Assemble stored results into a markdown report (matrix, single-knob tradeoffs, scenario effects)."""
    from failsafe.experiments.report import build_report

    md = build_report(directory)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(md)
    console.print(md)
    console.print(f"wrote {out}")


@app.command("sweep")
def sweep(
    mission: Path = Path("missions/restricted-zone.yaml"),
    configs: str = typer.Option("normal,island", help="comma-separated config names, or 'all'"),
    scenarios: str = typer.Option("healthy,wan_offline", help="comma-separated scenario names, or 'all'"),
    tier: str = "quick",
    seed: int = 1,
    out: Path = Path("artifacts/sweeps"),
    name: str = "phase1",
    device: str = "cpu",
):
    """Run a matrix of configs × scenarios sequentially (real time) and write a results table."""
    from failsafe.experiments.sweep import run_sweep

    run_sweep(mission, configs, scenarios, tier, seed, out, name, device, console)


@app.command("campaign")
def campaign_cmd(
    mission: Path = Path("missions/restricted-zone.yaml"),
    configs: str = typer.Option("all", help="comma-separated config names, or 'all'"),
    scenarios: str = typer.Option("wan_zombie", help="comma-separated scenario names, or 'all'"),
    seeds: str = typer.Option("1", help="comma-separated seeds"),
    tier: str = "quick",
    executor: str = typer.Option("local", help="local (sequential) | nebius (parallel jobs)"),
    dry_run: bool = typer.Option(False, help="nebius: print the job commands, submit nothing"),
    device: str = "cpu",
):
    """Run a discovery campaign via an executor. `local` is sequential (timing-valid); `nebius` fans
    each experiment out to an isolated Serverless Job (contention-free parallelism, + GPU metrics)."""
    from failsafe.executors import CampaignJob, LocalExecutor, NebiusJobsExecutor
    from failsafe.experiments.catalog import CONFIGS, load_scenarios

    cfgs = list(CONFIGS) if configs == "all" else configs.split(",")
    scns = list(load_scenarios(Path("scenarios/phase1.yaml"))) if scenarios == "all" else scenarios.split(",")
    sds = [int(s) for s in seeds.split(",")]
    jobs = [CampaignJob(config=c, scenario=s, seed=d, tier=tier, mission=str(mission))
            for c in cfgs for s in scns for d in sds]
    console.print(f"campaign: {len(jobs)} experiments · executor={executor}"
                  + (" · DRY RUN" if dry_run else ""))
    ex = NebiusJobsExecutor(dry_run=dry_run, device=("cuda" if device != "cpu" else "cpu")) \
        if executor == "nebius" else LocalExecutor(device=device)
    results = ex.run_campaign(jobs)
    if results:
        npass = sum(1 for r in results if r.passed)
        console.print(f"done: {len(results)} results, {npass} PASS")


if __name__ == "__main__":
    app()
