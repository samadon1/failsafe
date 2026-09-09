"""Compare search strategies for one scenario by replay against the cached grid (D-026)."""

from __future__ import annotations

from failsafe.experiments.schema import CorpusRef, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.search.objective import NoVerifiedMode
from failsafe.search.strategies import (
    Evaluator,
    GreedySearch,
    GridSearch,
    MissingResult,
    ResultCache,
    default_space,
    simulate_random,
)


def compare_strategies(mission: MissionSpec, scenario: Scenario, corpus: CorpusRef, random_seeds: int = 1000) -> str:
    cache = ResultCache()
    ev = Evaluator(cache, mission, scenario, corpus)
    space = default_space()
    have = sum(1 for c in space if cache.get(ev.experiment(c).id) is not None)
    lines = [f"# Strategy comparison — scenario `{scenario.name}`", "", f"grid coverage: {have}/{len(space)} configurations measured", ""]
    if have < len(space):
        lines.append("Grid is incomplete: results below are partial and will change as measurements land.")
        lines.append("")
    try:
        grid = GridSearch().run(ev, space)
    except MissingResult:
        # partial grid: evaluate only what exists
        from failsafe.search.strategies import _finish, _new, _record

        traj = _new("grid", ev)
        results = []
        for cfg in space:
            r = cache.get(ev.experiment(cfg).id)
            if r is None:
                continue
            _record(traj, ev, cfg, r, "cached", "enumeration (partial)")
            results.append(r)
        grid = _finish(traj, ev, results, "partial grid")
    greedy = rnd = llm = None
    if have == len(space):
        rnd = simulate_random(ev, space, random_seeds)
        try:
            greedy = GreedySearch().run(ev, space)
        except MissingResult as e:
            lines.append(f"greedy: aborted — {e}")
        from failsafe.reasoning.mock import MockReasoningProvider
        from failsafe.reasoning.planner import LLMSearch

        try:
            llm = LLMSearch(MockReasoningProvider()).run(ev, space)  # nemotron: `failsafe search --strategy llm --provider nemotron`
        except MissingResult as e:
            lines.append(f"llm:mock: aborted — {e}")

    lines.append(f"feasible: {grid.feasible_found}/{grid.evaluations} · Pareto-optimal: {len(grid.pareto)}")
    if grid.no_verified_mode:
        lines.append("")
        lines.append("```")
        lines.append(NoVerifiedMode(**grid.no_verified_mode).render())
        lines.append("```")
    else:
        b = grid.best_config
        lines.append(
            f"best verified mode (grid): `{b.name}` — critical {b.critical_fps} fps, background {b.background_fps} fps, "
            f"{b.detector_resolution} px, cloud {'on/' + str(b.cloud_timeout_ms) + ' ms' if b.cloud_confirmation else 'off'}, "
            f"indexing {'on' if b.historical_indexing else 'off'}; capability retained {grid.best_capability:.2f}"
        )
        best_step = next(s for s in grid.steps if s.experiment_id == grid.best_experiment_id)
        lines.append(f"  recall {best_step.recall:.3f} · p95 {best_step.p95_ms:.0f} ms")
        lines.append("")
        lines.append("Pareto-optimal feasible configurations (capability ↑, CPU ↓, cloud bytes ↓):")
        lines.append("")
        lines.append("| config | capability | edge CPU % | cloud bytes | recall | p95 |")
        lines.append("|---|---|---|---|---|---|")
        for p in sorted(grid.pareto, key=lambda p: -p["capability"]):
            lines.append(f"| {p['config_name']} | {p['capability']:.2f} | {p['cpu_percent']:.0f} | {p['cloud_bytes']:,.0f} | {p['recall']:.3f} | {p['p95_ms']:.0f} ms |")
    lines.append("")
    lines.append("| strategy | evaluations to first feasible | capability retained (found) | capability retained (best in space) |")
    lines.append("|---|---|---|---|")
    best_cap = f"{grid.best_capability:.2f}" if grid.best_capability is not None else "—"
    lines.append(f"| grid | {grid.evaluations} (exhaustive) | {best_cap} | {best_cap} |")
    if rnd is not None:
        if rnd["evaluations_mean"] is None:
            lines.append("| random | none feasible | — | — |")
        else:
            lines.append(f"| random | {rnd['evaluations_mean']:.1f} ± {rnd['evaluations_sd']:.1f} (p90 {rnd['evaluations_p90']:.0f}; analytic {rnd['analytic_mean']:.1f}; {rnd['seeds']} orders) | {rnd['capability_mean']:.2f} (mean of first found) | {best_cap} |")
    if greedy is not None:
        gcap = f"{greedy.best_capability:.2f}" if greedy.best_capability is not None else "none found"
        lines.append(f"| greedy | {greedy.evaluations} ({greedy.termination_reason}) | {gcap} | {best_cap} |")
    if llm is not None:
        lcap = f"{llm.best_capability:.2f}" if llm.best_capability is not None else "none found"
        lines.append(f"| llm:mock | {llm.evaluations} ({llm.termination_reason}) | {lcap} | {best_cap} |")
    for name, traj in (("greedy", greedy), ("llm:mock", llm)):
        if traj is None:
            continue
        lines.append("")
        lines.append(f"{name} trajectory:")
        for s in traj.steps:
            rec = "n/a" if s.recall is None else f"{s.recall:.3f}"
            p95 = "n/a" if s.p95_ms is None else f"{s.p95_ms:.0f} ms"
            lines.append(f"  {s.index}. {s.config.name} → recall {rec}, p95 {p95}, {'PASS' if s.passed else 'FAIL'} — {s.reason}")
    return "\n".join(lines) + "\n"
