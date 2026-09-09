"""Search strategies: grid, random, greedy (deterministic bottleneck heuristic).

All strategies evaluate candidates through one `Evaluator`, which serves stored results from the
experiment cache and — only when explicitly allowed — runs missing experiments live. Strategy
comparison is therefore *replay against the same measured outcomes* (D-026): a strategy's cost is
how many evaluations it needed and what it settled for, under identical evidence. Run-to-run noise
is quantified separately by the repeatability runs.
"""

from __future__ import annotations

import hashlib
import statistics
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol

import numpy as np

from failsafe.experiments.catalog import SEARCH_SPACE, grid_configs
from failsafe.experiments.runner import LocalRunner, result_files
from failsafe.experiments.schema import CorpusRef, Experiment, ExperimentResult, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.search.objective import (
    REFERENCE,
    best_of,
    capability_retained,
    frontier,
    margin_tier,
    no_verified_mode,
    rank_key,
)
from failsafe.search.trajectory import SearchStep, SearchTrajectory, Source, to_jsonable


class MissingResult(Exception):
    pass


# ---------------------------------------------------------------------------------------------
# Cache + evaluator
# ---------------------------------------------------------------------------------------------


class ResultCache:
    def __init__(self, experiments_dir: Path = Path("artifacts/experiments")):
        self.dir = experiments_dir
        self._by_id: dict[str, list[ExperimentResult]] = {}
        for p in result_files(experiments_dir):
            r = ExperimentResult.from_json(p.read_text())
            self._by_id.setdefault(r.experiment.id, []).append(r)
        for v in self._by_id.values():
            v.sort(key=lambda r: r.provenance.started_at)

    def runs(self, experiment_id: str) -> list[ExperimentResult]:
        return self._by_id.get(experiment_id, [])

    def get(self, experiment_id: str, repeat: int = 0) -> ExperimentResult | None:
        """The repeat-th run, preferring runs not flagged as contended (D-028)."""
        runs = self.runs(experiment_id)
        clean = [r for r in runs if not (r.metric_value("qc_suspect_contention") or 0)]
        ordered = clean + [r for r in runs if r not in clean]
        return ordered[repeat] if repeat < len(ordered) else None

    def add(self, result: ExperimentResult) -> None:
        self._by_id.setdefault(result.experiment.id, []).append(result)


class Evaluator:
    def __init__(
        self,
        cache: ResultCache,
        mission: MissionSpec,
        scenario: Scenario,
        corpus: CorpusRef = CorpusRef(tier="quick", seed=1),
        runner: LocalRunner | None = None,
        allow_live: bool = False,
        repeat: int = 0,
    ):
        self.cache, self.mission, self.scenario, self.corpus = cache, mission, scenario, corpus
        self.runner, self.allow_live, self.repeat = runner, allow_live, repeat
        self.live_runs = 0

    def experiment(self, cfg: OperatingConfig, proposed_by: str = "search") -> Experiment:
        return Experiment(mission=self.mission, scenario=self.scenario, config=cfg, corpus=self.corpus, proposed_by=proposed_by)

    def evaluate(self, cfg: OperatingConfig, proposed_by: str = "search") -> tuple[ExperimentResult, Source]:
        exp = self.experiment(cfg, proposed_by)
        r = self.cache.get(exp.id, self.repeat)
        if r is not None:
            return r, "cached"
        if not (self.allow_live and self.runner is not None):
            raise MissingResult(f"no stored result for {cfg.name} × {self.scenario.name} ({exp.id}); run the grid or pass --live")
        r = self.runner.run(exp)
        self.cache.add(r)
        self.live_runs += 1
        return r, "live"


# ---------------------------------------------------------------------------------------------
# Trajectory helpers
# ---------------------------------------------------------------------------------------------


def _search_id(strategy: str, scenario: str, mission: MissionSpec, extra: str = "") -> str:
    s = f"{strategy}|{scenario}|{mission.hash}|{extra}|{datetime.now(UTC).isoformat()}"
    return hashlib.sha256(s.encode()).hexdigest()[:12]


def _record(traj: SearchTrajectory, ev: Evaluator, cfg: OperatingConfig, result: ExperimentResult, source: Source, reason: str) -> SearchStep:
    step = SearchStep(
        index=len(traj.steps) + 1,
        config=cfg,
        experiment_id=result.experiment.id,
        source=source,
        passed=bool(result.passed),
        recall=result.metric_value("critical_event_recall"),
        p95_ms=result.metric_value("alert_latency_p95_ms"),
        frames_dropped=result.metric_value("frames_dropped"),
        capability=round(capability_retained(cfg, ev.mission), 3),
        rank=rank_key(result, ev.mission).as_tuple(),
        tier=margin_tier(result, ev.mission),
        suspect=bool(result.metric_value("qc_suspect_contention") or 0),
        reason=reason,
    )
    traj.steps.append(step)
    traj.evaluations += 1
    if step.passed:
        traj.feasible_found += 1
    return step


def _finish(traj: SearchTrajectory, ev: Evaluator, results: list[ExperimentResult], reason: str) -> SearchTrajectory:
    traj.termination_reason = reason
    best = best_of(results, ev.mission)
    if best is not None:
        # report the candidate as the strategy named it (stored results may carry legacy names)
        by_hash = {st.config.hash: st.config for st in traj.steps}
        traj.best_config = by_hash.get(best.experiment.config.hash, best.experiment.config)
        traj.best_experiment_id = best.experiment.id
        traj.best_capability = round(capability_retained(best.experiment.config, ev.mission), 3)
        _, pareto = frontier(results, ev.mission, names={h: c.name for h, c in by_hash.items()})
        traj.pareto = [to_jsonable(p.__dict__) for p in pareto]
    else:
        traj.no_verified_mode = to_jsonable(no_verified_mode(results, ev.mission, ev.scenario.name).__dict__)
    traj.finished_at = datetime.now(UTC)
    return traj


def _new(strategy: str, ev: Evaluator, extra: str = "") -> SearchTrajectory:
    return SearchTrajectory(
        search_id=_search_id(strategy, ev.scenario.name, ev.mission, extra),
        strategy=strategy,
        scenario=ev.scenario.name,
        mission_hash=ev.mission.hash,
        corpus=f"{ev.corpus.tier}/seed{ev.corpus.seed}",
    )


# ---------------------------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------------------------


class Strategy(Protocol):
    name: str

    def run(self, ev: Evaluator, space: list[OperatingConfig]) -> SearchTrajectory: ...


class GridSearch:
    """Exhaustive: evaluate every configuration; ground truth for what is achievable."""

    name = "grid"

    def run(self, ev: Evaluator, space: list[OperatingConfig]) -> SearchTrajectory:
        traj = _new(self.name, ev)
        results = []
        for cfg in space:
            r, src = ev.evaluate(cfg, "grid")
            _record(traj, ev, cfg, r, src, "enumeration")
            results.append(r)
        return _finish(traj, ev, results, "space exhausted")


class RandomSearch:
    """Uniform random order without replacement; stops at the first feasible configuration."""

    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def run(self, ev: Evaluator, space: list[OperatingConfig]) -> SearchTrajectory:
        traj = _new(self.name, ev, f"seed{self.seed}")
        traj.notes.append(f"seed={self.seed}")
        rng = np.random.default_rng(self.seed)
        results = []
        for i in rng.permutation(len(space)):
            cfg = space[int(i)]
            r, src = ev.evaluate(cfg, "random")
            _record(traj, ev, cfg, r, src, "random draw")
            results.append(r)
            if r.passed:
                return _finish(traj, ev, results, "first feasible found")
        return _finish(traj, ev, results, "space exhausted, none feasible")


def _next_lower(values: list, current):
    lower = [v for v in values if v < current]
    return max(lower) if lower else None


def _next_higher(values: list, current):
    higher = [v for v in values if v > current]
    return min(higher) if higher else None


TIMEOUTS = sorted(int(c.split("_")[1]) for c in SEARCH_SPACE["cloud"] if c != "off")


def greedy_move(cfg: OperatingConfig, result: ExperimentResult, mission: MissionSpec) -> tuple[OperatingConfig, str] | None:
    """One deterministic degradation step aimed at the diagnosed bottleneck, cheapest first.

    Disabling the cloud resets the timeout to the space's canonical 3000 ms (the knob is inert
    with the cloud off; leaving it would hash outside the bounded space).

    Diagnosis uses only measured signals: which invariant failed, the dropped-frame fraction
    (overload/backpressure, D-027), and cloud rejections. Returns None when no move is left.
    """
    checks = result.verification.checks if result.verification else []
    latency_fail = any(c.metric.startswith("alert_latency") and not c.passed for c in checks)
    recall_fail = any(c.metric == "critical_event_recall" and not c.passed for c in checks)
    released = result.metric_value("frames_released") or ((result.metric_value("frames_processed") or 0) + (result.metric_value("frames_dropped") or 0))
    drop_frac = (result.metric_value("frames_dropped") or 0.0) / max(1.0, released)
    overloaded = drop_frac > 0.03
    bg_values, cf_values, res_values = SEARCH_SPACE["background_fps"], SEARCH_SPACE["critical_fps"], SEARCH_SPACE["detector_resolution"]

    def upd(**kw):
        return cfg.model_copy(update=kw)

    # 1. latency violated while waiting on the cloud → fail over sooner, then drop the cloud path
    if latency_fail and cfg.cloud_confirmation:
        lower = _next_lower(TIMEOUTS, cfg.cloud_timeout_ms)
        if lower is not None:
            return upd(cloud_timeout_ms=lower), f"latency violated with cloud on → cloud_timeout {cfg.cloud_timeout_ms}→{lower} ms"
        return upd(cloud_confirmation=False, cloud_timeout_ms=3000), "latency violated at minimum timeout → cloud_confirmation off"
    # 2. overload: demand exceeds capacity → shed the cheapest load first
    if overloaded:
        if cfg.historical_indexing:
            return upd(historical_indexing=False), f"overloaded ({drop_frac:.0%} dropped) → historical_indexing off"
        lower = _next_lower(bg_values, cfg.background_fps)
        if lower is not None:
            return upd(background_fps=lower), f"overloaded ({drop_frac:.0%} dropped) → background_fps {cfg.background_fps}→{lower}"
        lower = _next_lower(cf_values, cfg.critical_fps)
        if lower is not None:
            return upd(critical_fps=lower), f"overloaded ({drop_frac:.0%} dropped) → critical_fps {cfg.critical_fps}→{lower}"
        lower = _next_lower(res_values, cfg.detector_resolution)
        if lower is not None:
            return upd(detector_resolution=lower), f"overloaded ({drop_frac:.0%} dropped) → detector_resolution {cfg.detector_resolution}→{lower}"
        return None
    # 3. recall violated without overload
    if recall_fail:
        if cfg.cloud_confirmation:
            return upd(cloud_confirmation=False, cloud_timeout_ms=3000), "recall violated with cloud on (rejections) → cloud_confirmation off"
        higher = _next_higher(res_values, cfg.detector_resolution)
        if higher is not None:
            return upd(detector_resolution=higher), f"recall violated → detector_resolution {cfg.detector_resolution}→{higher}"
        if cfg.historical_indexing:
            return upd(historical_indexing=False), "recall violated → historical_indexing off (free capacity)"
        return None
    # 4. latency violated with cloud already off → reduce load
    if latency_fail:
        if cfg.historical_indexing:
            return upd(historical_indexing=False), "latency violated locally → historical_indexing off"
        lower = _next_lower(bg_values, cfg.background_fps)
        if lower is not None:
            return upd(background_fps=lower), f"latency violated locally → background_fps {cfg.background_fps}→{lower}"
        return None
    return None


class GreedySearch:
    """Start from the highest-capability configuration; while the mission fails, apply the
    cheapest degradation expected to relieve the diagnosed bottleneck; stop at the first pass."""

    name = "greedy"

    def __init__(self, start: OperatingConfig = REFERENCE, max_steps: int = 20):
        self.start, self.max_steps = start, max_steps

    def run(self, ev: Evaluator, space: list[OperatingConfig]) -> SearchTrajectory:
        traj = _new(self.name, ev)
        by_hash = {c.hash: c for c in space}
        cfg = by_hash.get(self.start.hash, self.start)
        results, seen, reason = [], set(), "start at highest-capability configuration"
        for _ in range(self.max_steps):
            if cfg.hash in seen:
                return _finish(traj, ev, results, "cycle: candidate already evaluated")
            seen.add(cfg.hash)
            r, src = ev.evaluate(by_hash.get(cfg.hash, cfg), "greedy")
            _record(traj, ev, by_hash.get(cfg.hash, cfg), r, src, reason)
            results.append(r)
            if r.passed:
                return _finish(traj, ev, results, "first feasible found")
            move = greedy_move(cfg, r, ev.mission)
            if move is None:
                return _finish(traj, ev, results, "no degradation move left")
            cfg, reason = move
        return _finish(traj, ev, results, "max steps reached")


STRATEGIES = {"grid": GridSearch, "random": RandomSearch, "greedy": GreedySearch}


# ---------------------------------------------------------------------------------------------
# Random-search statistics by replay (no new experiments)
# ---------------------------------------------------------------------------------------------


def simulate_random(ev: Evaluator, space: list[OperatingConfig], n_seeds: int = 1000) -> dict:
    """Distribution of evaluations-to-first-feasible and capability of the config found, over
    many random orders, using cached outcomes only. Also the analytic expectation (N+1)/(k+1)."""
    outcomes = []
    for cfg in space:
        r, _ = ev.evaluate(cfg, "random-sim")
        outcomes.append((bool(r.passed), capability_retained(cfg, ev.mission)))
    n = len(outcomes)
    k = sum(1 for p, _ in outcomes if p)
    if k == 0:
        return {"n": n, "feasible": 0, "evaluations_mean": None, "evaluations_sd": None, "capability_mean": None, "analytic_mean": None}
    rng = np.random.default_rng(0)
    evals, caps = [], []
    for _ in range(n_seeds):
        order = rng.permutation(n)
        for i, j in enumerate(order, 1):
            if outcomes[j][0]:
                evals.append(i)
                caps.append(outcomes[j][1])
                break
    return {
        "n": n,
        "feasible": k,
        "evaluations_mean": statistics.fmean(evals),
        "evaluations_sd": statistics.pstdev(evals),
        "evaluations_p90": float(np.percentile(evals, 90)),
        "capability_mean": statistics.fmean(caps),
        "analytic_mean": (n + 1) / (k + 1),
        "seeds": n_seeds,
    }


def default_space() -> list[OperatingConfig]:
    return grid_configs()
