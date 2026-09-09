"""LLM-guided search: a ReasoningProvider proposes candidates round by round; the deterministic
evaluator (via the experiment cache, or live when allowed) decides. Same trajectory format and
stopping rule as the other strategies, so it is compared on identical terms (D-026).

Guard rails:
  * proposals outside the bounded search space are rejected (recorded, not evaluated);
  * a malformed / non-validating model reply counts as a rejected round; the search stops after
    `max_rejections` consecutive rejections rather than guessing;
  * the model never sees or sets pass/fail.
"""

from __future__ import annotations

from failsafe.experiments.catalog import SEARCH_SPACE
from failsafe.experiments.schema import ExperimentResult, OperatingConfig
from failsafe.reasoning.base import Observation, PlanningContext, ReasoningProvider, RejectedOutput
from failsafe.search.objective import capability_retained
from failsafe.search.strategies import Evaluator, _finish, _new, _record
from failsafe.search.trajectory import SearchTrajectory


def observation(result: ExperimentResult, mission) -> Observation:
    m = result.metric_value
    return Observation(
        config=result.experiment.config,
        passed=bool(result.passed),
        recall=m("critical_event_recall"),
        p95_latency_ms=m("alert_latency_p95_ms"),
        precision=m("alert_precision"),
        frames_dropped=m("frames_dropped"),
        frames_released=m("frames_released"),
        cloud_timeouts=m("cloud_timeouts"),
        cloud_rejected=m("cloud_rejected"),
        detector_ms_mean=m("detector_ms_mean"),
        capability_retained=round(capability_retained(result.experiment.config, mission), 3),
    )


def snap_to_space(cfg: OperatingConfig, space: list[OperatingConfig]) -> OperatingConfig | None:
    """The space member with the same SEARCH_SPACE knob values (non-search fields canonicalised
    to grid defaults), or None if any search knob is out of range. See catalog.snap_to_grid."""
    from failsafe.experiments.catalog import snap_to_grid

    m = snap_to_grid(cfg)
    if m is None:
        return None
    by_hash = {c.hash: c for c in space}
    return by_hash.get(m.hash, m)


class LLMSearch:
    """Nemotron-guided (or mock-guided) search. `name` carries the provider so trajectories from
    different providers are distinguishable."""

    def __init__(self, provider: ReasoningProvider, max_rounds: int = 6, candidates_per_round: int = 4, max_rejections: int = 2, stop_on_first: bool = True):
        self.provider = provider
        self.max_rounds, self.per_round, self.max_rejections, self.stop_on_first = max_rounds, candidates_per_round, max_rejections, stop_on_first
        self.name = f"llm:{provider.name}"

    def run(self, ev: Evaluator, space: list[OperatingConfig]) -> SearchTrajectory:
        traj = _new(self.name, ev)
        results: list[ExperimentResult] = []
        observations: list[Observation] = []
        seen: set[str] = set()
        rejections = 0
        for rnd in range(self.max_rounds):
            ctx = PlanningContext(
                mission=ev.mission, scenario=ev.scenario, space={k: list(v) for k, v in SEARCH_SPACE.items()},
                observations=observations, round_index=rnd, max_candidates=self.per_round,
            )
            try:
                proposal = self.provider.propose_configs(ctx)
                if not proposal.candidates:
                    raise RejectedOutput("empty candidate list")
            except RejectedOutput as e:
                rejections += 1
                traj.notes.append(f"round {rnd}: model output rejected ({e})")
                if rejections >= self.max_rejections:
                    return _finish(traj, ev, results, f"stopped: {rejections} consecutive rejected model outputs")
                continue
            except Exception as e:  # transport errors etc. — never guess
                traj.notes.append(f"round {rnd}: provider error ({type(e).__name__}: {str(e)[:120]})")
                return _finish(traj, ev, results, "stopped: provider error")
            rejections = 0
            traj.notes.append(f"round {rnd} hypothesis: {proposal.hypothesis[:300]}")
            evaluated_any = False
            for cand in proposal.candidates[: self.per_round]:
                member = snap_to_space(cand.config, space)
                if member is None:
                    traj.notes.append(f"round {rnd}: candidate {cand.config.name} outside the search space — rejected")
                    continue
                if member.hash in seen:
                    traj.notes.append(f"round {rnd}: candidate {member.name} already evaluated — skipped")
                    continue
                seen.add(member.hash)
                r, src = ev.evaluate(member, self.name)
                _record(traj, ev, member, r, src, f"round {rnd}: {cand.rationale[:200]}")
                results.append(r)
                observations.append(observation(r, ev.mission))
                evaluated_any = True
                if r.passed and self.stop_on_first:
                    return _finish(traj, ev, results, "first feasible found")
            if not evaluated_any:
                traj.notes.append(f"round {rnd}: no evaluable candidates")
        return _finish(traj, ev, results, "max rounds reached")
