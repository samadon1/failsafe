"""Deterministic mock provider: exercises the whole planner loop with no API calls.

It is deliberately *not* clever: it proposes the greedy heuristic's moves for candidates it has
not seen, so tests can check plumbing, schema rejection and replay — not planning quality.
"""

from __future__ import annotations

from failsafe.experiments.schema import OperatingConfig
from failsafe.mission.schema import Invariant, MissionSpec, Objective, Survivability
from failsafe.reasoning.base import Analysis, CandidateProposal, Finding, PlanningContext, ProposalSet
from failsafe.search.objective import REFERENCE


class MockReasoningProvider:
    name = "mock"

    def __init__(self, scripted: list[list[OperatingConfig]] | None = None, invalid_on_round: int | None = None):
        self.scripted = scripted
        self.invalid_on_round = invalid_on_round
        self.calls = 0

    def compile_mission(self, text: str) -> MissionSpec:
        # a fixed, valid spec — the real provider must produce one that validates the same way
        return MissionSpec(
            name="mock-mission",
            description=text[:200],
            invariants=[Invariant(metric="critical_event_recall", min=0.95), Invariant(metric="alert_latency_p95_ms", max=2000)],
            objectives=[Objective(metric="alert_precision", direction="maximize")],
            priorities={"critical_detection": 100, "emergency_alerting": 100, "cloud_confirmation": 50, "historical_indexing": 20, "background_streams": 10},
            degradable_capabilities=["cloud_confirmation", "historical_indexing", "background_stream_fps"],
            survivability=Survivability(wan_outage="required"),
        )

    def propose_configs(self, ctx: PlanningContext) -> ProposalSet:
        self.calls += 1
        if self.invalid_on_round is not None and ctx.round_index == self.invalid_on_round:
            # simulate a malformed model answer: the planner must reject it, not guess
            return ProposalSet.model_construct(hypothesis="", candidates=[], reasoning_summary="malformed")
        if self.scripted is not None:
            batch = self.scripted[min(ctx.round_index, len(self.scripted) - 1)]
            return ProposalSet(
                hypothesis="scripted",
                candidates=[CandidateProposal(config=c, rationale="scripted candidate") for c in batch],
            )
        seen = {o.config.hash for o in ctx.observations}
        from failsafe.search.strategies import greedy_move  # local import to avoid a cycle

        cands: list[CandidateProposal] = []
        if not ctx.observations:
            cands.append(CandidateProposal(config=REFERENCE.model_copy(update={"name": "mock_reference"}), rationale="start from full capability"))
        else:
            # propose the greedy move from every failing observation not yet followed up
            for o in ctx.observations:
                if o.passed:
                    continue
                fake = _FakeResult(o)
                mv = greedy_move(o.config, fake, ctx.mission)
                if mv and mv[0].hash not in seen and all(mv[0].hash != c.config.hash for c in cands):
                    cands.append(CandidateProposal(config=mv[0].model_copy(update={"name": f"mock_r{ctx.round_index}_{len(cands)}"}), rationale=mv[1]))
                if len(cands) >= ctx.max_candidates:
                    break
        if not cands:
            cands.append(CandidateProposal(config=REFERENCE.model_copy(update={"name": "mock_reference"}), rationale="nothing left to propose"))
        return ProposalSet(hypothesis="mock: follow the bottleneck heuristic", candidates=cands)

    def analyze_results(self, ctx: PlanningContext) -> Analysis:
        self.calls += 1
        passed = [o for o in ctx.observations if o.passed]
        return Analysis(
            findings=[Finding(statement=f"{len(passed)} of {len(ctx.observations)} observed candidates satisfied the mission", evidence=[o.config.name for o in passed][:8])]
        )


class _FakeResult:
    """Adapts an Observation to the small surface greedy_move reads from an ExperimentResult."""

    def __init__(self, o):
        from failsafe.experiments.schema import InvariantCheck, VerificationResult

        self._o = o
        checks = []
        if o.recall is not None:
            checks.append(InvariantCheck(metric="critical_event_recall", value=o.recall, operator=">=", threshold=0.95, passed=o.recall >= 0.95))
        if o.p95_latency_ms is not None:
            checks.append(InvariantCheck(metric="alert_latency_p95_ms", value=o.p95_latency_ms, operator="<=", threshold=2000, passed=o.p95_latency_ms <= 2000))
        self.verification = VerificationResult(passed=o.passed, checks=checks, mission_hash="mock")

    def metric_value(self, k: str):
        return {
            "frames_released": self._o.frames_released,
            "frames_processed": None,
            "frames_dropped": self._o.frames_dropped,
            "cloud_rejected": self._o.cloud_rejected,
        }.get(k)
