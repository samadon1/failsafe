"""Compile verified experiment results into a ResiliencePolicy.

Admission (D-037). An experiment (one configuration under one scenario) is *verified* only when
it has at least `min_clean_runs` runs that are not flagged as contended and EVERY clean run passes
the mission invariants. Its tier is that of the worst clean run (robust only if every clean run
clears the noise floor), and the recall / p95 / precision it records are the worst observed across
clean runs. A single passing run is evidence, not verification: the scenario is listed as
`insufficient_evidence` with the candidate to repeat. Scenarios with no stored result are
`untested`; scenarios where every candidate failed a clean run are `refuted` (and `exhaustive`
when the whole search space was tried).

For every scenario, the admitted experiment with the best lexicographic rank (robust > marginal;
then capability; then quality; then cost) becomes the mode. The fallback is the most conservative
admitted configuration across all scenarios — no cloud wait first, then least demand — used
fail-closed for unverified conditions and labelled as such. Nothing in here consults an LLM.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from failsafe.experiments.schema import ExperimentResult, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.policy.schema import Admission, Conditions, Mode, ResiliencePolicy, UnverifiedCondition, Verification
from failsafe.search.objective import REFERENCE, capability_retained, capability_terms, margin_tier, no_verified_mode, rank_key
from failsafe.search.strategies import ResultCache

DEFAULT_MIN_CLEAN_RUNS = 2
DEFAULT_MIN_CLEAN_SEEDS = 2
ADMISSION_RULE = "every clean run passes, across >=2 corpus seeds; tier = worst clean run; recall/p95/precision = worst case over clean runs"


def conditions_for(sc: Scenario) -> Conditions:
    return Conditions(cloud_state=[sc.cloud_state], compute_pressure=[sc.compute_pressure], bandwidth_mbps_max=sc.bandwidth_mbps)


def conservativeness_key(cfg: OperatingConfig, capability: float) -> tuple:
    """Lower = more conservative (D-037). A fallback serves conditions nobody verified, so the
    order is: never block on the network (cloud confirmation off first, else the shortest
    timeout), then least detector demand, then smallest input, then indexing off, then least
    capability, then the config hash so ties are deterministic."""
    demand = cfg.critical_fps + 3 * cfg.background_fps
    return (
        1 if cfg.cloud_confirmation else 0,
        cfg.cloud_timeout_ms if cfg.cloud_confirmation else 0,
        demand,
        cfg.detector_resolution,
        1 if cfg.historical_indexing else 0,
        round(capability, 3),
        cfg.hash,
    )


def sacrifices(cfg: OperatingConfig, ref: OperatingConfig = REFERENCE) -> list[str]:
    out = []
    t, tr = capability_terms(cfg, ref), capability_terms(ref, ref)
    if cfg.cloud_confirmation != ref.cloud_confirmation:
        out.append("cloud confirmation off" if not cfg.cloud_confirmation else "cloud confirmation on")
    elif cfg.cloud_confirmation and cfg.cloud_timeout_ms != ref.cloud_timeout_ms:
        out.append(f"cloud timeout {ref.cloud_timeout_ms}→{cfg.cloud_timeout_ms} ms")
    if cfg.historical_indexing != ref.historical_indexing:
        out.append("historical indexing off")
    if t["background_streams"] < tr["background_streams"]:
        out.append(f"background streams {cfg.background_fps} fps" + (" (some cameras dropped)" if cfg.drop_background_streams != "none" or cfg.background_fps == 0 else ""))
    if cfg.critical_fps != ref.critical_fps:
        out.append(f"critical camera {ref.critical_fps}→{cfg.critical_fps} fps")
    if cfg.detector_resolution != ref.detector_resolution:
        out.append(f"detector input {ref.detector_resolution}→{cfg.detector_resolution} px")
    return out


# ---------------------------------------------------------------------------------------------
# Evidence: what all the stored runs of one experiment say, taken together
# ---------------------------------------------------------------------------------------------


def _is_clean(r: ExperimentResult) -> bool:
    return not (r.metric_value("qc_suspect_contention") or 0)


def evidence_scope(exp) -> str:
    """The part of a scenario a configuration can actually observe (D-047). A cloud-off
    configuration has no network injector, no confirmer and no bandwidth path, so every cloud state
    and bandwidth cap is the same experiment for it: its runs pool across those scenarios and are
    keyed by compute pressure alone. A cloud-on configuration is keyed by the full scenario."""
    if exp.config.cloud_confirmation:
        return exp.scenario.name
    return f"compute:{exp.scenario.compute_pressure.value}"


def scope_for(sc: Scenario, cloud_on: bool) -> str:
    return sc.name if cloud_on else f"compute:{sc.compute_pressure.value}"


@dataclass
class Evidence:
    """Every stored run of one configuration under one scenario, pooled across corpus seeds and
    repeats: a pass on each of three seeds is three clean runs (the seed-generalization data is
    evidence, not a separate study)."""

    key: str  # config hash
    runs: list[ExperimentResult]  # every stored run
    clean: list[ExperimentResult]  # runs not flagged as contended (D-028); admission is decided on these
    worst: ExperimentResult | None  # the clean run with the lowest tier, then lowest recall, then highest p95
    tier: int  # 0 fail · 1 marginal · 2 robust — over the clean runs, i.e. the worst one
    all_clean_pass: bool

    @property
    def config(self) -> OperatingConfig:
        return self.runs[0].experiment.config

    @property
    def scenario(self) -> Scenario:
        return self.runs[0].experiment.scenario

    @property
    def experiment_ids(self) -> list[str]:
        return sorted({r.experiment.id for r in self.runs})

    @property
    def seeds(self) -> list[int]:
        return sorted({r.experiment.corpus.seed for r in self.runs})

    @property
    def clean_seeds(self) -> list[int]:
        return sorted({r.experiment.corpus.seed for r in self.clean})

    @property
    def rep_id(self) -> str:
        """The experiment the recorded worst case came from."""
        return (self.worst or self.runs[0]).experiment.id

    def admitted(self, min_clean_runs: int, require_robust: bool, min_clean_seeds: int = 1) -> bool:
        return (len(self.clean) >= min_clean_runs and len(self.clean_seeds) >= min_clean_seeds
                and self.all_clean_pass and (self.tier == 2 or not require_robust))


def evidence_for(key: str, runs: list[ExperimentResult], mission: MissionSpec) -> Evidence:
    clean = [r for r in runs if _is_clean(r)]
    if not clean:
        return Evidence(key, runs, [], None, 0, False)

    def badness(r: ExperimentResult) -> tuple:
        rec = r.metric_value("critical_event_recall")
        p95 = r.metric_value("alert_latency_p95_ms")
        return (margin_tier(r, mission), rec if rec is not None else -1.0, -(p95 if p95 is not None else float("inf")))

    worst = min(clean, key=badness)
    return Evidence(key, runs, clean, worst, min(margin_tier(r, mission) for r in clean), all(r.passed for r in clean))


def _worst_case(ev: Evidence) -> tuple[float | None, float | None, float | None]:
    def vals(k: str) -> list[float]:
        return [v for v in (r.metric_value(k) for r in ev.clean) if v is not None]

    rec, p95, prec = vals("critical_event_recall"), vals("alert_latency_p95_ms"), vals("alert_precision")
    return (min(rec) if rec else None, max(p95) if p95 else None, min(prec) if prec else None)


def _verification(ev: Evidence, mission: MissionSpec, scenario_name: str | None = None) -> Verification:
    rec, p95, prec = _worst_case(ev)
    rep = ev.worst or ev.runs[0]
    return Verification(
        experiment_id=ev.rep_id,
        experiment_ids=ev.experiment_ids,
        seeds=ev.seeds,
        scenario=scenario_name or ev.scenario.name,
        recall=rec,
        p95_latency_ms=p95,
        precision=prec,
        tier="robust" if ev.tier == 2 else "marginal",
        runs=len(ev.runs),
        clean_runs=len(ev.clean),
        pass_rate=sum(1 for r in ev.runs if r.passed) / max(1, len(ev.runs)),
        clean_pass_rate=sum(1 for r in ev.clean if r.passed) / max(1, len(ev.clean)),
        corpus_hash=rep.provenance.corpus_hash,
        git_sha=rep.provenance.git_sha,
        scoring_rule=rep.provenance.extra.get("scoring_rule", ""),
    )


def _mode(ev: Evidence, name: str, conditions: Conditions, mission: MissionSpec, scenario_name: str | None = None) -> Mode:
    cfg = ev.config
    return Mode(
        name=name,
        conditions=conditions,
        config=cfg.model_copy(update={"name": name}),
        capability_retained=round(capability_retained(cfg, mission), 3),
        verification=_verification(ev, mission, scenario_name),
        sacrifices=sacrifices(cfg),
    )


def _rank(ev: Evidence, mission: MissionSpec) -> tuple:
    return rank_key(ev.worst, mission).as_tuple()


def _required(mission: MissionSpec) -> dict[str, str]:
    return {inv.metric: (f">= {inv.min}" if inv.min is not None else f"<= {inv.max}") for inv in mission.invariants}


# ---------------------------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------------------------


def compile_policy(
    mission: MissionSpec,
    scenarios: list[Scenario],
    cache: ResultCache,
    corpus_label: str | None = None,
    require_robust: bool = False,
    mode_names: dict[str, str] | None = None,
    min_clean_runs: int = DEFAULT_MIN_CLEAN_RUNS,
    min_clean_seeds: int = DEFAULT_MIN_CLEAN_SEEDS,
    space_size: int | None = None,
) -> ResiliencePolicy:
    """Build the policy from all stored results for the given scenarios.

    `min_clean_runs`  how many un-contended runs a configuration × scenario needs (pooled across
                      corpus seeds and repeats) before it can be verified.
    `min_clean_seeds` how many DISTINCT corpus seeds those clean runs must span, so a mode cannot
                      be "verified" on a single seed's luck (D-048).
    `require_robust`  additionally refuse marginal passes (inside the noise floor).
    `space_size`      size of the search space, so a refuted scenario can be marked exhaustive.
    `mode_names`      optionally maps scenario name → mode name (else the scenario name is used).
    `corpus_label`    overrides the policy's corpus field (default: derived, e.g. "quick/seeds 1,2,3").
    """
    names = mode_names or {}
    pooled: dict[tuple[str, str], list[ExperimentResult]] = defaultdict(list)
    considered: list[str] = []
    corpora: set[tuple[str, int]] = set()
    for eid, runs in cache._by_id.items():
        r0 = runs[0]
        if r0.experiment.mission.hash != mission.hash or r0.experiment.corpus.kind != "synthetic":
            continue
        # pool by what the configuration can observe: cloud-on configs per scenario, cloud-off
        # configs across every cloud state at the same compute pressure (D-047)
        pooled[(r0.experiment.config.hash, evidence_scope(r0.experiment))].extend(runs)
        corpora.add((r0.experiment.corpus.tier, r0.experiment.corpus.seed))
        considered.append(eid)
    groups: dict[tuple[str, str], Evidence] = {k: evidence_for(k[0], runs, mission) for k, runs in pooled.items()}

    def evidence_under(sc: Scenario) -> list[Evidence]:
        return [ev for (chash, scope), ev in groups.items()
                if scope == scope_for(sc, cloud_on=ev.config.cloud_confirmation)]
    if corpus_label is None:
        tiers = sorted({t for t, _ in corpora})
        seeds = sorted({s for _, s in corpora})
        corpus_label = f"{'/'.join(tiers) or 'none'}/seeds {','.join(map(str, seeds)) or '-'}"

    modes: list[Mode] = []
    unverified: list[UnverifiedCondition] = []
    notes: list[str] = []
    admitted_all: list[Evidence] = []
    for sc in scenarios:
        evs = evidence_under(sc)
        if not evs:
            unverified.append(UnverifiedCondition(scenario=sc.name, conditions=conditions_for(sc), reason="untested",
                                                  candidates_tested=0, ceilings={}, required=_required(mission)))
            continue
        admitted = [e for e in evs if e.admitted(min_clean_runs, require_robust, min_clean_seeds)]
        if admitted:
            best = max(admitted, key=lambda e: _rank(e, mission))
            admitted_all.extend(a for a in admitted if a not in admitted_all)
            modes.append(_mode(best, names.get(sc.name, sc.name), conditions_for(sc), mission, sc.name))
            continue

        # No verified mode: say precisely why, and what to repeat. Ceilings are upper bounds over
        # every run (a contended run cannot make a ceiling look better than a clean one would).
        nvm = no_verified_mode([r for e in evs for r in e.runs], mission, sc.name)
        all_pass = [e for e in evs if e.clean and e.all_clean_pass]
        # passes every clean run, blocked only by require_robust
        marginal = [e for e in all_pass if len(e.clean) >= min_clean_runs and len(e.clean_seeds) >= min_clean_seeds]
        # passes, but too few clean runs or too few distinct seeds (D-048)
        thin = [e for e in all_pass if e not in marginal]
        if marginal:
            reason, cand = "marginal", max(marginal, key=lambda e: _rank(e, mission))
        elif thin:
            reason, cand = "insufficient_evidence", max(thin, key=lambda e: _rank(e, mission))
        else:
            reason, cand = "refuted", None
        unverified.append(
            UnverifiedCondition(
                scenario=sc.name,
                conditions=conditions_for(sc),
                reason=reason,
                candidates_tested=len(evs),
                exhaustive=space_size is not None and len(evs) >= space_size,
                best_candidate=cand.rep_id if cand else None,
                best_candidate_config=cand.config.name if cand else None,
                best_candidate_clean_runs=len(cand.clean) if cand else 0,
                ceilings=nvm.ceilings,
                required=nvm.required,
            )
        )

    modes.sort(key=lambda m: -m.capability_retained)

    # Fallback: the most conservative configuration admitted under ANY scenario. If that
    # configuration is already one of the compiled modes, reuse that mode (same name).
    fallback: Mode | None = None
    if admitted_all:
        fb = min(admitted_all, key=lambda e: conservativeness_key(e.config, capability_retained(e.config, mission)))
        same = [e for e in admitted_all if e.config.hash == fb.config.hash]
        verified_under = sorted({r.experiment.scenario.name for e in same for r in e.runs})
        fb = next((e for e in same if "healthy" in {r.experiment.scenario.name for r in e.runs}), same[0])
        fallback = next((m for m in modes if m.config.hash == fb.config.hash), None) or _mode(fb, fb.config.name, conditions_for(fb.scenario), mission)
        if unverified:
            notes.append(
                f"fallback for unverified conditions is `{fallback.name}` (verified under {', '.join(verified_under)} only); "
                "the runtime reports NO VERIFIED MODE when it is used outside those conditions"
            )
    return ResiliencePolicy(
        mission_name=mission.name,
        mission_hash=mission.hash,
        corpus=corpus_label,
        compiled_from=sorted(considered),
        admission=Admission(min_clean_runs=min_clean_runs, min_clean_seeds=min_clean_seeds, require_robust=require_robust, rule=ADMISSION_RULE),
        modes=modes,
        unverified=unverified,
        fallback=fallback,
        notes=notes,
    )


def compile_to_file(mission: MissionSpec, scenarios: list[Scenario], out: Path, **kw) -> tuple[ResiliencePolicy, Path]:
    policy = compile_policy(mission, scenarios, ResultCache(), **kw)
    return policy, policy.to_yaml(out)
