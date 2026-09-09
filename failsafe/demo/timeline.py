"""Timeline demo: NORMAL → cut WAN → (policy reacts) → event → WAN restored → NORMAL.

    conditions timeline  ──►  NetworkInjector.set_scenario()  (the world changes)
                                        │
    pipeline (live)  ◄── apply_config ── PolicyRuntime.step(Prober.observe())  every probe_interval
                                        │
                              transitions + per-phase outcomes → DemoReport

Two runs make the story: `policy=None` (the naive system keeps its static configuration) and
`policy=<compiled>` (Failsafe switches to verified modes). Same corpus, same timeline, same
measured metrics. Nothing here fabricates a number.
"""

from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from pathlib import Path

from failsafe.corpus.ground_truth import GroundTruthEvent
from failsafe.experiments.evaluator import verify
from failsafe.experiments.schema import AlertRecord, OperatingConfig, Scenario
from failsafe.mission.schema import MissionSpec
from failsafe.policy.prober import Prober
from failsafe.policy.runtime import Decision, PolicyRuntime
from failsafe.policy.schema import ResiliencePolicy
from failsafe.workload.clock import Clock
from failsafe.workload.cloud import ConfirmationService
from failsafe.workload.detector import Detector
from failsafe.workload.metrics import ResourceSamples, compute_metrics
from failsafe.workload.pipeline import NetworkInjector, Pipeline
from failsafe.workload.source import FrameSource


@dataclass
class Phase:
    """A stretch of scene time during which the world is in one condition."""

    start_t: float
    scenario: Scenario

    @property
    def name(self) -> str:
        return self.scenario.name


@dataclass
class PhaseOutcome:
    name: str
    start_t: float
    end_t: float
    events: int
    detected: int
    latencies_ms: list[float]
    alerts: int
    false_alerts: int
    active_modes: list[str]

    @property
    def recall(self) -> float | None:
        return None if self.events == 0 else self.detected / self.events

    @property
    def p95_ms(self) -> float | None:
        if not self.latencies_ms:
            return None
        s = sorted(self.latencies_ms)
        return s[min(len(s) - 1, int(0.95 * (len(s) - 1)))]


@dataclass
class DemoReport:
    label: str
    policy_used: bool
    phases: list[PhaseOutcome]
    transitions: list[dict]
    config_changes: list[tuple[float, str]]
    overall: dict
    verification_passed: bool
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(
            {
                "label": self.label,
                "policy_used": self.policy_used,
                "phases": [
                    {**p.__dict__, "recall": p.recall, "p95_ms": p.p95_ms} for p in self.phases
                ],
                "transitions": self.transitions,
                "config_changes": self.config_changes,
                "overall": self.overall,
                "verification_passed": self.verification_passed,
                "notes": self.notes,
            },
            indent=2,
            default=str,
        )

    def render(self) -> str:
        lines = [f"== {self.label} ({'with policy' if self.policy_used else 'no policy'})"]
        for p in self.phases:
            rec = "n/a" if p.recall is None else f"{p.recall:.3f} ({p.detected}/{p.events})"
            p95 = "n/a" if p.p95_ms is None else f"{p.p95_ms:.0f} ms"
            lines.append(f"  {p.start_t:6.1f}–{p.end_t:6.1f}s {p.name:18} recall {rec:16} p95 {p95:9} false alerts {p.false_alerts}  modes {'/'.join(p.active_modes) or '-'}")
        for t in self.transitions:
            lines.append(f"  → t={t['scene_t']:6.1f}s {t['from'] or '-'} → {t['to']}  [{t['condition']}] {'verified' if t['verified'] else 'UNVERIFIED'}: {t['reason']}")
        o = self.overall
        lines.append(f"  overall recall {o['critical_event_recall']:.3f}, p95 {o['alert_latency_p95_ms']:.0f} ms → {'PASS' if self.verification_passed else 'FAIL'}")
        return "\n".join(lines)


class TimelineDemo:
    def __init__(
        self,
        source: FrameSource,
        mission: MissionSpec,
        phases: list[Phase],
        initial_config: OperatingConfig,
        detector: Detector,
        clock: Clock,
        confirmer: ConfirmationService | None,
        policy: ResiliencePolicy | None = None,
        probe_interval_s: float = 1.0,
        baseline_ms: float = 40.0,
        degrade_after: int = 3,
        recover_after: int = 5,
        stream: bool = False,
        ping_timeout_s: float = 1.5,
    ):
        self.source, self.mission, self.phases = source, mission, sorted(phases, key=lambda p: p.start_t)
        self.policy = policy
        self.injector = NetworkInjector(self.phases[0].scenario)
        self.injector.clock_fn = clock.now  # rolling windows follow the pipeline clock (simulated or real)
        self.pipe = Pipeline(
            source, initial_config, self.phases[0].scenario, detector, clock,
            confirmer=confirmer, injector=self.injector, time_scale=1.0, stream=stream, on_tick=self._tick,
        )
        self.clock = clock
        self.prober = Prober(self.injector, self.pipe.detector_ms, baseline_ms, ping_timeout_s=ping_timeout_s)
        self.runtime = PolicyRuntime(policy, degrade_after=degrade_after, recover_after=recover_after) if policy else None
        self.probe_interval_s = probe_interval_s
        self._next_probe_t = 0.0
        self._phase_idx = 0
        self.transitions: list[dict] = []
        self.decisions: list[tuple[float, Decision]] = []
        self._lock = threading.Lock()

    # driver callback: advance the world and let the runtime look at it
    def _tick(self, scene_t: float) -> None:
        while self._phase_idx + 1 < len(self.phases) and scene_t >= self.phases[self._phase_idx + 1].start_t:
            self._phase_idx += 1
            self.injector.set_scenario(self.phases[self._phase_idx].scenario)
        if scene_t < self._next_probe_t:
            return
        self._next_probe_t = scene_t + self.probe_interval_s
        self.prober.maybe_ping(self.clock.now(), wait=not self.clock.is_real_time)
        if self.runtime is None:
            return
        if len(self.pipe.detector_ms) < 40:
            return  # warm-up: the slowdown estimate is not stable yet — do not act on it
        obs = self.prober.observe()
        if obs is None:
            return  # no cloud evidence yet (startup) — do not act on ignorance
        d = self.runtime.step(obs)
        self.decisions.append((scene_t, d))
        if d.changed:
            self.transitions.append({
                "scene_t": round(scene_t, 2), "from": self.runtime.transitions[-1].from_mode, "to": d.mode.name if d.mode else None,
                "condition": d.condition.label(), "verified": d.verified, "reason": d.reason,
            })
            if d.mode is not None:
                self.pipe.apply_config(d.mode.config)

    def run(self, label: str) -> DemoReport:
        out = self.pipe.run()
        gt = self.source.ground_truth
        metrics, alerts = compute_metrics(
            out.alerts, gt, real_time=self.clock.is_real_time, frames_processed=out.frames_processed, frames_dropped=out.frames_dropped,
            run_wall_s=max(out.run_wall_s, 1e-6), detector_ms=out.detector_ms, baseline_detector_ms=None, link=out.link,
            resources=ResourceSamples(), cameras_active=out.cameras_active,
        )
        v = verify(metrics, self.mission)
        phases = self._phase_outcomes(alerts, gt.events)
        return DemoReport(
            label=label,
            policy_used=self.policy is not None,
            phases=phases,
            transitions=self.transitions,
            config_changes=self.pipe.config_changes,
            overall={k: metrics[k].value for k in ("critical_event_recall", "alert_precision", "alert_latency_p95_ms", "frames_dropped") if metrics[k].value is not None} | {"alert_latency_p95_ms": metrics["alert_latency_p95_ms"].value or float("nan")},
            verification_passed=v.passed,
            notes=list(out.notes),
        )

    def _phase_outcomes(self, alerts: list[AlertRecord], events: list[GroundTruthEvent]) -> list[PhaseOutcome]:
        bounds = [(p.start_t, (self.phases[i + 1].start_t if i + 1 < len(self.phases) else self.source.duration_s), p.name) for i, p in enumerate(self.phases)]
        first_alert = {}
        for a in sorted(alerts, key=lambda a: a.emitted_wall):
            if a.matched_event_id and a.matched_event_id not in first_alert:
                first_alert[a.matched_event_id] = a
        out = []
        for s, e, name in bounds:
            evs = [ev for ev in events if s <= ev.start < e]
            det = [ev for ev in evs if ev.id in first_alert]
            lat = [first_alert[ev.id].latency_ms for ev in det if first_alert[ev.id].latency_ms is not None]
            in_phase_alerts = [a for a in alerts if s <= a.scene_t < e]
            modes = []
            for t, d in self.decisions:
                if s <= t < e and d.changed and d.mode is not None and (not modes or modes[-1] != d.mode.name):
                    modes.append(d.mode.name)
            out.append(PhaseOutcome(name, s, e, len(evs), len(det), lat, len(in_phase_alerts), sum(1 for a in in_phase_alerts if a.matched_event_id is None), modes))
        return out


def save_report(report: DemoReport, directory: Path = Path("artifacts/demo")) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    p = directory / f"{report.label}.json"
    p.write_text(report.to_json())
    return p
