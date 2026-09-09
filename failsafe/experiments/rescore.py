"""Re-derive alert-based metrics (recall, precision, latency, alert counts) for stored results.

Every ExperimentResult keeps its raw alerts (scene time, wall-clock emission and frame release
times). When the *scoring rule* changes (e.g. the matching window), stored experiments can be
rescored without re-running them; all non-alert metrics (throughput, resources, cloud stats) are
kept as measured. The rescore is recorded in `notes` and provenance so it is never silent.
"""

from __future__ import annotations

from pathlib import Path

from failsafe.corpus.ground_truth import GRACE_S, LEAD_S, compute_ground_truth
from failsafe.corpus.scene import generate_scene
from failsafe.experiments.evaluator import verify
from failsafe.experiments.schema import ExperimentResult
from failsafe.workload.metrics import ResourceSamples, compute_metrics

ALERT_METRICS = {
    "critical_event_recall", "alert_precision", "alerts_total", "false_alerts", "gt_events_total",
    "gt_events_on_dropped_cameras", "alert_latency_mean_ms", "alert_latency_median_ms",
    "alert_latency_p95_ms", "alert_latency_max_ms",
}

def rule_version(grace_s: float = GRACE_S, lead_s: float = LEAD_S) -> str:
    return f"match-window[start-{lead_s}s, end+{grace_s}s]"


RULE_VERSION = rule_version()  # the scoring rule in force

# Isolated (nothing else running) detector ms/frame per resolution on the dev laptop, used to
# flag stored runs post hoc when their own per-run baseline was itself contaminated (D-028).
ISOLATED_BASELINE_MS = {640: 41.0, 480: 27.0, 320: 14.0}
CONTENTION_RATIO = 1.25


class StaleCorpus(Exception):
    """The stored result's corpus no longer matches the generator; it cannot be rescored."""


def rescore(result: ExperimentResult, grace_s: float = GRACE_S, lead_s: float = LEAD_S) -> ExperimentResult:
    """Rescore one stored result under a matching window (default: the rule in force). Pass a
    different window to *measure* the window's influence; the returned result then carries that
    window in its scoring rule, so it can never be mistaken for one scored under the default."""
    exp = result.experiment
    if exp.corpus.kind != "synthetic":
        raise NotImplementedError("rescoring real-corpus results needs the clip labels")
    scene = generate_scene(exp.corpus.tier, exp.corpus.seed)
    if scene.hash != result.provenance.corpus_hash:
        raise StaleCorpus(
            f"result was measured on corpus {result.provenance.corpus_hash}, current generator yields {scene.hash}"
        )
    gt = compute_ground_truth(scene)
    # cameras_active only affects gt_events_on_dropped_cameras; recover it from the config
    from failsafe.workload.pipeline import active_cameras

    class _Src:  # minimal FrameSource view for active_cameras()
        cameras = ["A", "B", "C", "D"]
        critical_camera = "A"

    cameras_active = active_cameras(_Src(), exp.config)  # type: ignore[arg-type]
    fresh, alerts = compute_metrics(
        [a.model_copy(update={"matched_event_id": None, "latency_ms": None}) for a in result.alerts],
        gt,
        real_time=(exp.time_scale == 1.0),
        frames_processed=0,
        frames_dropped=0,
        run_wall_s=1.0,
        detector_ms=[],
        baseline_detector_ms=None,
        link=None,
        resources=ResourceSamples(),
        cameras_active=cameras_active,
        grace_s=grace_s,
        lead_s=lead_s,
    )
    rule = rule_version(grace_s, lead_s)
    metrics = dict(result.metrics)
    for k in ALERT_METRICS:
        if k in fresh:
            metrics[k] = fresh[k]
    # QC flag (post hoc): detector slowdown without injected pressure → suspect run
    from failsafe.experiments.schema import ComputePressure, Metric

    if exp.scenario.compute_pressure == ComputePressure.NORMAL:
        from failsafe.experiments.runner import load_host_baseline

        det = result.metric_value("detector_ms_mean")
        host = load_host_baseline(hostname=result.provenance.hostname)  # per-host calibration (D-041) when it exists
        base = (host or ISOLATED_BASELINE_MS).get(exp.config.detector_resolution)
        suspect = bool(det is not None and base and det / base > CONTENTION_RATIO)
        src = "host calibration" if host else "dev-laptop constant"
        metrics["qc_suspect_contention"] = Metric.measured(1.0 if suspect else 0.0, "flag", f"post hoc: detector_ms_mean ÷ isolated baseline ({src}) > 1.25 without injected pressure")
    elif "qc_suspect_contention" not in metrics:
        metrics["qc_suspect_contention"] = Metric.measured(0.0, "flag", "pressure scenario; contention is injected by design")
    verification = verify(metrics, exp.mission)
    notes = [n for n in result.notes if not n.startswith("rescored")] + [f"rescored with {rule}"]
    prov = result.provenance.model_copy(update={"extra": {**result.provenance.extra, "scoring_rule": rule}})
    return result.model_copy(update={"metrics": metrics, "alerts": alerts, "verification": verification, "notes": notes, "provenance": prov})


def rescore_file(path: Path, write: bool = True) -> tuple[ExperimentResult, ExperimentResult]:
    before = ExperimentResult.from_json(path.read_text())
    after = rescore(before)
    if write:
        path.write_text(after.to_json())
    return before, after


def _m(r: ExperimentResult, k: str) -> str:
    v = r.metric_value(k)
    return "n/a" if v is None else f"{v:.3f}"


def rescore_dir(directory: Path, write: bool = True) -> list[str]:
    lines = []
    from failsafe.experiments.runner import result_files

    for p in result_files(directory):
        try:
            b, a = rescore_file(p, write)
        except StaleCorpus as e:
            lines.append(f"{p.name}: SKIPPED — {e}")
            continue
        lines.append(
            f"{p.name}: {b.experiment.config.name}×{b.experiment.scenario.name} recall {_m(b,'critical_event_recall')}→{_m(a,'critical_event_recall')} "
            f"precision {_m(b,'alert_precision')}→{_m(a,'alert_precision')} p95 {_m(b,'alert_latency_p95_ms')}→{_m(a,'alert_latency_p95_ms')} "
            f"{'PASS' if b.passed else 'FAIL'}→{'PASS' if a.passed else 'FAIL'}"
        )
    return lines

