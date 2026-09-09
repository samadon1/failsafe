# Architecture

## Thesis

Failsafe discovers the cheapest capability degradation necessary to preserve mission-critical
behaviour under changing real-world resource failures, verifies it experimentally, and compiles it
into a deterministic runtime policy.

Two hard separations define the design:

1. **Discovery (offline, AI-assisted) vs. runtime (online, deterministic).** Nemotron reasons about
   missions, failures and candidate configurations *before* deployment. At runtime a compiled
   policy is looked up deterministically. No LLM call is on the path that decides what a failing
   system does.
2. **Proposal vs. verification.** Anything (grid search, greedy heuristic, Nemotron) may propose a
   configuration. Only the deterministic evaluator, checking measured metrics against the
   MissionSpec, may accept it.

## Components

```
                       ┌──────────────────────┐
   missions/*.yaml ──► │ MissionSpec          │  invariants · objectives · degradable capabilities
                       └──────────┬───────────┘
                                  │
   Scenario (failure condition) ──┤── OperatingConfig (candidate degraded mode)
                                  ▼
                       ┌──────────────────────┐
                       │ experiments/runner   │  corpus + config + scenario → ExperimentResult
                       │   workload/pipeline  │  real-time replay, measured metrics
                       │   faults/*           │  network · bandwidth · compute injectors
                       │   idle gate + QC     │  per-host calibration; no replay starts on a busy host (D-041)
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ experiments/evaluator│  deterministic invariant checks → PASS / FAIL
                       └──────────┬───────────┘
                                  ▼
                       ┌──────────────────────┐
                       │ policy/compiler      │  admitted results → resilience-policy.yaml (D-037)
                       │ policy/runtime       │  condition → verified mode, hysteresis, fail-closed with a stated reason
                       └──────────────────────┘
```

The verification bar (D-037): a configuration is admitted for a condition only when it has at least
two un-contended runs under it, pooled across corpus seeds and repeats, and every one of them passes
the invariants; the mode records its worst run, not its best. Conditions without a verified mode
carry a reason (untested / insufficient evidence / marginal / refuted), and the fallback is the most
conservative admitted configuration (no cloud wait first), not the baseline. Replays only count when
the host was idle: `failsafe calibrate` records the host's isolated detector speed and the runner
waits, before every run, until the machine is within 1.25× of it (D-041).

Search strategies (`failsafe/search`) sit above the runner: **grid** (exhaustive, ground truth of
what is achievable), **random** (neutral baseline), **greedy** (deterministic bottleneck heuristic:
shed the cheapest capability expected to relieve the diagnosed bottleneck), and
**Nemotron-guided** (`reasoning/planner.py`; `propose_configs` is the only LLM stage wired into a
command, and the model can only propose — the evaluator verifies; D-035). Candidates are ordered lexicographically — feasible → capability retained →
quality objectives → cost objectives (D-025) — never by a scalar weighted score. Strategies are
compared by replay against the same measured grid (D-026): evaluations to first feasible mode and
capability retained by what they settled for. Every search persists its full trajectory
(`artifacts/searches/`), the feasible set and Pareto frontier (capability ↑, CPU ↓, cloud bytes ↓),
and an explicit `NO VERIFIED MODE` result when nothing passes — the least-bad failing configuration
is never promoted. The benchmark is never constructed to favour the LLM.

Execution backends (`executors/`): `LocalExecutor` (Phase 3) and `NebiusJobsExecutor` (Phase 7,
Serverless Jobs, one experiment per job, provenance carries the job id). The Nebius executor is
built and dry-run tested; no job has been submitted yet (D-036).

## Phase 1 workload: restricted-zone monitoring

```
FrameSource (synthetic scene | real MP4)
   │  frames released at wall-clock time  (RealTimeClock; SimulatedClock in tests only)
   ▼
Sampler        per-camera target FPS, backlog policy (queue | drop_oldest)
   ▼
Detector       YOLOv8n person detector at configurable input resolution (CPU/MPS locally)
   ▼
Zone logic     foot-point-in-polygon, per-camera hysteresis
   ▼
Local decision  confidence threshold → candidate critical event
   ▼
[Cloud confirmation]  crop → ConfirmationService behind network fault injector
   │                  Phase 1: heavier local model (YOLOv8m, full-res crop) as an honest stand-in
   │                  Phase 4: Nemotron VL via Token Factory behind the same interface
   ▼
Alert          timestamped; metrics collector computes recall / precision / latency
```

`historical_indexing` is a real background cost (frame perceptual hash → SQLite) so that disabling
it produces a measurable saving rather than being a decorative flag.

### Why cloud confirmation matters

Without it, WAN loss cannot touch recall, only latency, and every "discovery" is tautological. With
cloud confirmation in normal mode, degraded modes involve a genuine trade: local-only alerting
preserves recall and latency but sacrifices precision (more false alarms) and semantic capability.
Precision is a *soft objective* in the MissionSpec, never an invariant, so Failsafe is looking for
"among all configurations that pass, which has the best tradeoff", not merely pass/fail.

### Failure model (Phase 1–2)

- Cloud state: `healthy` (≈50 ms) · `slow` (≈500 ms) · `severely_slow` (≈1500 ms) · `zombie`
  (accepted, responds after 5–10 s) · `timeout` (never responds) · `offline` (fails immediately).
  `slow ≠ dead` is modelled explicitly because a reachable-but-slow cloud may be *worse* than an
  unreachable one for latency-bound missions. The runtime probes must distinguish these.
- Bandwidth: token bucket on bytes through the cloud path.
- Compute pressure: competing busy-loop processes; the *observed* impact is recorded, not assumed.

## Corpus

- **Discovery corpus (synthetic, labelled):** seeded scene generator composites alpha-masked person
  cutouts (extracted with YOLOv8-seg from ultralytics' sample images) onto procedurally drawn
  facility backgrounds with a restricted-zone polygon. Ground truth is analytic (foot point inside
  polygon), so recall/precision are exact. Event duration, person scale, path, occlusion, contrast
  and speed are varied so FPS and resolution *materially* affect outcomes.
- **Validation holdout (real video):** `FileVideoSource` + labelling tool. Never used by search.

## Metric provenance

Every metric carries a kind: `MEASURED`, `SIMULATED`, `DERIVED`, `UNAVAILABLE`. Latency is
`UNAVAILABLE` unless replay ran at `time_scale == 1.0`. GPU metrics are `UNAVAILABLE` on the
local laptop and `MEASURED` on Nebius.
