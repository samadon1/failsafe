# Decisions

Append-only log. Newest at the bottom. Each entry: what, why, consequences.

## D-001 — Pivot from adaptive inference routing to a resilience compiler (2026-08-26)
Adaptive edge/cloud routing is crowded (Sedna, AxiomVision, DACC, MacEdge, SenSem, NetsPresso
agent, ...). The under-served problem is *automatically discovering and verifying* degraded modes
against mission invariants. Failsafe owns: application + invariants → generate failures → explore
degradation → measure → verify → compile policy → deterministic runtime.

## D-002 — LLM proposes, evaluator verifies, runtime is deterministic (2026-08-26)
No LLM on the runtime decision path. Reproducibility, low latency, resilience to cloud loss, and a
crisp answer to "why trust the agent". Fail closed when no verified mode matches.

## D-003 — Degraded modes must cost something (2026-08-26)
Normal mode uses cloud confirmation of local detections (higher precision). Island mode alerts on
the local detector alone. Precision and bandwidth/CPU are *soft objectives*; recall and latency
are *invariants*. Otherwise the only discovery is "local inference works offline", which is trivial.

## D-004 — Synthetic discovery corpus + real holdout (2026-08-26)
Recall ≥ 95% is meaningless on 20 events. A seeded synthetic generator gives hundreds of exactly
labelled events with controllable duration/scale/occlusion/contrast so FPS and resolution matter.
A small real-video holdout, never touched by search, guards against "policies that exploit the
generator". Results from the two corpora are always reported separately.

## D-005 — Person cutouts come from ultralytics' own sample images (2026-08-26)
`bus.jpg` and `zidane.jpg` contain real people; YOLOv8n-seg extracts alpha-masked cutouts. Real
detector-compatible, no external dataset, no licensing surprises. Variety is limited (~6 people,
flips, scales); acceptable for Phase 1 and flagged as a corpus limitation.

## D-006 — Phase 1 "cloud" confirmer is a heavier local model behind a fault injector (2026-08-26)
YOLOv8m on the full-resolution crop stands in for a cloud VLM until Nemotron VL is wired in
(Phase 4). Same `ConfirmationService` interface. It is labelled as a stand-in everywhere.

## D-007 — Real-time replay for all latency measurements (2026-08-26)
Frames are released at wall-clock `start + t`. Faster-than-real-time replay makes latency
meaningless, so `time_scale != 1.0` marks latency UNAVAILABLE. Simulated clock in unit tests only.
Consequence: one experiment takes as long as the clip; parallelism (Nebius) is the answer at scale.

## D-008 — Slow ≠ dead (2026-08-26)
Cloud states include `zombie` and `timeout` distinct from `offline`. Hypothesis to test: a
reachable-but-slow cloud violates the latency invariant more often than a clean outage because the
pipeline waits. Runtime probes will need to distinguish RTT degradation from reachability.

## D-009 — Search baselines: grid AND greedy heuristic (2026-08-26)
Nemotron-guided search is compared against exhaustive grid *and* a domain greedy heuristic
("disable the highest-cost degradable capability first, retest"). Results reported as measured,
including Nemotron losing.

## D-010 — Python 3.12 + uv; ultralytics YOLOv8n; CPU/MPS locally (2026-08-26)
Local machine is an Apple M3 with no NVIDIA GPU. torch/ultralytics support 3.12 cleanly (3.14 is
too new). GPU metrics are UNAVAILABLE locally, MEASURED on Nebius.

## D-011 — Failsafe has its own CLAUDE.md (2026-08-26)
The parent directory's CLAUDE.md belongs to an unrelated project and would be inherited. A local
CLAUDE.md overrides it with the Failsafe non-negotiables.

## D-012 — Consecutive zone events are separated by ≥ 1.6 s of clear zone (2026-08-26)
The pipeline suppresses re-alerting until the zone has been clear for 1 s (one intrusion = one
alert). Two people entering 0.5 s apart would then be one event operationally but two in ground
truth, making recall ambiguous. The generator enforces `MIN_EVENT_GAP_S = 1.6` between zone events
on a camera. Distractors may still overlap events in time.

## D-013 — Simulated clock ⇒ lock-step processing (2026-08-26)
Under `SimulatedClock` frames are processed synchronously on the driver thread and cloud
confirmations resolve inline: processing takes zero simulated time, so no backlog can form and
results are deterministic. Real-time runs use a worker thread + confirmation pool. Simulated runs
never produce latency numbers (UNAVAILABLE).

## D-014 — Stand-in cloud confirmer runs out of process (2026-08-26)
`CloudServerProcess` hosts the YOLOv8m confirmer in a child process whose PID is excluded from the
edge device's CPU accounting, so "normal" mode is not charged for compute that would live in the
cloud. Compute-pressure workers are excluded the same way. Requests are serialised through a pipe
(one in flight at the server); network delays happen on the caller side.

## D-015 — Local detector runs on CPU by default (2026-08-26)
So that compute-pressure workers contend with inference directly and the edge budget is honest.
MPS/CUDA remain selectable (`--device`). Detector ms/frame is calibrated unpressured per run and
`compute_pressure_observed` is derived from it.

## D-016 — Corpus difficulty is calibrated, and the calibration is written down (2026-08-26)
The synthetic corpus is an *evaluation instrument*, not a claim about the world. Its difficulty
profile is tuned so that the full-fidelity configuration can reach the recall invariant while
reduced resolution / FPS / occlusion materially lose events — otherwise every configuration fails
and there is nothing to discover. Measured with `failsafe corpus detectability` (YOLOv8n, conf 0.4,
3 frames/event, quick seed 1):

| profile | 640 | 480 | 320 |
|---|---|---|---|
| heights 22–200 px, alpha 0.55–1, bottom occlusion ≤40 %, 3 cutouts | 0.49* | 0.31* | 0.26* |
| heights 45–200 px, alpha 0.7–1, overhead occlusion ≤25 %, 2 cutouts | 0.93 | 0.84 | 0.54 |

(*) alpha ≥ 0.7 subset. The `bus_0` cutout was excluded (D-005 addendum): a thin dark back-view
figure YOLOv8n scores 0.06–0.36 even at 96–165 px. Bottom occluders were replaced by overhead ones
because hiding the feet makes the foot-point rule unsatisfiable (impossible, not hard).
Current profile constants live in `failsafe/corpus/scene.py` (`HEIGHT_RANGE_PX`, `ALPHA_RANGE`,
`OCCLUSION_LEVELS`). Any change to them changes every scene hash and must be re-documented here.

## D-017 — `normal` is defined within measured edge capacity (2026-08-26)
Sustainable detector throughput on the dev laptop (CPU) is ≈26 fps at 640. `normal` demands
15 + 3×2 = 21 fps. Over-capacity probes (`normal_overload`, `normal_30fps`, `island_bg5`) exist on
purpose to expose the backlog/drop landscape. On Nebius GPUs the capacity differs; per-run
`detector_ms_baseline` records it.

## D-018 — Simulated cameras render in a child process (2026-08-26)
In the first quick-tier run the detector measured 50 ms/frame inside the pipeline versus 39 ms in
isolation: frame rendering on the driver thread competed for CPU and the GIL, and its cost was
charged to the edge device. `RemoteSyntheticSource` renders in a separate process (PID excluded
from CPU accounting) and streams frames ahead of consumption over a pipe. Simulation overhead is
not edge work.

## D-019 — Alert refractory period instead of "zone clear" cooldown (2026-08-26)
The original rule re-armed alerting only after the zone had been clear for 1 s. A near-miss walker
whose detected foot point jitters across the boundary kept the state armed indefinitely and masked
real intrusions on the critical camera (2 of 7 misses in the first quick run were 86 px / 119 px
unoccluded people). Now: one candidate per `refractory_s = 1.5` s per camera. Ground-truth events
are ≥ 1.6 s apart (D-012), so a false alarm can never mask a real event for a whole event.

## D-020 — Background cameras carry longer events (2026-08-26)
Critical camera A: in-zone durations 0.15–4 s. Background cameras B/C/D: 0.8–4 s (people linger in
non-critical areas). This encodes the priority structure in the corpus: reducing background FPS is
a *cheap* degradation (it costs some recall on long events only at 1 fps) rather than a free one,
while critical-camera FPS remains expensive to cut.

## D-021 — Alert matching window has a 0.5 s lead tolerance; results are rescorable (2026-08-26)
The analytic ground truth starts the instant the exact foot point crosses the polygon. A real
detector's box bottom crosses the painted line a frame or two earlier, so an alert 33 ms *before*
the GT start was scored as a false alarm — and the refractory period then hid the true alert
(`A-ev014`, 86 px, unoccluded, detected at 0.76 conf). An alert is attributable to an event if it
falls in `[start − 0.5 s, end + 2 s]`; early alerts get latency 0. Because every result stores its
raw alerts, `failsafe rescore` re-derives recall/precision/latency for stored experiments under the
current rule and records the rule version in provenance — no silent re-scoring, no re-running.

## D-022 — Precision cost of local-only mode: null result on this corpus (2026-08-26)
Hypothesis (D-003): island mode preserves recall/latency but pays in precision. Measured: after the
lead-tolerance fix every experiment scores precision 1.000, and the offline probe
(`failsafe corpus precision-probe`) finds 0/163 near-miss frames producing an in-zone candidate at
local thresholds 0.15/0.25/0.4 and 0/87 hallucinations on empty frames. YOLOv8n's foot-point
localisation is tighter than the 2–18 px near-miss margins. The geometric distractor design does
not create a precision landscape. Reported as measured; not tuned away.
Plan (Phase 2 corpus extension, not built yet): make the semantic distinction the cloud is *for* —
authorised persons (hi-vis marker) may enter the zone, unauthorised may not. The local detector
sees "person"; only the confirmer (VLM in Phase 4; colour-marker stand-in before that) can tell
them apart. Island mode then has a real, meaningful precision cost: every authorised entry alerts.

## D-023 — Replay-driver lag is measured and reported, and is small (2026-08-26)
Diagnosis on the smoke tier: remote cameras + real detector → driver release lag p50 5.7 ms,
p99 15 ms, max 18 ms. Sporadic 150–260 ms stalls also occur with a *null* detector, i.e. they are
OS scheduling hiccups, not GIL contention. Runs now record `driver_lag_p50_ms` / `driver_lag_p95_ms`
so harness lag can always be separated from system latency. The ≈0.9–1.1 s p95 alert latency of
passing configurations is dominated by detection onset (the detector's box bottom crosses the
line some frames after the analytic foot point), which is a property of the system under test.

## D-024 — `normal` means the verified healthy operating point (2026-08-27)
Phase 1's cloud-heavy designer default (3 s timeout, indexing, 21 fps @640) fails the mission even
when healthy (recall 0.947). It is renamed `naive_default` and kept as the "system without
Failsafe" baseline; `normal` is now the configuration that passed healthy / offline / zombie
(1 s timeout). Config identity is the hash, so no experiment id changed; reports resolve legacy
names by hash.

## D-025 — Lexicographic mission optimisation, no scalar score (2026-08-27)
Candidates are ordered by (feasible, capability retained, quality objectives, cost objectives).
Capability is a property of the configuration, weighted by the mission's `priorities`, relative
to the highest-capability configuration of the space. Measured objectives are compared in
tolerance bands (precision 0.01, cloud bytes 250 KB, CPU 10 %) so noise cannot decide the order.
Why: a weighted sum invites "why those weights?"; graceful degradation means *sacrifice as little
as possible while staying feasible*, which is exactly a lexicographic order.

## D-026 — Strategies are compared by replay against the same measured grid (2026-08-27)
The 72-config space is measured exhaustively per scenario (real time, ~5.4 h per scenario
locally). Grid, random and greedy — and later Nemotron — are then evaluated by replaying their
candidate sequences against those stored outcomes: identical evidence, zero extra experiments,
exact "evaluations to first feasible" and "capability retained". Limitation: replay ignores
run-to-run noise; the repeatability runs quantify it separately. A strategy that proposes a
configuration outside the measured space must run it live (allowed with `--live`).

## D-027 — Overload / backpressure is a first-class search signal (2026-08-27)
Phase 1 showed more sensing can make the system worse (30 fps @640: 1,672 drops, recall 0.912,
p95 1.68 s). The greedy heuristic diagnoses overload from the measured dropped-frame fraction
(> 3 %) and sheds load cheapest-first (indexing → background FPS → critical FPS → resolution)
before touching anything else; the space deliberately contains over-capacity points.

## D-028 — Measurement hygiene: nothing else runs on the machine during a campaign (2026-08-27)
Repeatability runs exposed outliers (`normal × healthy` run 3: recall 0.930, 878 drops, detector
57 ms/frame; `island × wan_zombie` run 3: 0.912, 606 drops, 61 ms) whose detector time was
1.3–1.5× the unpressured baseline in scenarios with *no* injected pressure. They coincide with
test suites and CLI tools being run on the same laptop while the campaign measured. Uncontended
repeats agree to ±1 event and ±~200 ms p95. Rules: (1) no CPU-heavy work while a campaign runs;
(2) every run carries `qc_suspect_contention` (detector > 1.25× baseline without injected
pressure) and a note; (3) the result cache prefers clean runs; reports mark suspect runs with †;
(4) the detector baseline is calibrated once per resolution per session, as a median, before any
cloud server or pressure workers exist (an earlier per-run mean was itself contaminated:
`naive_overload` recorded a 163 ms "baseline").

## D-029 — Verification margin: robust vs marginal pass (2026-08-27)
With 57 ground-truth events one event is 1.75 % of recall, and the 0.95 threshold sits between
2 misses (0.965, pass) and 3 (0.947, fail). Run-to-run noise is about one event, so single-run
verdicts at the threshold are noise-level. A pass is *robust* when every invariant clears its
threshold by at least the noise floor (recall: 1 / n_events; p95 latency: 250 ms, from the
repeatability spread), otherwise *marginal*. The lexicographic order ranks robust above marginal
above fail before considering capability, so the search never prefers a configuration whose
feasibility rests on noise. Marginal modes are candidates for repeats (and for a larger corpus
on Nebius), not for the policy.

## D-030 — LLM-guided search runs inside the same harness, with guard rails (2026-08-27)
`LLMSearch` wraps any `ReasoningProvider` (Nemotron via Token Factory, or the deterministic
mock). Per round the model proposes ≤ 4 candidates from the bounded space; the evaluator
(cache, or live only when allowed) decides; the loop stops at the first verified mode like the
other strategies, so evaluations-to-first-feasible and capability retained are directly
comparable. Guard rails: outputs are schema-validated (Pydantic) and rejected on any failure;
two consecutive rejected outputs end the search with an explicit negative; proposals outside the
space are rejected, not snapped; duplicates are skipped; the model never sees or sets pass/fail.
Billing: nothing calls the API unless `--provider nemotron` is chosen and a key is present.

## D-031 — Runtime condition classification (2026-08-27)
The runtime observes only what an edge box can measure: cloud reachability (connection accepted),
p95 round-trip over a recent window, caller-timeout rate, detector slowdown (ms/frame ÷ calibrated
baseline) and an uplink estimate. Classes mirror the scenario definitions: RTT < 300 ms healthy,
< 1 s slow, < 3 s severely slow, else zombie; refused = offline; reachable with no completed round
trips = timeout-class (unknown is never healthy). Slowdown < 1.5 normal, < 2.2 moderate, < 3.5
severe, else critical (first written as 1.25 / 2; the 1.5 floor is the measured value from D-034,
and D-040 records which of these are measured and which asserted). Hysteresis: degrade after 3
consistent probes, recover after 5.

## D-032 — Policy compiler semantics (2026-08-27)
One mode per scenario: the best passing configuration by the lexicographic order (robust before
marginal). Conditions are the scenario's own (cloud state × compute class × bandwidth cap); a
condition that was never measured is not covered — the runtime does not interpolate. Scenarios
with no passing configuration are compiled as explicit `unverified` entries with measured
ceilings; the runtime then activates the most *conservative* verified mode (least capability, no
cloud dependency, least demand) labelled UNVERIFIED and escalates. First real compile (2026-08-27,
28 experiments): healthy / zombie / offline all resolve to the same config (cloud on, 1 s timeout,
capability 1.00, robust) — for those conditions the only thing to give up is a 3 s timeout;
low-bandwidth is marginal; severely-slow and compute-severe are unverified.
Superseded in part by D-037: admission now needs every clean run of a configuration to pass (pooled
across seeds and repeats), the fallback order puts *no cloud wait* before capability, and
`unverified` entries carry a reason. The first-compile result above (healthy / zombie / offline all
the 1 s-timeout config, "robust") was an artefact of deciding on one run; see D-037.

## D-033 — Runtime probes are cheap health pings, not confirmations (2026-08-27)
If the runtime only learned about the cloud from confirmation calls, a local-only mode would never
notice recovery (it makes no calls). The prober sends a 200-byte health ping through the same
fault model every 2 s (background thread in real time, so a zombie cloud cannot stall the driver)
and classifies from a rolling window of ping + confirmation outcomes keyed to the pipeline clock.
At startup, with no evidence, the runtime does not act: unknown is neither healthy nor a reason to
degrade. Reconfiguration is applied at the next released frame; the schedule is rebuilt from the
current scene time, so camera FPS, cloud path, timeout and indexing all switch live.

## D-034 — Classify network state from ping RTTs; slowdown thresholds above the noise floor (2026-09-05)
The first live demo run misclassified a healthy cloud as `slow × moderate/severe`: (1) the RTT
window mixed confirmation round trips — which include the confirmer's inference time (~160–600 ms)
— with 50 ms pings; (2) the detector baseline was calibrated before the stand-in cloud server
existed, so its CPU side-load read as compute pressure. Fixes: pings are tagged `ping_ok` and are
the primary network signal (confirmation RTTs only until a first ping lands); the demo calibrates
with the cloud server idle-loaded; slowdown classes start at 1.5× (clean in-pipeline runs measure
1.0–1.3×). Effect in the first run was cosmetic (the fallback config equalled `normal`), but with a
richer ladder a misclassification would switch modes — the classifier must be calibrated so that
verified-normal operation classifies as normal.

## D-035 — LLM path runs on NVIDIA Build during development; snap on search dimensions (2026-09-05)
Ghana is not excluded by the hackathon rules (only Brazil, Quebec, Russia, Crimea, Cuba, Iran,
North Korea + OFAC), but the Token Factory *billing* form has no Ghana country entry, blocking
promo-credit redemption. Workaround: develop the Nemotron path on NVIDIA Build
(`https://integrate.api.nvidia.com/v1`, free key, no billing form), which serves the identical
model id `nvidia/nemotron-3-super-120b-a12b`. Provider credentials resolve from `FAILSAFE_LLM_*`
env (then Nebius vars); the provider name records the real host so provenance never claims Nebius
when it was not. Submission runs must still go through Nebius — one env swap, same measured grids.

Two integration fixes from the first live calls: (1) Nemotron 3 Super is a reasoning model that
puts its chain in a separate `reasoning_content` field and needs generous `max_tokens` (2000
truncated the JSON mid-object; raised to 8000). (2) `snap_to_space` matched on full-config hash,
so any non-search field the model set (e.g. a timeout while cloud was off) put the candidate
"outside the space"; it now snaps on the five SEARCH_SPACE dimensions and canonicalises the rest
to grid defaults (`catalog.snap_to_grid`).

## D-036 — Nebius Serverless Jobs executor (parallel campaigns) + Token Factory switch

Context: local campaigns must run serially for timing validity (parallel real-time replays contend
and corrupt latency — D-028), so a full grid takes hours; and GPU metrics are UNAVAILABLE on the M3.
Nebius Serverless Jobs give each experiment an isolated machine — the one place parallelism is both
safe (no cross-experiment contention) and worth it, and a GPU preset additionally yields the MEASURED
GPU metrics we cannot get locally.

Decision: add `failsafe/executors/` — a small `Executor` protocol with `LocalExecutor` (sequential,
in-process; the timing-valid default) and `NebiusJobsExecutor` (one `nebius ai create --type job` per
experiment, built against the documented CLI, results synced from S3-compatible Object Storage).
`failsafe campaign --executor nebius --dry-run` prints the exact job commands and submits nothing, so
the integration is verifiable and demonstrable with zero spend; the command construction is unit-
tested (`tests/test_executors.py`). A `Dockerfile` builds the job image. Nemotron already resolves its
host from `FAILSAFE_LLM_*` / `NEBIUS_BASE_URL` (default = Token Factory), so pointing proposals at
Nebius Token Factory is one env swap — no code change.

Status: code-complete and tested; live submission is blocked on Nebius/Token Factory billing (a
platform issue, tracked separately), not on the code. On unblock: drop `--dry-run`.

## D-037 — Admission is decided on every clean run; the fallback is chosen for conservativeness (2026-09-06)

Context (from a code review of the shipped policy): repeated runs were never combined. The
compiler took the *earliest* un-contended run of an experiment (`clean[0]`) and admitted or
rejected on that one run; `pass_rate` was computed and never read. The shipped
`artifacts/resilience-policy.yaml` therefore carried `zombie-cloud` as `tier: robust` while one of
its three clean runs exceeded the 2000 ms latency invariant (p95 2030 ms), and `EXPERIMENTS.md`
described a "marginal on latency" outcome the compiler could not actually produce. Separately,
every compiled mode had the identical configuration, so the "most conservative verified mode"
fallback degenerated (all keys tied; `min()` returned the highest-capability, cloud-dependent
config) — the fallback for a compute-starved condition was the full-fidelity cloud-on mode.
Finally, `NO VERIFIED MODE` after 73 candidates and after 1 candidate looked identical, and
scenarios with no results at all vanished into a note.

Decision:
1. **Admission.** An experiment is verified only if it has at least `min_clean_runs` (default 2)
   runs not flagged as contended (D-028) and EVERY clean run passes the invariants. The tier is the
   worst clean run's (robust only if every clean run clears the noise floor, D-029). The mode
   records the worst-case recall / p95 / precision over clean runs, plus `runs`, `clean_runs`,
   `pass_rate`, `clean_pass_rate`. Runs are pooled per configuration × scenario across corpus
   seeds and repeats: a pass on each of three seeds is three clean runs (the seed-generalization
   data is evidence, not a separate study), and one failing seed spoils the pool exactly like one
   failing repeat. A single passing run is evidence, not verification. The rule is
   written into the policy (`admission`), so the artefact states its own bar; a pre-D-037 file
   loaded through the new schema is labelled `legacy`, not silently upgraded.
2. **Reasons.** `unverified` entries carry `reason` ∈ {untested, insufficient_evidence, marginal,
   refuted}, `exhaustive` (the whole search space was tried) and the best candidate to repeat.
   Zero-result scenarios are listed as `untested`. The runtime's NO VERIFIED MODE reason string
   includes the reason and the number of candidates tested.
3. **Fallback.** Chosen over every admitted experiment (not just the per-scenario winners) by a
   conservativeness order that puts *no cloud wait* first (a fallback serves conditions nobody
   verified; it must never block on the network), then the shortest timeout, then least detector
   demand, then smallest input, indexing off, least capability, and finally the config hash so ties
   are deterministic. If the chosen configuration is already a compiled mode, that mode is reused.

Consequence, deliberately accepted: the policy gets *smaller* until the evidence is there. Modes
that rested on a single run drop to `insufficient_evidence` with their candidate named; repeats
are run locally (real-time, no spend) and the policy recompiled. Whatever the repeats show is what
ships (CLAUDE.md non-negotiables 6 and 11). Tests: `tests/test_policy.py` now constructs
disagreeing runs, contended runs, single runs, the all-identical-config degeneracy and untested
scenarios; every classifier boundary constant is pinned so a silent change fails a test.

## D-038 — Provenance records the corpus seed and flags dirty trees (2026-09-06)

Review finding: `Provenance.seed` stored the *scenario* seed, which is 0 in every one of the 205
stored results, while the seed that actually varies between runs (corpus seed 1/2/3) lived only in
`experiment.corpus.seed` and was recoverable from the top-level provenance only through
`corpus_hash`. `git_sha` recorded HEAD with no dirty-tree check, so a recorded SHA did not pin the
code that ran. Decision: `seed` is the corpus seed; `git_sha` is suffixed `-dirty` when tracked
files have uncommitted changes. Stored results are not rewritten (their `corpus_hash` remains the
authoritative link); results written after this entry carry the corrected fields.

## D-039 — Operator activity contaminated four replays; flagged runs are quarantined and re-run idle (2026-09-06)

What happened: while the D-037 repeat queue was replaying in real time, the operator (this
session) ran test suites, recompiles and exports on the same laptop. Each `uv run` imports torch
and saturates the cores for several seconds; four replays overlapped with those spikes and their
detector averaged 75–132 ms/frame against the 41 ms isolated baseline. The runner's run-time QC
check (`qc_suspect_contention`, D-028) flagged all four, so none of them can count as clean
evidence and the compiler ignored them; `island × wan_slow` "failing" with 1377 dropped frames was
this, not the scenario.

Decision: (1) a real-time replay campaign owns the machine — no other Python, Chrome, or build
process runs until the campaign's log says `end`; (2) a run flagged by the QC check on a
no-pressure scenario is moved to `artifacts/experiments/stale/<reason-date>/` (kept for
provenance, excluded from the result cache by `result_files`) and re-run on an idle machine;
(3) any run recorded today that carries the flag is treated this way, whether or not the cause is
known. This is the D-028 contention rule applied to the operator, not just to parallel jobs.

## D-040 — Runtime and pipeline constants: which are measured, which are asserted (2026-09-06)

A review found several constants that decide results without a recorded basis. This entry states
each one's status so nobody mistakes an asserted number for a measured one; every value below is
pinned by a test so a silent change fails.

| constant | value | status |
|---|---|---|
| `SLOWDOWN_MODERATE` | 1.5× | **measured**: clean in-pipeline detector runs sit at 1.0–1.3× the isolated baseline (D-034) |
| `SLOWDOWN_SEVERE`, `SLOWDOWN_CRITICAL` | 2.2×, 3.5× | asserted; no run yet sits near either boundary |
| `RTT_HEALTHY_MS`, `RTT_SLOW_MS`, `RTT_SEVERE_MS` | 300, 1000, 3000 ms | asserted boundaries between the scenario RTTs (50/500/1500/7500 ms); the zombie class (≥ 3 s) starts beyond every cloud timeout in the search space |
| `TIMEOUT_RATE_ZOMBIE` | 0.5 | asserted |
| `degrade_after`, `recover_after` | 3, 5 probes (6 s / 10 s at the 2 s ping interval) | asserted; to be derived from the repeatability data (false-transition rate under healthy-case noise) |
| prober `window_s`, `ping_timeout_s`, `PING_BYTES` | 10 s, 1.5 s, 200 B | asserted |
| `MAX_BACKLOG_DROP_OLDEST` | 3 frames | asserted; it is the lever behind every `frames_dropped` result (D-027) and needs a sensitivity run |
| `DRAIN_TIMEOUT_S` | 20 s | asserted |
| `WORKER_FRACTION` (compute pressure) | 0 / 0.5 / 1.0 / 2.0 busy workers per core | asserted; it defines what "severe compute pressure" means, and therefore the `compute_severe` refutation |
| `GRACE_S` / `LEAD_S` (scoring window) | 2.0 s / 0.5 s | lead measured (D-021); grace under study (`scripts/grace_sensitivity.py`) |
| `ISOLATED_BASELINE_MS` (post-hoc QC) | 41 / 27 / 14 ms | measured on the dev laptop only (D-028); the run-time QC uses a per-run baseline, so only the post-hoc rescore flag is host-specific |

D-031's earlier slowdown boundaries (1.25 / 2 / 3.5) were superseded by D-034 and this table; the
code is the reference.

## D-041 — A replay may only start on an idle host; the QC check compares against a per-host isolated baseline (2026-09-06)

What happened: the D-037 repeat campaign (16 replays, 19:08–19:58 UTC) ran while macOS
Spotlight was indexing the artifact directory (`mds` + six `mdworker` processes at 200%+ CPU in
total). Every replay was starved: detector 75–132 ms/frame at 640 px against the 41 ms isolated
baseline, thousands of dropped frames, every run FAIL, including `island` under conditions that
cannot affect it. Ten runs were caught by the run-time QC flag. Six were not, because the runner
measured its session baseline once at process start — under the same load — and then judged each
run against that already-slow number. A baseline measured under load hides the load.

Decision:
1. **Per-host isolated calibration.** `failsafe calibrate` measures the detector's ms/frame per
   resolution on an idle machine, twice; it refuses to write if the two disagree by more than 15%
   (the machine was not idle), and stores the result in `artifacts/calibration/<hostname>.json`
   with hostname, platform, git SHA and timestamp. This replaces the dev-laptop constant
   `ISOLATED_BASELINE_MS` wherever a calibration exists.
2. **The idle gate.** Before every replay the runner re-measures the session baseline (never
   cached across runs) and, if the host is calibrated, waits — 30 s at a time, up to 10 minutes —
   while the measurement exceeds 1.25× the isolated baseline. If the host is still busy after
   that it runs anyway, and the result is flagged.
3. **Three-way QC.** A no-pressure run is suspect if the detector ran more than 1.25× slower than
   its session baseline, *or* than the isolated host baseline, *or* if the session baseline itself
   was more than 1.25× the isolated one. The isolated baseline is recorded on the result
   (`detector_ms_isolated`, UNAVAILABLE on an uncalibrated host). The post-hoc rescore uses the same
   per-host file.
4. **The 2026-09-06 19:08–19:58 campaign is quarantined whole** (`artifacts/experiments/stale/
   contaminated-2026-09-06/`), flagged or not: the host was loaded for its entire duration. Every
   experiment in it is re-run behind the gate.

Operator side: the artifact and scratch directories are excluded from Spotlight indexing on the
dev laptop, since a campaign writes hundreds of files the indexer would otherwise chase.

## D-042 — The 1.25× contention threshold was re-examined against all stored runs and kept (2026-09-06)

Question: D-034 measured clean in-pipeline runs at 1.0–1.3× the isolated baseline (the confirmer
shares the CPU on the single-box lab setup), and the QC flag fires at 1.25×. If cloud-on
configurations naturally sit at 1.3× under load, the flag would exclude them systematically and
bias the compiled policy toward cloud-off modes. A `normal × wan_slow` re-run flagged at 1.30× on
an otherwise idle host prompted the check.

Measurement (126 no-pressure runs, contaminated campaign excluded): cloud-on median 1.01×, cloud-off
median 0.97×; 58 of 79 cloud-on and 39 of 47 cloud-off runs sit below 1.25×; the [1.25, 1.30) bin
is **empty**, a natural break exactly at the threshold; and **every one of the 16 runs between
1.25× and 1.6× is a FAIL** — not a single passing run exists above 1.25×. A slowdown past the
threshold coincides with a degraded run every time, which is what the flag is for. Raising it to
1.5× would have turned flagged failures into "clean" evidence and refuted `normal` under `healthy`
on the strength of two contaminated runs.

Decision: the threshold stays at 1.25× for both the run-time and post-hoc checks. A cloud-on run
that reads above it under a slow cloud on an idle host is treated as suspect, not as evidence
against the configuration; if that repeats across seeds on an idle host it is a finding about the
configuration (its own confirmer backlog starving the detector) and will be reported as such, with
the measurement, rather than absorbed by a looser threshold.

## D-043 — Pressure scenarios run last, with a cooldown; heat is contention too (2026-09-06)

Observation, on an idle host (Spotlight at 0%) during the gated re-run campaign: a cloud-off
`island` run read 1.12× the isolated baseline and passed; the two `island × wan_timeout` runs that
came shortly after a `survival × offline_compute_severe` run (two busy workers per core for
3½ minutes) read 1.49× and 1.30×, dropped 735 and 472 frames, failed, and were flagged. The cloud
is off in `island`, so the scenario cannot be the cause. The idle gate measures the host *before* a
run, so heat that builds *during* the previous run and throttles a passively cooled laptop is
invisible to it. Hypothesis, not yet confirmed: thermal throttling after pressure runs.

Decision: `run_sweep` orders compute-pressure scenarios last (`pressure_last`) and idles
`cooldown_s` (default 90 s; `failsafe repeat --cooldown-s`) after each pressure run before the next
replay. Flagged runs are still excluded by the QC check; this reduces how many get wasted. If the
next campaign shows the same pattern with the cooldown in place, the gate will additionally
require a short *sustained* benchmark to pass, not just a 30-frame one.

## D-044 — Pressure workers die with their parent; a campaign aborts rather than replay on a loaded host (2026-09-06)

What happened: the first gated campaign was force-killed (`kill -9`) while an
`offline_compute_severe` replay was running. Its eight compute-pressure workers — busy-loop
processes started with `daemon=True` — survived the parent's death (`daemon` only acts on a clean
exit) and spun on under `ppid 1` for 67 minutes at ~57% CPU each: severe compute pressure, injected
by accident, on the whole machine. The second campaign's calibration passed (the workers were
briefly starved by the calibration itself), its first replay ran at 5.24× the isolated baseline
with 3071 dropped frames, and the gate — having waited its full budget — ran anyway and flagged it.
Half an hour per discarded result. (The sustained gate reading and the warm-machine refusal from
D-043 were correct but not sufficient; the D-043 "thermal" readings after the 20:44 pressure run
were most likely these workers, not heat.)

Decision:
1. A pressure worker keeps running only while its parent asks *and its parent is alive*
   (`os.getppid()` still equals the parent's pid) *and* a hard lifetime cap (30 min) holds. A
   force-killed campaign can no longer leave load behind.
2. When the idle gate exhausts its wait budget, the default is to **abort** (`HostBusy`), not to
   run flagged: `failsafe repeat --on-busy abort|run`. A sweep stops at the first `HostBusy` with
   exit code 3; results already stored are kept.
3. Operator rule: stop a campaign with SIGTERM to the process group, never `kill -9` on a replay;
   after any forced stop, check for orphaned `Python` processes with `ppid 1` before starting
   anything that measures time.

## D-045 — The 2.0 s matching grace is kept: it decides nothing (2026-09-06)

Question (from the review): D-021 justified the 0.5 s lead with a measured incident; the 2.0 s
post-event grace had no justification on record and, on a corpus whose median event lasts 1.2 s,
looked capable of inflating recall and precision.

Measurement (`scripts/grace_sensitivity.py`, read-only, 245 stored synthetic results rescored at
grace 0.5 s, 1.0 s and 2.0 s): **zero verdict flips**, and per-scenario mean recall and precision
identical to three decimals at every window (e.g. `wan_zombie` n=85: 18 pass / 0.755 / 0.998 at all
three). The window is immaterial on this corpus because alerts either land while the person is
still inside the zone or not at all; the grace only ever had to absorb sub-second timing jitter.

Decision: keep 2.0 s so stored results remain comparable with the scoring rule they carry, and
record here that any value between 0.5 s and 2.0 s would have compiled the same policy. The study
is re-run whenever the corpus or the alert path changes; a non-zero flip count would reopen this.

## D-046 — Quarantined results were still being read; the cache now excludes `stale/` at any depth (2026-09-06)

What happened: `result_files()` excluded only files whose parent directory was exactly
`artifacts/experiments/stale/`. The D-039/D-041 quarantine puts results under
`stale/<reason-date>/<experiment-id>/`, one level deeper, so 27 of the 30 quarantined results were
still read by `ResultCache`. The policy compiled at the end of the third campaign (22:5x UTC), the
`docs/site/policy.json` exported from it, the report's generated coverage table, the `failsafe
demo` re-measurement that followed and the ladder diagram were all built on a cache that included
contaminated runs — which is why `wan_timeout` compiled as *refuted* while `island` had just passed
it cleanly on both seeds. None of that reached a release: it was caught in the ladder before
anything was pushed.

Decision: anything under `stale/` at any depth is excluded (`"stale" not in relative path parts`),
pinned by a test. Everything compiled from the polluted cache was regenerated from the corrected
one, and this entry exists so the 22:5x artefacts, if ever seen in a log, are not mistaken for
results.

## D-047 — Evidence pools by what a configuration can observe (2026-09-06)

Finding: after the third campaign, `island` (cloud confirmation off) compiled as *verified* under
`wan_slow`, `wan_zombie` and `wan_timeout` and *refuted* under `bandwidth_2mbps` and `wan_offline`.
But a cloud-off configuration has no network injector, no confirmer process and no bandwidth path
(`runner.py` creates them only when `cloud_confirmation` is on): for `island`, every cloud state and
bandwidth cap is the same experiment. Pooled, its 17 clean no-pressure runs pass 14 times at recall
0.965–1.000 and miss one or two extra events in 3 (0.930–0.947), one of them at 0.88× baseline
with nothing slow. Its margin over the 0.95 invariant is one event (the D-029 *marginal* tier);
per-scenario splitting with two runs each turned that one behaviour into "verified" where both
runs happened to land well and "refuted" where one did not.

Decision: evidence is pooled by the scope a configuration can observe. A cloud-on configuration
is judged per scenario (cloud state and bandwidth act on it). A cloud-off configuration is judged
per **compute pressure** only, across every cloud scenario at that pressure; it is verified for
all of them or for none. Consequence on this hardware: `island` is **refuted** everywhere (3 clean
failures in 17), so the compiled zombie / slow / timeout modes it carried are gone, and those
conditions read `NO VERIFIED MODE` with the fallback labelled unverified. That is the honest
reading: on an M3 CPU the cloud-off configuration sits on the invariant boundary, and Failsafe's
job is to say so rather than average it away. Headroom (a GPU node) is what would move it.

## D-049 — A real NVIDIA VLM as the cloud confirmer; Cosmos Reason needs a GPU node (2026-09-07)

The Phase-1 confirmer is a heavier YOLO standing in for a cloud VLM (D-006). `failsafe/workload/
vlm_confirmer.py` makes it real: `VlmConfirmer` draws the zone polygon and the candidate box on the
frame and asks an NVIDIA multimodal reasoning model whether the person's feet are inside the zone,
parsing a JSON verdict. It is endpoint/model-agnostic (OpenAI-compatible `/v1/chat/completions` with
`image_url`).

What is reachable (probed with the .env NVIDIA Build key):
- `nvidia/nemotron-3-nano-omni-30b-a3b-reasoning` — **hosted, accessible, works.** Verified live on a
  real warehouse frame: confirmed person-in-zone, confidence 0.95, with a correct rationale. This is
  what the confirmer uses today. It is NVIDIA sponsor tech (Nemotron family, multimodal member).
- `nvidia/cosmos3-nano-reasoner` — **self-hosted NIM only** (`docker run --gpus all
  nvcr.io/nim/nvidia/cosmos3-reasoner`, then `127.0.0.1:8000/v1/chat/completions`). No hosted
  catalog endpoint; needs an NVIDIA GPU we do not have locally → a Nebius/GPU node. The confirmer
  points at it with one env swap (`FAILSAFE_VLM_BASE_URL`, `FAILSAFE_VLM_MODEL=nvidia/cosmos3-nano-reasoner`),
  no code change. This ties the Cosmos story to the GPU-node story.
- `nvidia/cosmos-reason2-8b` — hosted but account-gated (404 "not found for account").
- `nvidia/Cosmos3-Nano` (HF weights) — 16B, BF16-only, requires NVIDIA Ampere/Hopper/Blackwell on
  Linux; ~32 GB in BF16, so it does not fit the M3's 16 GB and Apple Silicon is unsupported. (This
  repo is the omni *generation* model, not the reasoner, but the GPU requirement is the same.)

Every path to Cosmos needs an NVIDIA GPU we do not have locally — a hardware wall, not a code gap.

Why it is NOT in the timing grid: a remote VLM call is seconds long and non-deterministic, which
would corrupt real-time alert latency (D-005) and reproducibility. Every verdict is cached by a
content hash of the overlaid frame; a replay reuses the cached verdict while the NetworkInjector
still supplies the simulated cloud latency. The deterministic YOLO stand-in stays the default for
the measured 200-run grid; the VLM confirmer is for the demo and a labelled qualitative set. A call
failure never takes down a run — it reads as "unconfirmed" and the local decision stands.

Claim discipline: this is "NVIDIA multimodal reasoning as the cloud confirmer (Nemotron-omni today;
Cosmos Reason on a GPU node)", never "Cosmos is used".
