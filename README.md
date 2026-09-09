# Failsafe

**A resilience compiler for edge / Physical-AI systems.**

Give Failsafe an application and its mission invariants. It experimentally discovers the cheapest
capability degradation that preserves mission-critical behaviour when the real world stops behaving
like a data center (WAN loss, slow or zombie cloud, bandwidth collapse, compute pressure), verifies
every degraded mode with a deterministic evaluator, and compiles the verified modes into an
executable runtime policy.

> Failsafe doesn't ask an LLM what to do while a physical system is failing. It uses AI beforehand
> to discover what works, proves it experimentally, and deploys only verified degraded modes.

**[Live demo & results ↗](https://samadon1.github.io/failsafe/)** · **[Research report ↗](https://samadon1.github.io/failsafe/research.html)** · [Decision log](docs/DECISIONS.md) · [Metrics & experiments](docs/EXPERIMENTS.md)

**Status:** research prototype. Phases 0-6 complete and measured locally: 200+ real-time
experiments (two full 72-configuration grids plus repeats across three corpus seeds), a compiled
resilience policy, and the end-to-end cut-WAN demo (docs/EXPERIMENTS.md, Phase 5-6). Built:
lexicographic search engine with grid / random / greedy strategies, LLM-guided search behind a
schema-validated provider interface, policy compiler, deterministic fail-closed runtime, cut-WAN
timeline demo. Nemotron runs against NVIDIA Build (four live searches recorded: one completed, two
rejected outright by the evaluator, one correctly found nothing under compute pressure). The Nebius
Serverless Jobs executor is built and dry-run tested; no job has been submitted yet (account
activation pending). See `docs/EXPERIMENTS.md` and `docs/DECISIONS.md`. Not a certified safety
system. "Verified" has a stated bar (D-037): a configuration is verified for a condition only when
at least two un-contended runs, pooled across corpus seeds and repeats, *all* pass the mission
invariants; the figures reported for a mode are its worst run, not its best. Nothing more.

## The loop

```
APPLICATION + MISSION INVARIANTS
        │
        ▼
generate failure conditions          (faults/)
        │
        ▼
propose degraded configurations      (grid · greedy · Nemotron-guided)
        │
        ▼
run real experiments                 (local · Nebius Serverless Jobs)
        │
        ▼
measure · verify invariants          (deterministic evaluator, no LLM)
        │
        ▼
compile verified resilience policy → deterministic runtime (fail-closed)
```

## Quick start (Phase 1)

```bash
uv sync --extra dev
uv run pytest                                   # 101 tests, ~3 s (simulated clock)
uv run failsafe calibrate                       # once per host, on an idle machine: isolated detector speed (D-041)
uv run failsafe corpus build-cutouts            # one-off: person cutouts from ultralytics sample images
uv run failsafe corpus stats --tier quick       # ground-truth statistics of the discovery corpus
uv run failsafe corpus detectability            # offline recall ceiling by resolution
uv run failsafe configs                         # named operating configurations (probes)
uv run failsafe run --config island --scenario wan_offline --tier quick   # one real-time experiment (~4 min)
uv run failsafe sweep --configs normal,island --scenarios healthy,wan_zombie
uv run failsafe report                          # matrix + single-knob tradeoffs from artifacts/experiments
```

Experiments replay the corpus in **wall-clock real time** (a 180 s corpus takes 180 s); latency is
only reported for real-time runs. Every result JSON carries full provenance (seed, hashes, git SHA,
versions, backend, scoring rule) and its raw alerts, so results can be re-scored when the scoring
rule changes (`failsafe rescore`) without re-running.

## Phase 1 in one paragraph

On a 57-event synthetic restricted-zone corpus (4 cameras, 180 s each) with a real YOLOv8n edge
detector and a heavier stand-in for the cloud confirmer, 24 real-time experiments show a
non-trivial landscape: detector resolution is the dominant recall knob (640→480→320 px:
0.965→0.737→0.596), demand above the measured edge capacity costs recall through dropped frames,
critical-camera FPS 15→5 costs nothing, and the headline result is that a *reachable but slow* cloud violates
the 2 s alert invariant (zombie: p95 5.05 s; ~1.5 s RTT: p95 2.95 s) where a *dead* cloud does not
(offline: p95 0.84 s), because the pipeline waits out its timeout. Shortening the cloud timeout
from 3 s to 1 s turns the zombie case into a passing mode (p95 1.58 s) and passes healthy and
offline as well. Under severe CPU contention no probe configuration passes, a real
no-verified-mode condition. Precision showed no
measurable cost in island mode on this corpus (null result, D-022).

See `docs/ARCHITECTURE.md`, `docs/EXPERIMENTS.md` (metric definitions + measured results only) and
`docs/DECISIONS.md`.

## License

Apache-2.0. See `LICENSE`.
