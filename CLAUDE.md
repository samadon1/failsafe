# CLAUDE.md — Failsafe

> Project spec and working context, read at the start of each session. Keep it current: when a
> decision changes, record it here and in `docs/DECISIONS.md`.

## What this is

**Failsafe** is a resilience compiler for edge / Physical-AI systems.

Given an application and its **mission invariants**, Failsafe experimentally discovers the
**cheapest capability degradation** that preserves mission-critical behaviour under real-world
resource failures (WAN loss, slow/zombie cloud, bandwidth collapse, compute pressure), verifies
every candidate degraded mode with a deterministic evaluator, and compiles the verified modes into
an executable runtime policy.

Thesis in one sentence:

> Failsafe doesn't ask an LLM what to do while a physical system is failing. It uses AI beforehand
> to discover what works, proves it experimentally, and deploys only verified degraded modes.

Built for the Nebius × NVIDIA Global AI Hackathon (Physical AI track). Sponsor tech that must be
materially used, not decorative: **Nemotron** (via **Nebius Token Factory**, OpenAI-compatible API)
for mission compilation / candidate reasoning / result analysis; **Nebius Serverless Jobs** for
parallel experiment campaigns.

## Non-negotiables

1. **No fabricated metrics.** Every number shown is MEASURED, SIMULATED, DERIVED or UNAVAILABLE and
   labelled as such. Never invent example numbers in docs/README that could be mistaken for results.
2. **LLM proposes. Evaluator verifies.** Nemotron may compile missions, propose configurations and
   analyse results. Only the deterministic evaluator decides pass/fail. No LLM code path in the
   runtime decision loop.
3. **No unverified mode at runtime.** The runtime activates only modes that are linked to a passing
   experiment. If no verified mode matches the current condition: **fail closed** into the most
   conservative explicitly defined fallback and say `NO VERIFIED MODE AVAILABLE`.
4. **Measurements must be reproducible.** Every experiment records seed, config hash, scenario hash,
   mission hash, git SHA, versions, backend, timestamps.
5. **Latency is only meaningful under wall-clock real-time replay.** If `time_scale != 1.0`, latency
   metrics are UNAVAILABLE. Simulated clocks are for unit tests only.
6. **Never rig a benchmark so the LLM wins.** Search strategies are compared as grid / greedy
   heuristic / Nemotron-guided and results are reported as measured, including losses.
7. **Synthetic vs real evidence stays separated.** Discovery corpus = synthetic (labelled). Validation
   holdout = real video, never used by the search.
8. **Prefer small systems.** Python + SQLite + containers + Token Factory + Serverless Jobs.
   No Kubernetes, Kafka, Redis, ROS, multi-agent swarms, custom training.
9. **CLI before UI.** The full loop must work from the command line before any frontend exists.
10. **Do not claim certified safety**, production readiness, guaranteed hazard detection or
    regulatory compliance. Use: prototype, research system, experimental resilience policy,
    "verified against the evaluation corpus".
11. **Do not optimise for demo appearance at the expense of experimental validity.**

## Stack

- Python 3.12, `uv`, Pydantic v2, NumPy, OpenCV (headless), ultralytics YOLOv8 (person detector),
  psutil, Typer CLI, pytest.
- Local dev is an Apple M3 laptop (no NVIDIA GPU): detector runs on CPU/MPS; GPU metrics are
  UNAVAILABLE locally and MEASURED only on Nebius.

## Layout

See `docs/ARCHITECTURE.md`. Short version: `failsafe/mission` (MissionSpec), `failsafe/corpus`
(synthetic scene generator + ground truth), `failsafe/workload` (real-time replay pipeline:
sampler → detector → zone → local decision → optional cloud confirmation → alert), `failsafe/faults`
(network / bandwidth / compute injectors), `failsafe/experiments` (Scenario, OperatingConfig,
Experiment, ExperimentResult, evaluator, runner, sweep), `failsafe/policy` (compiler + runtime,
Phase 5), `failsafe/reasoning` (Nemotron provider + mock, Phase 4), `failsafe/executors`
(local + Nebius, Phase 3/7).

## Working method

- Small vertical slices, commit per slice. Tests alongside code.
- Keep `docs/DECISIONS.md` current: when a decision is made or changed, write it down with the why.
- `docs/EXPERIMENTS.md` holds metric definitions and results from real runs only.
- When an integration is blocked by credentials: interface + mock + tests, keep going.
- Don't guess external API syntax (Nebius, Token Factory, ultralytics): check current docs.

## Phase order

0 scaffold → 1 deterministic workload + corpus + metrics → 2 fault injection → 3 experiment engine
→ 4 Nemotron → 5 policy compiler + runtime → 6 end-to-end demo → 7 Nebius Serverless → 8 UI
→ 9 research results. Do not skip ahead to UI or cloud before the local loop is proven.
