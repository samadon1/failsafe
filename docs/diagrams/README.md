# Failsafe diagrams & charts

Reproducible figures for the README / submission. Branded NVIDIA-style system (see `kit.py`).

**Illustrations** (SVG+PNG, regenerate with `uv run python <file>` from this dir):
- `d1_architecture.py` — the core loop (hero)
- `d2_trust.py` — the trust boundary (AI proposes · evaluator verifies · runtime decides)
- `d3_ladder.py` — the degradation ladder, read from `artifacts/resilience-policy.yaml` (regenerate after `failsafe compile-policy`)
- `d4_runtime.py` — the runtime state machine

**Data charts** (PNG, every number read from `artifacts/`):
- `charts.py` → `c5_slow_not_dead`, `c6_search_efficiency`, `c7_pareto`, `c8_demo_timeline`, `c9_recall_landscape`
  Run: `uv run --with matplotlib python docs/diagrams/charts.py` from the repo root.

Charts pull from `artifacts/experiments/`, `artifacts/searches/`, `artifacts/demo/`. If those change,
re-run to refresh. Nothing here hand-codes a result number.
