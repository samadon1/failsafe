# Failsafe console (UI)

`console.html` — a single-file operator console (no build step, no backend).

Open it in a browser, or it's published as a Claude artifact. Inject a condition
(healthy / cut WAN / zombie cloud / compute pressure) and watch the deterministic
policy runtime switch to a verified degraded mode — or fail closed (NO VERIFIED MODE).
Toggle **Failsafe** off to see the naive static system miss intrusions during an outage.

The camera feeds are illustrative; every recall / latency / capability figure and the
compiled-policy ladder are the real measured numbers from `artifacts/` (see docs/EXPERIMENTS.md).
Runtime logic mirrors `failsafe/policy/runtime.py` — no model runs in the decision loop.
Dark/light themed; IBM Plex Sans + Mono; inherits the diagram design system.
