"""Export the compiled resilience policy for the release page.

Reads artifacts/resilience-policy.yaml (the compiler's output, D-037) and writes a trimmed JSON
to docs/site/policy.json, so the demo's Failsafe track shows the modes, conditions, worst-case
figures and unverified reasons the policy actually contains, not a hand-written imitation of it.
Nothing is computed here; it is a projection of the artefact.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from failsafe.policy.schema import ResiliencePolicy

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts" / "resilience-policy.yaml"
OUT = ROOT / "docs" / "site" / "policy.json"


def _cfg(c) -> dict:
    return {k: getattr(c, k) for k in ("cloud_confirmation", "cloud_timeout_ms", "critical_fps", "background_fps",
                                        "detector_resolution", "historical_indexing", "on_cloud_failure")}


def _cond(c) -> dict:
    return {"cloud_state": [s.value for s in c.cloud_state], "compute_pressure": [s.value for s in c.compute_pressure],
            "bandwidth_mbps_max": c.bandwidth_mbps_max}


def _ver(v) -> dict:
    return {k: getattr(v, k) for k in ("tier", "recall", "p95_latency_ms", "precision", "runs", "clean_runs",
                                        "pass_rate", "clean_pass_rate", "seeds", "scenario", "scoring_rule")}


def main() -> None:
    p = ResiliencePolicy.from_yaml(SRC)
    out = {
        "mission": p.mission_name,
        "mission_hash": p.mission_hash,
        "corpus": p.corpus,
        "compiled_at": p.compiled_at.isoformat(),
        "admission": p.admission.model_dump(),
        "modes": [{"name": m.name, "conditions": _cond(m.conditions), "config": _cfg(m.config),
                   "capability_retained": m.capability_retained, "verification": _ver(m.verification),
                   "sacrifices": m.sacrifices} for m in p.modes],
        "unverified": [{"scenario": u.scenario, "conditions": _cond(u.conditions), "reason": u.reason,
                        "exhaustive": u.exhaustive, "candidates_tested": u.candidates_tested,
                        "best_candidate_config": u.best_candidate_config, "ceilings": u.ceilings, "required": u.required}
                       for u in p.unverified],
        "fallback": {"name": p.fallback.name, "verified_under": p.fallback.verification.scenario} if p.fallback else None,
        "notes": p.notes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    print(f"wrote {OUT.relative_to(ROOT)}: {len(out['modes'])} modes, {len(out['unverified'])} unverified, fallback {out['fallback'] and out['fallback']['name']}")
    # the report's coverage table is generated from the same JSON, so it cannot drift from the artefact
    subprocess.run([sys.executable, str(ROOT / "scripts" / "policy_table.py")], check=True)


if __name__ == "__main__":
    main()
