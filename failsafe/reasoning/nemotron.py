"""Nemotron via Nebius Token Factory (OpenAI-compatible chat completions).

Configuration (env): NEBIUS_API_KEY, NEBIUS_BASE_URL (default https://api.tokenfactory.nebius.com/v1),
NEMOTRON_MODEL. Nothing here is called unless a provider is constructed with a key, so the rest
of the system runs and tests without network or billing.

Structured output: the model is asked for a single JSON object; the reply is parsed and
validated against the Pydantic schema. Anything else is a RejectedOutput — never a guess.
`response_format={"type": "json_object"}` is requested when the endpoint accepts it; if the
endpoint rejects the parameter we retry once without it (the validation step is what actually
guarantees structure).
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from failsafe.mission.schema import MissionSpec
from failsafe.reasoning.base import Analysis, PlanningContext, ProposalSet, RejectedOutput, validate_or_reject

DEFAULT_BASE_URL = "https://api.tokenfactory.nebius.com/v1"

SYSTEM_PLANNER = """You are the planning component of Failsafe, a resilience compiler for edge AI systems.
You propose candidate operating configurations for a camera-based restricted-zone monitor that must keep
satisfying its mission invariants while infrastructure fails. You never decide whether a configuration
passes — a deterministic evaluator measures that. Reason from the evidence given; do not invent numbers.
Prefer configurations that keep as much application capability as possible while fixing the diagnosed
bottleneck. Answer with ONE JSON object and nothing else."""

SYSTEM_COMPILER = """You translate an operator's natural-language requirements for a camera-based
restricted-zone monitor into a strict JSON MissionSpec. Only the metrics critical_event_recall,
alert_latency_p95_ms, alert_precision, cloud_bytes_total and cpu_percent_mean exist. Answer with ONE
JSON object and nothing else."""

SYSTEM_ANALYST = """You analyse measured experiment results for Failsafe. State only findings that are
directly supported by the observations given, citing the configuration names. Answer with ONE JSON
object and nothing else."""


@dataclass
class CallRecord:
    purpose: str
    model: str
    prompt_chars: int
    completion_chars: int
    latency_s: float
    ok: bool
    error: str = ""
    usage: dict = field(default_factory=dict)


class NemotronProvider:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 8000,  # Nemotron reasoning models spend many completion tokens before the JSON
        timeout_s: float = 120.0,
    ):
        # Credentials resolve from an explicit arg, then the generic FAILSAFE_LLM_* vars (any
        # OpenAI-compatible Nemotron host — NVIDIA Build, etc.), then the Nebius-specific vars.
        # The provider name records the real host so provenance never claims Nebius when it was not.
        self.api_key = api_key or os.environ.get("FAILSAFE_LLM_API_KEY") or os.environ.get("NEBIUS_API_KEY") or os.environ.get("NVIDIA_API_KEY")
        if not self.api_key:
            raise RuntimeError("no LLM API key set (FAILSAFE_LLM_API_KEY / NEBIUS_API_KEY); use MockReasoningProvider or configure .env")
        self.base_url = base_url or os.environ.get("FAILSAFE_LLM_BASE_URL") or os.environ.get("NEBIUS_BASE_URL") or DEFAULT_BASE_URL
        self.model = model or os.environ.get("FAILSAFE_LLM_MODEL") or os.environ.get("NEMOTRON_MODEL")
        if not self.model:
            raise RuntimeError("no model set (FAILSAFE_LLM_MODEL / NEMOTRON_MODEL)")
        self.temperature, self.max_tokens, self.timeout_s = temperature, max_tokens, timeout_s
        host = self.base_url.split("//", 1)[-1].split("/", 1)[0]
        self.name = f"nemotron:{self.model}@{host}"
        self.calls: list[CallRecord] = []
        self._json_mode_supported: bool | None = None

    # -- transport ----------------------------------------------------------------------------

    def _client(self):
        from openai import OpenAI  # imported lazily: optional dependency

        return OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout_s)

    def _chat_json(self, purpose: str, system: str, user: str) -> dict:
        client = self._client()
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        t0 = time.perf_counter()
        text, usage = "", {}
        try:
            kwargs = dict(model=self.model, messages=messages, temperature=self.temperature, max_tokens=self.max_tokens)
            if self._json_mode_supported is not False:
                try:
                    resp = client.chat.completions.create(**kwargs, response_format={"type": "json_object"})
                    self._json_mode_supported = True
                except Exception as e:  # endpoint may not accept response_format
                    if self._json_mode_supported is True:
                        raise
                    self._json_mode_supported = False
                    resp = client.chat.completions.create(**kwargs)
            else:
                resp = client.chat.completions.create(**kwargs)
            text = resp.choices[0].message.content or ""
            usage = getattr(resp, "usage", None).model_dump() if getattr(resp, "usage", None) else {}
            obj = extract_json_object(text)
            self.calls.append(CallRecord(purpose, self.model, len(system) + len(user), len(text), time.perf_counter() - t0, True, usage=usage))
            return obj
        except RejectedOutput:
            self.calls.append(CallRecord(purpose, self.model, len(system) + len(user), len(text), time.perf_counter() - t0, False, "no JSON object in reply", usage))
            raise
        except Exception as e:
            self.calls.append(CallRecord(purpose, self.model, len(system) + len(user), len(text), time.perf_counter() - t0, False, str(e)[:200], usage))
            raise

    # -- ReasoningProvider --------------------------------------------------------------------

    def compile_mission(self, text: str) -> MissionSpec:
        user = (
            "Operator requirements:\n" + text.strip() + "\n\n"
            "Produce a MissionSpec JSON with keys: name, description, invariants (list of {metric, min|max, description}), "
            "objectives (list of {metric, direction: maximize|minimize, weight}), priorities (object of capability→integer), "
            "degradable_capabilities (list), survivability ({wan_outage: required|optional}). "
            "Invariants are hard requirements the operator stated; everything they said may be sacrificed goes to degradable_capabilities. A metric that appears as an invariant must NOT also appear in objectives; objectives are only for soft preferences the operator did not make hard requirements (e.g. maximize alert_precision)."
        )
        obj = self._chat_json("compile_mission", SYSTEM_COMPILER, user)
        if isinstance(obj, dict) and "mission" in obj and len(obj) == 1:
            obj = obj["mission"]
        return validate_or_reject(MissionSpec, obj)  # type: ignore[return-value]

    def propose_configs(self, ctx: PlanningContext) -> ProposalSet:
        user = planning_prompt(ctx)
        obj = self._chat_json("propose_configs", SYSTEM_PLANNER, user)
        return validate_or_reject(ProposalSet, obj)  # type: ignore[return-value]

    def analyze_results(self, ctx: PlanningContext) -> Analysis:
        user = analysis_prompt(ctx)
        obj = self._chat_json("analyze_results", SYSTEM_ANALYST, user)
        return validate_or_reject(Analysis, obj)  # type: ignore[return-value]


# ---------------------------------------------------------------------------------------------
# Prompts + parsing (pure functions, unit-testable without a key)
# ---------------------------------------------------------------------------------------------

CONFIG_SCHEMA_DOC = (
    "OperatingConfig fields: name (string), critical_fps (30|15|10|5), background_fps (30|15|10|5|2|1|0), "
    "detector_resolution (640|480|320), cloud_confirmation (bool), cloud_timeout_ms (100..30000), "
    "on_cloud_failure ('alert_local'|'drop'), alert_confirm_frames (1|2|3), historical_indexing (bool), "
    "drop_background_streams ('none'|'lowest_priority'|'all'), backlog_policy ('queue'|'drop_oldest'), "
    "local_confidence_threshold (0.05..0.95)."
)


def _obs_line(o) -> str:
    c = o.config
    cloud = f"on/{c.cloud_timeout_ms}ms" if c.cloud_confirmation else "off"
    f = lambda v, fmt: "n/a" if v is None else fmt.format(v)
    return (
        f"- {c.name}: crit {c.critical_fps}fps, bg {c.background_fps}fps, {c.detector_resolution}px, cloud {cloud}, "
        f"indexing {'on' if c.historical_indexing else 'off'} → {'PASS' if o.passed else 'FAIL'}; recall {f(o.recall, '{:.3f}')}, "
        f"p95 {f(o.p95_latency_ms, '{:.0f}ms')}, dropped {f(o.frames_dropped, '{:.0f}')}/{f(o.frames_released, '{:.0f}')} frames, "
        f"cloud timeouts {f(o.cloud_timeouts, '{:.0f}')}, cloud rejections {f(o.cloud_rejected, '{:.0f}')}, "
        f"detector {f(o.detector_ms_mean, '{:.1f}ms')}/frame, capability retained {o.capability_retained:.2f}"
    )


def planning_prompt(ctx: PlanningContext) -> str:
    m, s = ctx.mission, ctx.scenario
    inv = "; ".join(f"{i.metric} {'>= ' + str(i.min) if i.min is not None else '<= ' + str(i.max)}" for i in m.invariants)
    obj = "; ".join(f"{o.direction} {o.metric}" for o in m.objectives)
    space = "; ".join(f"{k}: {v}" for k, v in ctx.space.items())
    obs = "\n".join(_obs_line(o) for o in ctx.observations) or "(none yet)"
    return (
        f"MISSION `{m.name}`\ninvariants (hard): {inv}\nsoft objectives: {obj}\n"
        f"capability priorities: {json.dumps(m.priorities)}\n\n"
        f"SCENARIO `{s.name}`: cloud_state={s.cloud_state.value}, bandwidth_mbps={s.bandwidth_mbps}, compute_pressure={s.compute_pressure.value}. {s.description}\n\n"
        f"ALLOWED SEARCH SPACE (choose values only from here; 'cloud' means cloud_confirmation + cloud_timeout_ms): {space}\n{CONFIG_SCHEMA_DOC}\n\n"
        f"EVIDENCE so far (round {ctx.round_index}):\n{obs}\n\n"
        "Facts about this system measured earlier: the edge detector sustains ≈21–23 frames/s at 640px and ≈38 at 480px; "
        "demand is critical_fps + 3×background_fps; when demand exceeds capacity frames are dropped and recall falls; "
        "480px loses small people; a cloud call that times out delays the alert by the full timeout.\n\n"
        f"Propose up to {ctx.max_candidates} NEW candidate configurations (not already in the evidence) most likely to satisfy every "
        "invariant while retaining the most capability. Return JSON: {\"hypothesis\": str, \"candidates\": [{\"config\": OperatingConfig, "
        "\"rationale\": str, \"expected_tradeoffs\": [str]}], \"reasoning_summary\": str}."
    )


def analysis_prompt(ctx: PlanningContext) -> str:
    obs = "\n".join(_obs_line(o) for o in ctx.observations) or "(none)"
    return (
        f"SCENARIO `{ctx.scenario.name}` ({ctx.scenario.cloud_state.value}, compute {ctx.scenario.compute_pressure.value}).\n"
        f"Mission invariants: {'; '.join(i.metric for i in ctx.mission.invariants)}.\n\nOBSERVATIONS:\n{obs}\n\n"
        "Return JSON: {\"findings\": [{\"statement\": str, \"evidence\": [config names], \"confidence\": low|medium|high}], "
        "\"suggested_experiments\": [{\"config\": OperatingConfig, \"rationale\": str}], \"open_questions\": [str]}. "
        "Only state what the observations support."
    )


_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.S)


def extract_json_object(text: str) -> dict:
    """Find the single JSON object in a reply (bare or fenced). Raises RejectedOutput otherwise."""
    text = text.strip()
    candidates = []
    m = _FENCE.search(text)
    if m:
        candidates.append(m.group(1))
    if text.startswith("{"):
        candidates.append(text)
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        candidates.append(text[start : end + 1])
    for c in candidates:
        try:
            obj = json.loads(c)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue
    raise RejectedOutput("reply did not contain a JSON object")
