"""Named operating configurations, the bounded search space, and scenario loading.

Naming (DECISIONS D-024): `normal` is the *verified* healthy operating point (passes the mission
under `healthy`). The cloud-heavy designer default that does not pass even when healthy is kept
as `naive_default` — it is the "system without Failsafe" baseline. Names are labels only;
experiment identity is the config hash, so renaming never changes an experiment id.

Capacity note (measured on the dev laptop, CPU, DECISIONS D-015/D-017): YOLOv8n costs ≈39 / 26 /
13 ms per frame at 640 / 480 / 320 in isolation (≈47 ms at 640 in-pipeline). The demanded rate is
critical_fps + 3 × background_fps (four cameras).
"""

from __future__ import annotations

import itertools
from pathlib import Path

import yaml

from failsafe.experiments.schema import OperatingConfig, Scenario


def _c(name: str, description: str, **kw) -> OperatingConfig:
    return OperatingConfig(name=name, description=description, **kw)


CONFIGS: dict[str, OperatingConfig] = {
    c.name: c
    for c in [
        # ---- baselines ------------------------------------------------------------------------
        _c("naive_default", "designer default: cloud confirmation, 3 s timeout, indexing, 15+3×2 fps @640 (fails healthy)",
           critical_fps=15, background_fps=2, detector_resolution=640, cloud_confirmation=True, cloud_timeout_ms=3000,
           historical_indexing=True),
        _c("naive_drop_on_fail", "naive_default but a failed cloud call drops the candidate (fail closed on alerting)",
           critical_fps=15, background_fps=2, detector_resolution=640, cloud_confirmation=True, on_cloud_failure="drop"),
        _c("naive_overload", "naive_default with background at 5 fps: 30 fps @640 (over capacity)",
           critical_fps=15, background_fps=5, detector_resolution=640, cloud_confirmation=True, historical_indexing=True),
        _c("naive_30fps", "naive_default with critical camera at 30 fps: 36 fps @640 (over capacity)",
           critical_fps=30, background_fps=2, detector_resolution=640, cloud_confirmation=True, historical_indexing=True),
        _c("naive_queue", "naive_overload with an unbounded backlog (never drop frames)",
           critical_fps=15, background_fps=5, detector_resolution=640, cloud_confirmation=True, historical_indexing=True,
           backlog_policy="queue"),
        # ---- verified healthy operating point ---------------------------------------------------
        _c("normal", "verified healthy mode: cloud confirmation with a 1 s timeout, indexing, 15+3×2 fps @640",
           critical_fps=15, background_fps=2, detector_resolution=640, cloud_confirmation=True, cloud_timeout_ms=1000,
           historical_indexing=True),
        # ---- degraded ladder probes ----------------------------------------------------------------
        _c("degraded", "cloud on (3 s timeout), indexing off, background 1 fps",
           critical_fps=15, background_fps=1, detector_resolution=640, cloud_confirmation=True, historical_indexing=False),
        _c("island", "local only, indexing off, background 1 fps",
           critical_fps=15, background_fps=1, detector_resolution=640, cloud_confirmation=False, historical_indexing=False),
        _c("island_bg5", "island with background at 5 fps: 30 fps @640 (over capacity)",
           critical_fps=15, background_fps=5, detector_resolution=640, cloud_confirmation=False, historical_indexing=False),
        _c("island_480", "island at 480 px detector input",
           critical_fps=15, background_fps=1, detector_resolution=480, cloud_confirmation=False, historical_indexing=False),
        _c("island_320", "island at 320 px detector input",
           critical_fps=15, background_fps=1, detector_resolution=320, cloud_confirmation=False, historical_indexing=False),
        _c("island_10fps", "island, critical camera at 10 fps",
           critical_fps=10, background_fps=1, detector_resolution=640, cloud_confirmation=False, historical_indexing=False),
        _c("island_5fps", "island, critical camera at 5 fps",
           critical_fps=5, background_fps=1, detector_resolution=640, cloud_confirmation=False, historical_indexing=False),
        _c("island_drop_lowest", "island with the lowest-priority background camera dropped",
           critical_fps=15, background_fps=1, detector_resolution=640, cloud_confirmation=False, historical_indexing=False,
           drop_background_streams="lowest_priority"),
        _c("survival", "critical camera only, 5 fps, 320 px, local only",
           critical_fps=5, background_fps=0, detector_resolution=320, cloud_confirmation=False, historical_indexing=False,
           drop_background_streams="all"),
    ]
}

# Names used by Phase 1 artifacts before the rename (D-024); resolved by config hash in reports.
LEGACY_NAMES = {"normal_short_timeout": "normal", "normal_drop_on_fail": "naive_drop_on_fail", "normal_overload": "naive_overload"}


def config_display_name(cfg: OperatingConfig) -> str:
    """Current catalogue name for a config (by hash), else its stored name."""
    for c in CONFIGS.values():
        if c.hash == cfg.hash:
            return c.name
    return LEGACY_NAMES.get(cfg.name, cfg.name)


# ---------------------------------------------------------------------------------------------
# Bounded search space (Phase 2–3). 2 × 3 × 2 × 3 × 2 = 72 configurations.
# ---------------------------------------------------------------------------------------------

SEARCH_SPACE: dict[str, list] = {
    "critical_fps": [15, 5],
    "background_fps": [2, 1, 0],
    "detector_resolution": [640, 480],
    "cloud": ["off", "on_3000", "on_1000"],  # cloud_confirmation × cloud_timeout_ms
    "historical_indexing": [True, False],
}


def grid_configs() -> list[OperatingConfig]:
    out = []
    for cf, bg, res, cloud, idx in itertools.product(*SEARCH_SPACE.values()):
        on = cloud != "off"
        timeout = int(cloud.split("_")[1]) if on else 3000
        name = f"g_c{cf}_b{bg}_r{res}_{cloud}_{'idx' if idx else 'noidx'}"
        out.append(
            OperatingConfig(
                name=name,
                critical_fps=cf,
                background_fps=bg,
                detector_resolution=res,
                cloud_confirmation=on,
                cloud_timeout_ms=timeout,
                historical_indexing=idx,
            )
        )
    return out


def load_scenarios(path: str | Path = "scenarios/phase1.yaml") -> dict[str, Scenario]:
    data = yaml.safe_load(Path(path).read_text())
    return {s["name"]: Scenario.model_validate(s) for s in data["scenarios"]}


def snap_to_grid(cfg: OperatingConfig) -> OperatingConfig | None:
    """Map an arbitrary OperatingConfig onto the grid member with the same SEARCH_SPACE knob
    values, canonicalising every non-search field to the grid's defaults. Returns None if any
    search knob is out of range. This lets a proposer (LLM) set only the knobs it reasons about
    without having to reproduce every default exactly (D-035)."""
    if cfg.critical_fps not in SEARCH_SPACE["critical_fps"]:
        return None
    if cfg.background_fps not in SEARCH_SPACE["background_fps"]:
        return None
    if cfg.detector_resolution not in SEARCH_SPACE["detector_resolution"]:
        return None
    if cfg.historical_indexing not in SEARCH_SPACE["historical_indexing"]:
        return None
    # cloud dimension: off, or on with a timeout that is one of the grid's timeouts
    if not cfg.cloud_confirmation:
        cloud = "off"
    else:
        cloud = f"on_{cfg.cloud_timeout_ms}"
        if cloud not in SEARCH_SPACE["cloud"]:
            return None
    by = {c.hash: c for c in grid_configs()}
    on = cloud != "off"
    timeout = int(cloud.split("_")[1]) if on else 3000
    canon = OperatingConfig(
        name="snap", critical_fps=cfg.critical_fps, background_fps=cfg.background_fps,
        detector_resolution=cfg.detector_resolution, cloud_confirmation=on, cloud_timeout_ms=timeout,
        historical_indexing=cfg.historical_indexing,
    )
    return by.get(canon.hash)
