"""Search trajectories: every candidate a strategy evaluated, in order, with its outcome — not just
the winner. Persisted under artifacts/searches/."""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from failsafe.experiments.schema import OperatingConfig

Source = Literal["cached", "live"]


class SearchStep(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int
    config: OperatingConfig
    experiment_id: str
    source: Source
    passed: bool
    recall: float | None
    p95_ms: float | None
    frames_dropped: float | None
    capability: float
    rank: tuple
    tier: int = 0  # 2 robust pass · 1 marginal pass · 0 fail (D-029)
    suspect: bool = False  # QC flag: possible external contention (D-028)
    reason: str = ""  # why the strategy proposed this candidate


class SearchTrajectory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    search_id: str
    strategy: str
    scenario: str
    mission_hash: str
    corpus: str
    steps: list[SearchStep] = Field(default_factory=list)
    termination_reason: str = ""
    evaluations: int = 0
    feasible_found: int = 0
    best_config: OperatingConfig | None = None
    best_experiment_id: str | None = None
    best_capability: float | None = None
    pareto: list[dict[str, Any]] = Field(default_factory=list)
    no_verified_mode: dict[str, Any] | None = None
    notes: list[str] = Field(default_factory=list)
    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None

    def save(self, directory: Path = Path("artifacts/searches")) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        slug = re.sub(r"[^A-Za-z0-9._-]+", "-", f"{self.strategy}_{self.scenario}").strip("-")
        p = directory / f"{slug}_{self.started_at.strftime('%Y%m%dT%H%M%SZ')}.json"
        p.write_text(self.model_dump_json(indent=2))
        return p

    @classmethod
    def load(cls, path: Path) -> "SearchTrajectory":
        return cls.model_validate_json(path.read_text())

    def summary(self) -> str:
        best = "NONE" if self.best_config is None else self.best_config.name
        cap = "" if self.best_capability is None else f" capability={self.best_capability:.2f}"
        return (
            f"{self.strategy:8} {self.scenario:16} evaluations={self.evaluations:3d} feasible={self.feasible_found:2d} "
            f"best={best}{cap} [{self.termination_reason}]"
        )


def load_trajectories(directory: Path = Path("artifacts/searches")) -> list[SearchTrajectory]:
    if not directory.exists():
        return []
    return [SearchTrajectory.load(p) for p in sorted(directory.glob("*.json"))]


def to_jsonable(x: Any) -> Any:
    return json.loads(json.dumps(x, default=str))
