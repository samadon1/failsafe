"""MissionSpec: what the system must preserve, what it may trade, and what it may sacrifice.

Three concepts, deliberately distinct:

* invariants  — hard constraints. A configuration that violates any of them is rejected.
* objectives  — soft goals used to rank configurations that pass all invariants.
* degradable_capabilities — the things Failsafe is allowed to sacrifice.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class Invariant(BaseModel):
    """A hard constraint on a measured metric. At least one bound must be given."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    min: float | None = None
    max: float | None = None
    description: str = ""

    @model_validator(mode="after")
    def _at_least_one_bound(self) -> Invariant:
        if self.min is None and self.max is None:
            raise ValueError(f"invariant '{self.metric}' needs at least one of min/max")
        if self.min is not None and self.max is not None and self.min > self.max:
            raise ValueError(f"invariant '{self.metric}': min > max")
        return self


class Objective(BaseModel):
    """A soft goal. Used to rank passing configurations, never to reject them."""

    model_config = ConfigDict(extra="forbid")

    metric: str
    direction: Literal["maximize", "minimize"]
    weight: float = Field(default=1.0, gt=0)
    description: str = ""


class Survivability(BaseModel):
    model_config = ConfigDict(extra="forbid")

    wan_outage: Literal["required", "optional"] = "required"


class MissionSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1)
    description: str = ""
    invariants: list[Invariant] = Field(min_length=1)
    objectives: list[Objective] = Field(default_factory=list)
    priorities: dict[str, int] = Field(default_factory=dict)
    degradable_capabilities: list[str] = Field(default_factory=list)
    survivability: Survivability = Field(default_factory=Survivability)

    @model_validator(mode="after")
    def _no_duplicate_metrics(self) -> MissionSpec:
        inv = [i.metric for i in self.invariants]
        if len(set(inv)) != len(inv):
            raise ValueError("duplicate invariant metrics")
        overlap = set(inv) & {o.metric for o in self.objectives}
        if overlap:
            raise ValueError(f"metric(s) {sorted(overlap)} cannot be both invariant and objective")
        return self

    # -- identity -------------------------------------------------------------------------------

    def canonical_json(self) -> str:
        return json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))

    @property
    def hash(self) -> str:
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()[:12]

    # -- io -------------------------------------------------------------------------------------

    @classmethod
    def from_yaml(cls, path: str | Path) -> MissionSpec:
        data = yaml.safe_load(Path(path).read_text())
        if isinstance(data, dict) and "mission" in data and len(data) == 1:
            data = data["mission"]
        return cls.model_validate(data)

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(yaml.safe_dump(self.model_dump(mode="json"), sort_keys=False))
