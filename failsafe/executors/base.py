"""Executor interface + the unit of a campaign (one experiment = one job)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from failsafe.experiments.schema import ExperimentResult


@dataclass(frozen=True)
class CampaignJob:
    """One experiment to run: a named config × scenario × seed on a corpus tier.

    Maps 1:1 onto the `failsafe run` CLI, so the same job description drives both the in-process
    LocalExecutor and a containerised Nebius job.
    """
    config: str
    scenario: str
    seed: int = 1
    tier: str = "quick"
    mission: str = "missions/restricted-zone.yaml"
    scenarios_file: str = "scenarios/phase1.yaml"

    @property
    def name(self) -> str:
        """Nebius job name (DNS-ish: lowercase, hyphens)."""
        return f"fs-{self.config}-{self.scenario}-s{self.seed}".replace("_", "-").lower()

    def cli_args(self) -> list[str]:
        """The `failsafe run ...` argument list this job corresponds to."""
        return ["run", "--config", self.config, "--scenario", self.scenario,
                "--seed", str(self.seed), "--tier", self.tier,
                "--mission", self.mission, "--scenarios-file", self.scenarios_file]


@runtime_checkable
class Executor(Protocol):
    def run_campaign(self, jobs: list[CampaignJob]) -> list[ExperimentResult]:
        """Run every job and return the ExperimentResults (empty on a dry run)."""
        ...
