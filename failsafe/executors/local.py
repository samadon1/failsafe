"""LocalExecutor — run a campaign in-process, one experiment at a time.

Sequential is not a limitation to fix later; it is *required* for timing validity. Latency is only
meaningful under wall-clock real-time replay (non-negotiable #5), and running several replays at once
on one machine makes them contend for CPU and corrupts every latency measurement — this is the exact
failure the contention QC flag (D-028) was built to catch. So the honest local campaign is serial,
which is why a full grid takes hours. NebiusJobsExecutor removes that ceiling by giving each
experiment its own isolated machine — parallelism without contention.
"""
from __future__ import annotations

from pathlib import Path

from failsafe.executors.base import CampaignJob
from failsafe.experiments.schema import ExperimentResult


class LocalExecutor:
    def __init__(self, device: str = "cpu") -> None:
        self.device = device

    def run_campaign(self, jobs: list[CampaignJob]) -> list[ExperimentResult]:
        from failsafe.experiments.catalog import CONFIGS, load_scenarios
        from failsafe.experiments.runner import LocalRunner
        from failsafe.experiments.schema import CorpusRef, Experiment
        from failsafe.mission.schema import MissionSpec

        runner = LocalRunner(detector_device=self.device)
        out: list[ExperimentResult] = []
        try:
            for j in jobs:
                exp = Experiment(
                    mission=MissionSpec.from_yaml(Path(j.mission)),
                    scenario=load_scenarios(Path(j.scenarios_file))[j.scenario],
                    config=CONFIGS[j.config],
                    corpus=CorpusRef(tier=j.tier, seed=j.seed),
                )
                out.append(runner.run(exp))
        finally:
            runner.close()
        return out
