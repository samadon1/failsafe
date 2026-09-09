"""NebiusJobsExecutor — run each experiment as an isolated Nebius Serverless Job.

Why this exists: local campaigns must be serial for timing validity (see local.py / D-028), so a full
grid takes hours. Nebius Serverless Jobs give every experiment its own machine, so the same campaign
runs in parallel with no cross-experiment contention — the one place parallelism is both safe and
worth it. Running on a Nebius GPU preset additionally produces the GPU detector metrics that are
UNAVAILABLE locally (non-negotiable #5 / the M3 has no NVIDIA GPU).

Built against the documented Nebius CLI (`nebius ai create --type job …`, `nebius ai job get/logs`).
Requires: the `nebius` CLI configured (`~/.nebius/config.yaml`), a container image with Failsafe
installed (see `Dockerfile`), and — for Nemotron proposals inside a job — the `FAILSAFE_LLM_*` /
Nebius Token Factory env baked into the image. `dry_run=True` prints the exact commands without
submitting anything (no spend); that path is what the tests exercise.

Result collection: each job uploads its `artifacts/experiments/*.json` to an S3-compatible Nebius
Object Storage prefix (`results_uri`); the executor syncs them back. Live submit/poll/collect run
only when unblocked — the command construction is exact and tested regardless.
"""
from __future__ import annotations

import os
import shlex
import subprocess
import time
from pathlib import Path

from failsafe.executors.base import CampaignJob
from failsafe.experiments.schema import ExperimentResult

DEFAULT_IMAGE = os.environ.get("NEBIUS_FAILSAFE_IMAGE", "cr.nebius.cloud/<your-registry>/failsafe:latest")
DEFAULT_PLATFORM = os.environ.get("NEBIUS_PLATFORM", "gpu-l40s-a")     # GPU → also yields MEASURED GPU metrics
DEFAULT_PRESET = os.environ.get("NEBIUS_PRESET", "1gpu-8vcpu-32gb")
DEFAULT_TIMEOUT = os.environ.get("NEBIUS_TIMEOUT", "1h")


class NebiusJobsExecutor:
    def __init__(self, image: str = DEFAULT_IMAGE, platform: str = DEFAULT_PLATFORM,
                 preset: str = DEFAULT_PRESET, timeout: str = DEFAULT_TIMEOUT,
                 results_uri: str | None = None, device: str = "cuda", dry_run: bool = False,
                 poll_seconds: int = 15) -> None:
        self.image, self.platform, self.preset, self.timeout = image, platform, preset, timeout
        self.results_uri = results_uri or os.environ.get("NEBIUS_RESULTS_URI")  # e.g. s3://failsafe/experiments
        self.device, self.dry_run, self.poll_seconds = device, dry_run, poll_seconds

    # --- command construction (pure, exact, tested) ---
    def container_command(self, job: CampaignJob) -> str:
        """The shell run inside the container: run the experiment, then upload its artifact."""
        run = "failsafe " + " ".join(shlex.quote(a) for a in job.cli_args()) + f" --device {self.device}"
        if self.results_uri:
            up = f"aws s3 cp artifacts/experiments {self.results_uri.rstrip('/')}/{job.name}/ --recursive"
            return f"{run} && {up}"
        return run

    def create_argv(self, job: CampaignJob) -> list[str]:
        return [
            "nebius", "ai", "create", "--type", "job", "--name", job.name,
            "--image", self.image, "--container-command", "bash",
            "--args", f"-c {shlex.quote(self.container_command(job))}",
            "--platform", self.platform, "--preset", self.preset, "--timeout", self.timeout,
        ]

    # --- live orchestration (runs only when unblocked; dry_run short-circuits) ---
    def run_campaign(self, jobs: list[CampaignJob]) -> list[ExperimentResult]:
        argvs = [self.create_argv(j) for j in jobs]
        if self.dry_run:
            print(f"# DRY RUN — {len(jobs)} Nebius Serverless Jobs (nothing submitted, no spend)\n")
            for a in argvs:
                print(" ".join(shlex.quote(x) for x in a) + "\n")
            print("# then: nebius ai job logs <id> ; results sync from", self.results_uri or "<set NEBIUS_RESULTS_URI>")
            return []
        for j, a in zip(jobs, argvs):
            subprocess.run(a, check=True)
            print(f"submitted {j.name}")
        self._wait(jobs)
        return self._collect(jobs)

    def _job_state(self, job: CampaignJob) -> str:
        out = subprocess.run(["nebius", "ai", "job", "get-by-name", "--name", job.name,
                              "--format", "jsonpath={.status.state}"], capture_output=True, text=True)
        return (out.stdout or "").strip()

    def _wait(self, jobs: list[CampaignJob]) -> None:
        pending = {j.name: j for j in jobs}
        while pending:
            for name, j in list(pending.items()):
                st = self._job_state(j).upper()
                if any(k in st for k in ("COMPLETED", "SUCCEEDED", "FAILED", "ERROR", "CANCELLED")):
                    print(f"{name}: {st or 'DONE'}"); pending.pop(name)
            if pending:
                time.sleep(self.poll_seconds)

    def _collect(self, jobs: list[CampaignJob]) -> list[ExperimentResult]:
        if not self.results_uri:
            print("no NEBIUS_RESULTS_URI set — results stayed in each job; fetch with `nebius ai job logs`")
            return []
        dest = Path("artifacts/experiments"); dest.mkdir(parents=True, exist_ok=True)
        subprocess.run(["aws", "s3", "sync", self.results_uri, str(dest)], check=True)
        out = []
        for j in jobs:
            for p in dest.glob(f"{j.name}/*.json"):
                out.append(ExperimentResult.model_validate_json(p.read_text()))
        return out
