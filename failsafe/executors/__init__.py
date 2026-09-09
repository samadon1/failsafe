"""Campaign executors: run a set of experiments locally (sequential) or on Nebius Serverless Jobs
(parallel, isolated). Same interface; the runtime decision loop never touches either — these only
run the offline discovery campaigns."""
from failsafe.executors.base import CampaignJob, Executor
from failsafe.executors.local import LocalExecutor
from failsafe.executors.nebius import NebiusJobsExecutor

__all__ = ["CampaignJob", "Executor", "LocalExecutor", "NebiusJobsExecutor"]
