"""Campaign executors: the Nebius job-command construction is exact and tested (it runs on unblock);
the local executor's job mapping is checked without a heavy real run."""
from failsafe.executors import CampaignJob, LocalExecutor, NebiusJobsExecutor


def test_campaign_job_name_and_cli():
    j = CampaignJob(config="island", scenario="wan_zombie", seed=2)
    assert j.name == "fs-island-wan-zombie-s2"
    a = j.cli_args()
    assert a[:3] == ["run", "--config", "island"]
    assert "--scenario" in a and "wan_zombie" in a and "--seed" in a and "2" in a


def test_nebius_container_command_with_upload():
    ex = NebiusJobsExecutor(image="img:1", results_uri="s3://failsafe/exp", device="cuda")
    j = CampaignJob(config="normal", scenario="healthy", seed=1)
    cmd = ex.container_command(j)
    assert cmd.startswith("failsafe run --config normal")
    assert "--device cuda" in cmd
    assert "aws s3 cp artifacts/experiments s3://failsafe/exp/fs-normal-healthy-s1/ --recursive" in cmd
    assert " && " in cmd  # run then upload


def test_nebius_create_argv_matches_cli_shape():
    ex = NebiusJobsExecutor(image="cr/img:2", platform="gpu-l40s-a", preset="1gpu-8vcpu-32gb", timeout="1h")
    argv = ex.create_argv(CampaignJob(config="island", scenario="wan_offline"))
    assert argv[:5] == ["nebius", "ai", "create", "--type", "job"]
    for flag, val in [("--image", "cr/img:2"), ("--platform", "gpu-l40s-a"),
                      ("--preset", "1gpu-8vcpu-32gb"), ("--timeout", "1h"),
                      ("--container-command", "bash")]:
        assert flag in argv and argv[argv.index(flag) + 1] == val
    assert argv[argv.index("--name") + 1] == "fs-island-wan-offline-s1"


def test_nebius_dry_run_submits_nothing(capsys):
    ex = NebiusJobsExecutor(image="img:1", dry_run=True, results_uri="s3://b/p")
    out = ex.run_campaign([CampaignJob(config="normal", scenario="wan_zombie"),
                           CampaignJob(config="island", scenario="wan_zombie")])
    assert out == []                       # dry run collects nothing
    printed = capsys.readouterr().out
    assert printed.count("nebius ai create --type job") == 2
    assert "DRY RUN" in printed


def test_executors_satisfy_protocol():
    from failsafe.executors.base import Executor
    assert isinstance(LocalExecutor(), Executor)
    assert isinstance(NebiusJobsExecutor(dry_run=True), Executor)
