import pytest

from systemd_runner import RunnerError, SystemdRunner, UnitStatus


class RecordedRunner(SystemdRunner):
    def __init__(self, responses):
        super().__init__()
        self.responses = list(responses)
        self.commands = []

    async def _run(self, *args):
        self.commands.append(args)
        return self.responses.pop(0)


@pytest.mark.asyncio
async def test_launch_uses_argument_boundaries_and_parses_status(tmp_path):
    show = "\n".join([
        "Id=rl-scheduler-task-abc.service",
        "LoadState=loaded",
        "ActiveState=active",
        "SubState=running",
        "MainPID=123",
        "ExecMainStatus=0",
        "Result=success",
    ])
    runner = RecordedRunner([(0, "", ""), (0, show, "")])
    command = "echo hello; touch /tmp/must-not-run"
    status = await runner.launch(
        unit="rl-scheduler-task-abc.service",
        task_id="abc",
        username="alice",
        command=command,
        work_dir="/tmp",
        gpu_id=2,
        log_path=str(tmp_path / "task.log"),
    )
    launch_args = runner.commands[0]
    assert launch_args[0] == "systemd-run"
    assert launch_args[-1] == command
    assert launch_args[-3:-1] == ("-l", "-c")
    assert "--setenv=CUDA_VISIBLE_DEVICES=2" in launch_args
    assert status.running
    assert status.main_pid == 123


def test_unit_status_classifies_retained_exit():
    status = UnitStatus("unit", "loaded", "active", "exited", 0, 0, "success")
    assert status.finished
    assert not status.running


@pytest.mark.asyncio
async def test_stop_refuses_to_report_success_while_unit_is_running():
    running = "\n".join([
        "Id=unit.service", "LoadState=loaded", "ActiveState=active",
        "SubState=running", "MainPID=99", "ExecMainStatus=0", "Result=success",
    ])
    runner = RecordedRunner([
        (0, "249", ""),
        (0, "", ""),
        (0, "", ""),
        (0, running, ""),
    ])
    with pytest.raises(RunnerError, match="still running"):
        await runner.stop("unit.service")
