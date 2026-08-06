from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass


class RunnerError(RuntimeError):
    pass


class SupervisorUnavailable(RunnerError):
    pass


@dataclass(frozen=True)
class UnitStatus:
    unit: str
    load_state: str
    active_state: str
    sub_state: str
    main_pid: int
    exit_status: int
    result: str

    @property
    def exists(self) -> bool:
        return self.load_state == "loaded"

    @property
    def running(self) -> bool:
        return self.exists and self.active_state in {"active", "activating"} and self.sub_state not in {
            "dead", "exited", "failed",
        }

    @property
    def finished(self) -> bool:
        return self.exists and not self.running


class SystemdRunner:
    def __init__(self, command_timeout: float = 10.0):
        self.command_timeout = command_timeout

    @staticmethod
    def unit_name(task_id: str) -> str:
        return f"rl-scheduler-task-{task_id}.service"

    async def is_available(self) -> bool:
        try:
            returncode, stdout, _ = await self._run(
                "systemctl", "show", "--property=Version", "--value"
            )
        except (OSError, asyncio.TimeoutError):
            return False
        return returncode == 0 and bool(stdout.strip())

    async def launch(
        self,
        *,
        unit: str,
        task_id: str,
        username: str,
        command: str,
        work_dir: str,
        gpu_id: int,
        log_path: str,
    ) -> UnitStatus:
        args = (
            "systemd-run",
            "--quiet",
            f"--unit={unit}",
            f"--description=rl-scheduler task {task_id}",
            "--service-type=exec",
            "--remain-after-exit",
            f"--uid={username}",
            f"--working-directory={work_dir}",
            "--setenv=CUDA_DEVICE_ORDER=PCI_BUS_ID",
            f"--setenv=CUDA_VISIBLE_DEVICES={gpu_id}",
            "--property=KillMode=control-group",
            "--property=TimeoutStopSec=5s",
            f"--property=StandardOutput=append:{os.path.abspath(log_path)}",
            f"--property=StandardError=append:{os.path.abspath(log_path)}",
            "/bin/bash",
            "-l",
            "-c",
            command,
        )
        try:
            returncode, _, stderr = await self._run(*args)
        except (OSError, asyncio.TimeoutError) as exc:
            raise SupervisorUnavailable(f"systemd launch unavailable: {exc}") from exc
        if returncode != 0:
            if not await self.is_available():
                raise SupervisorUnavailable(stderr.strip() or "systemd is unavailable")
            raise RunnerError(stderr.strip() or f"systemd-run exited with {returncode}")
        return await self.inspect(unit)

    async def inspect(self, unit: str) -> UnitStatus:
        properties = "Id,LoadState,ActiveState,SubState,MainPID,ExecMainStatus,Result"
        try:
            _, stdout, stderr = await self._run(
                "systemctl", "show", "--no-pager", f"--property={properties}", unit
            )
        except (OSError, asyncio.TimeoutError) as exc:
            raise SupervisorUnavailable(f"systemd status unavailable: {exc}") from exc
        values: dict[str, str] = {}
        for line in stdout.splitlines():
            key, separator, value = line.partition("=")
            if separator:
                values[key] = value
        if not values and stderr:
            if not await self.is_available():
                raise SupervisorUnavailable(stderr.strip())
        return UnitStatus(
            unit=values.get("Id") or unit,
            load_state=values.get("LoadState", "not-found"),
            active_state=values.get("ActiveState", "inactive"),
            sub_state=values.get("SubState", "dead"),
            main_pid=_parse_int(values.get("MainPID")),
            exit_status=_parse_int(values.get("ExecMainStatus"), default=-1),
            result=values.get("Result", "unknown"),
        )

    async def stop(self, unit: str) -> None:
        if not await self.is_available():
            raise SupervisorUnavailable("systemd is unavailable")
        await self._run(
            "systemctl", "kill", "--signal=SIGKILL", "--kill-who=all", unit
        )
        await self._run("systemctl", "stop", unit)
        status = await self.inspect(unit)
        if status.running:
            raise RunnerError(f"unit {unit} is still running after stop")

    async def cleanup(self, unit: str) -> None:
        if not await self.is_available():
            raise SupervisorUnavailable("systemd is unavailable")
        await self._run("systemctl", "stop", unit)
        await self._run("systemctl", "reset-failed", unit)
        status = await self.inspect(unit)
        if status.running:
            raise RunnerError(f"unit {unit} is still running after cleanup")

    async def _run(self, *args: str) -> tuple[int, str, str]:
        process = await asyncio.create_subprocess_exec(
            *args,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=self.command_timeout
            )
        except asyncio.TimeoutError:
            process.kill()
            await process.wait()
            raise
        except asyncio.CancelledError:
            process.kill()
            await process.wait()
            raise
        return (
            process.returncode,
            stdout.decode("utf-8", errors="replace"),
            stderr.decode("utf-8", errors="replace"),
        )


def _parse_int(value: str | None, default: int = 0) -> int:
    try:
        return int(value or default)
    except ValueError:
        return default
