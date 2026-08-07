from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import time

import aiosqlite

from hardware import GpuManager
from environments import resolve_python
from models import (
    claim_task_starting,
    clear_runner_unit,
    fail_pending_task,
    get_managed_tasks,
    get_pending_tasks,
    get_task_by_id,
    get_tasks,
    set_gpu_scheduling_enabled as persist_gpu_scheduling_enabled,
    row_to_status,
    reset_starting_task,
    set_task_aborted,
    set_task_finished,
    set_task_running,
    TaskStatus,
)
from systemd_runner import RunnerError, SupervisorUnavailable, SystemdRunner, UnitStatus

logger = logging.getLogger("scheduler")

POLL_INTERVAL_SECONDS = 3.0
LOGS_DIR = "logs"


def _find_conda_python(username: str, conda_env: str) -> str | None:
    return resolve_python(username, conda_env, "conda")


def _find_python(username: str | None, env_name: str | None, env_type: str | None) -> str | None:
    if env_name is None or username is None:
        return None
    return resolve_python(username, env_name, env_type)


def _wrap_command(command: str, conda_env: str | None, username: str | None = None, gpu_id: int | None = None, work_dir: str | None = None, env_type: str | None = None) -> str:
    if conda_env is not None and username is not None:
        python_path = _find_python(username, conda_env, env_type)
        if python_path is None:
            raise ValueError(f"Python environment is no longer available: {conda_env}")
        command = re.sub(r'\bpython3?\b', shlex.quote(python_path), command)
    if gpu_id is not None:
        command = f"export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES={gpu_id} && {command}"
    if work_dir:
        command = f"cd {shlex.quote(work_dir)} && {command}"
    if username is not None:
        return f"sudo -u {shlex.quote(username)} bash -l -c {shlex.quote(command)}"
    return command


def _prepare_command(
    command: str,
    conda_env: str | None,
    username: str,
    env_type: str | None,
) -> str:
    if conda_env is None:
        return command
    python_path = _find_python(username, conda_env, env_type)
    if python_path is None:
        raise ValueError(f"Python environment is no longer available: {conda_env}")
    return re.sub(r"\bpython3?\b", shlex.quote(python_path), command)


def _parse_log_progress(log_path: str) -> tuple[float | None, float | None]:
    """Parse the tail of a training log for rsl_rl progress/ETA patterns.

    Returns (fraction, eta_seconds) or (None, None) if no rsl_rl output found.
    """
    try:
        with open(log_path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - 4096))
            tail = f.read().decode("utf-8", errors="replace")
    except (OSError, ValueError):
        return None, None

    fraction = None
    eta_seconds = None

    for line in reversed(tail.splitlines()):
        if eta_seconds is None:
            m = re.search(r"ETA:\s+(\d+):(\d+):(\d+)", line)
            if m:
                eta_seconds = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3))
        if fraction is None:
            m = re.search(r"Learning iteration\s+(\d+)/(\d+)", line)
            if m:
                it, total = int(m.group(1)), int(m.group(2))
                if total > 0:
                    fraction = it / total
        if fraction is not None and eta_seconds is not None:
            break

    return fraction, eta_seconds


class Scheduler:
    def __init__(
        self,
        db: aiosqlite.Connection,
        gpu_manager: GpuManager,
        poll_interval: float = POLL_INTERVAL_SECONDS,
        runner: SystemdRunner | None = None,
    ):
        self.db = db
        self.gpu_manager = gpu_manager
        self.poll_interval = poll_interval
        self.runner = runner or SystemdRunner()
        self.supervisor_available = False
        self._running = False
        self._loop_task: asyncio.Task | None = None
        self._wake_event = asyncio.Event()
        self._lifecycle_lock = asyncio.Lock()
        # task_id -> (fraction, eta_seconds)
        self._progress: dict[str, tuple[float | None, float | None]] = {}
        os.makedirs(LOGS_DIR, exist_ok=True)

    async def start(self) -> None:
        self._running = True
        self.supervisor_available = await self.runner.is_available()
        if self.supervisor_available:
            await self._reconcile_managed_tasks()
            await self._cleanup_terminal_units()
        else:
            logger.error("systemd supervisor unavailable; dispatch is paused")
        self._wake_event.set()
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started (poll interval: %.1fs)", self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        self._wake_event.set()
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        self._progress.clear()
        logger.info("Scheduler stopped; managed task units were left running")

    async def _run_loop(self) -> None:
        while self._running:
            self._wake_event.clear()
            try:
                self.supervisor_available = await self.runner.is_available()
                if self.supervisor_available:
                    await self._reconcile_managed_tasks()
                    if self.supervisor_available:
                        await self._dispatch_pending_tasks()
            except Exception:
                logger.exception("Error in scheduler loop")
            try:
                await asyncio.wait_for(self._wake_event.wait(), timeout=self.poll_interval)
            except asyncio.TimeoutError:
                pass

    async def _dispatch_pending_tasks(self) -> None:
        pending = await get_pending_tasks(self.db)
        if not pending:
            return
        available = set(await asyncio.to_thread(self.gpu_manager.get_available_gpu_ids))
        for row in pending:
            preferred = row["preferred_gpu_id"]
            if preferred is not None:
                # User pinned a specific GPU — only use it if available
                if preferred not in available:
                    logger.info(
                        "Task %s waiting: preferred GPU %d not available", row["id"], preferred,
                    )
                    continue
                gpu_id = preferred
            else:
                gpu_id = next((gid for gid in self.gpu_manager.managed_gpu_ids if gid in available), None)
                if gpu_id is None:
                    logger.info(
                        "Task %s waiting: no GPU available (checked %s)",
                        row["id"], self.gpu_manager.managed_gpu_ids,
                    )
                    break
            task_id = row["id"]
            command = row["command"]
            work_dir = row["work_dir"]
            conda_env = row["conda_env"]
            env_type = row["env_type"]
            username = row["username"]
            running = await self._spawn_task(
                task_id, command, work_dir, gpu_id, conda_env, env_type, username
            )
            if running:
                available.discard(gpu_id)

    async def _spawn_task(
        self,
        task_id: str,
        command: str,
        work_dir: str,
        gpu_id: int,
        conda_env: str | None = None,
        env_type: str | None = None,
        username: str | None = None,
    ) -> bool:
        if username is None:
            await fail_pending_task(self.db, task_id, "missing_username")
            return False
        timestamp = int(time.time())
        log_path = os.path.abspath(os.path.join(LOGS_DIR, f"{task_id}_{timestamp}.log"))
        unit = self.runner.unit_name(task_id)
        try:
            prepared = _prepare_command(command, conda_env, username, env_type)
        except ValueError as exc:
            await fail_pending_task(self.db, task_id, str(exc))
            return False

        async with self._lifecycle_lock:
            if not self.gpu_manager.is_scheduling_enabled(gpu_id):
                logger.info("Task %s waiting: GPU %d scheduling is disabled", task_id, gpu_id)
                return False
            claimed = await claim_task_starting(self.db, task_id, gpu_id, log_path, unit)
            if not claimed:
                return False
            open(log_path, "ab").close()
            try:
                status = await self.runner.launch(
                    unit=unit,
                    task_id=task_id,
                    username=username,
                    command=prepared,
                    work_dir=work_dir,
                    gpu_id=gpu_id,
                    log_path=log_path,
                )
            except SupervisorUnavailable:
                self.supervisor_available = False
                await reset_starting_task(self.db, task_id, "supervisor_unavailable")
                _remove_file(log_path)
                return False
            except RunnerError as exc:
                try:
                    status = await self.runner.inspect(unit)
                except SupervisorUnavailable:
                    self.supervisor_available = False
                    await reset_starting_task(self.db, task_id, "supervisor_unavailable")
                    _remove_file(log_path)
                    return False
                if not status.exists:
                    _append_log(log_path, f"systemd launch failed: {exc}\n")
                    await set_task_finished(self.db, task_id, -1, "systemd_launch_error")
                    return False
            row = await get_task_by_id(self.db, task_id)
            if row is None or row["state"] not in ("STARTING", "RUNNING"):
                await self.runner.stop(unit)
                return False
            await self._apply_unit_status(row, status)
            return status.running

    async def _reconcile_managed_tasks(self) -> None:
        async with self._lifecycle_lock:
            for row in await get_managed_tasks(self.db):
                unit = row.get("runner_unit")
                if not unit:
                    if row["state"] == "STARTING":
                        await reset_starting_task(self.db, row["id"], "missing_runner_unit")
                    else:
                        await set_task_finished(
                            self.db, row["id"], -1, "legacy_runner_unrecoverable"
                        )
                    self.gpu_manager.unregister_task(row["id"])
                    continue
                try:
                    status = await self.runner.inspect(unit)
                except SupervisorUnavailable:
                    self.supervisor_available = False
                    return
                await self._apply_unit_status(row, status)

    async def _apply_unit_status(self, row: dict, status: UnitStatus) -> None:
        task_id = row["id"]
        if not status.exists:
            if row["state"] == "STARTING":
                await reset_starting_task(self.db, task_id, "runner_missing_requeued")
                _remove_file(row.get("log_path"))
            else:
                await set_task_finished(self.db, task_id, -1, "runner_missing_or_stopped")
            self.gpu_manager.unregister_task(task_id)
            self._progress.pop(task_id, None)
            return
        if status.running:
            if row["state"] == "STARTING":
                await set_task_running(self.db, task_id, status.main_pid)
            if row.get("gpu_id") is not None:
                self.gpu_manager.register_task(
                    task_id, row["gpu_id"], status.main_pid or None
                )
            if row.get("log_path"):
                self._progress[task_id] = await asyncio.to_thread(
                    _parse_log_progress, row["log_path"]
                )
            return

        exit_code = status.exit_status
        if status.result != "success" and exit_code == 0:
            exit_code = -1
        reason = None if exit_code == 0 else f"systemd_{status.result}"
        await set_task_finished(self.db, task_id, exit_code, reason)
        self.gpu_manager.unregister_task(task_id)
        self._progress.pop(task_id, None)
        try:
            await self.runner.cleanup(status.unit)
            await clear_runner_unit(self.db, task_id)
        except (RunnerError, SupervisorUnavailable):
            logger.warning("Could not clean up unit %s", status.unit)

    async def _cleanup_terminal_units(self) -> None:
        for state in ("COMPLETED", "FAILED"):
            for row in await get_tasks(self.db, state=state):
                if row.get("runner_unit"):
                    try:
                        await self.runner.cleanup(row["runner_unit"])
                        await clear_runner_unit(self.db, row["id"])
                    except (RunnerError, SupervisorUnavailable):
                        return

    async def submit_task(
        self, username: str, command: str, work_dir: str, priority: int = 0,
        conda_env: str | None = None, env_type: str | None = None,
        preferred_gpu_id: int | None = None,
    ) -> str:
        from models import insert_task, TaskSubmit
        submit = TaskSubmit(
            username=username, command=command, work_dir=work_dir, priority=priority,
            conda_env=conda_env, env_type=env_type, preferred_gpu_id=preferred_gpu_id,
        )
        task_id = await insert_task(self.db, submit)
        self._wake_event.set()
        logger.info(
            "Task submitted: %s (user=%s, cmd=%r, env=%s/%s, gpu=%s, priority=%d)",
            task_id, username, command, env_type, conda_env, preferred_gpu_id, priority,
        )
        return task_id

    async def set_gpu_scheduling_enabled(self, gpu_id: int, enabled: bool) -> None:
        async with self._lifecycle_lock:
            bus_id = self.gpu_manager.bus_id_map[gpu_id]
            await persist_gpu_scheduling_enabled(self.db, bus_id, enabled)
            self.gpu_manager.set_scheduling_enabled(gpu_id, enabled)
        self._wake_event.set()
        logger.info(
            "GPU %d scheduling %s", gpu_id, "enabled" if enabled else "disabled"
        )

    async def abort_task(self, task_id: str) -> bool:
        async with self._lifecycle_lock:
            row = await get_task_by_id(self.db, task_id)
            if row is None or row["state"] not in ("STARTING", "RUNNING", "PENDING"):
                return False
            if row["state"] in ("STARTING", "RUNNING"):
                if not await self.is_supervisor_available():
                    raise SupervisorUnavailable("systemd is unavailable")
                if row.get("runner_unit"):
                    await self.runner.stop(row["runner_unit"])
            updated = await set_task_aborted(self.db, task_id)
            if not updated:
                return False
            if row.get("runner_unit"):
                try:
                    await self.runner.cleanup(row["runner_unit"])
                    await clear_runner_unit(self.db, task_id)
                except (RunnerError, SupervisorUnavailable):
                    logger.warning("Could not clean up aborted unit %s", row["runner_unit"])
            self.gpu_manager.unregister_task(task_id)
            self._progress.pop(task_id, None)
        self._wake_event.set()
        logger.info("Task %s aborted (was %s)", task_id, row["state"])
        return True

    async def is_supervisor_available(self) -> bool:
        self.supervisor_available = await self.runner.is_available()
        return self.supervisor_available

    async def get_tasks(
        self, state: str | None = None, username: str | None = None,
        limit: int | None = None, offset: int = 0, query: str | None = None,
    ) -> list[TaskStatus]:
        rows = await get_tasks(self.db, state, username, limit, offset, query)
        return [row_to_status(r) for r in rows]

    def get_progress(self, task_id: str) -> tuple[float | None, float | None]:
        return self._progress.get(task_id, (None, None))


def _remove_file(path: str | None) -> None:
    if not path:
        return
    try:
        os.remove(path)
    except FileNotFoundError:
        pass


def _append_log(path: str, message: str) -> None:
    try:
        with open(path, "a") as file_handle:
            file_handle.write(message)
    except OSError:
        pass
