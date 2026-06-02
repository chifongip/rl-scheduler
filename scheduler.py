from __future__ import annotations

import asyncio
import logging
import os
import re
import shlex
import shutil
import signal
import time

import aiosqlite

from hardware import GpuManager
from models import (
    get_pending_tasks,
    get_task_by_id,
    get_tasks,
    row_to_status,
    set_task_aborted,
    set_task_finished,
    set_task_running,
    TaskStatus,
)

logger = logging.getLogger("scheduler")

POLL_INTERVAL_SECONDS = 3.0
LOGS_DIR = "logs"


def _find_conda_python(username: str, conda_env: str) -> str | None:
    for conda_dir in ("miniconda3", "anaconda3", "miniforge3"):
        python_path = f"/home/{username}/{conda_dir}/envs/{conda_env}/bin/python"
        if os.path.isfile(python_path):
            return python_path
    return None


def _find_python(username: str | None, env_name: str | None, env_type: str | None) -> str | None:
    if env_name is None or username is None:
        return None
    if env_type == "venv":
        # env_name is the full path to the venv directory — must be under user's home
        user_home = f"/home/{username}"
        real_home = os.path.realpath(user_home)
        real_env = os.path.realpath(env_name)
        if not real_env.startswith(real_home + os.sep) and real_env != real_home:
            return None
        python_path = os.path.join(env_name, "bin", "python")
        return python_path if os.path.isfile(python_path) else None
    # Default: conda resolution
    return _find_conda_python(username, env_name)


def _wrap_command(command: str, conda_env: str | None, username: str | None = None, gpu_id: int | None = None, work_dir: str | None = None, env_type: str | None = None) -> str:
    if conda_env is not None and username is not None:
        python_path = _find_python(username, conda_env, env_type)
        if python_path:
            command = re.sub(r'\bpython3?\b', shlex.quote(python_path), command)
    if gpu_id is not None:
        command = f"export CUDA_DEVICE_ORDER=PCI_BUS_ID && export CUDA_VISIBLE_DEVICES={gpu_id} && {command}"
    if work_dir:
        command = f"cd {shlex.quote(work_dir)} && {command}"
    if username is not None:
        return f"sudo -u {shlex.quote(username)} bash -l -c {shlex.quote(command)}"
    return command


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
    ):
        self.db = db
        self.gpu_manager = gpu_manager
        self.poll_interval = poll_interval
        self._running = False
        self._loop_task: asyncio.Task | None = None
        # task_id -> (asyncio.Process, file_handle, log_path)
        self._processes: dict[str, tuple[asyncio.subprocess.Process, object, str]] = {}
        # task_id -> (fraction, eta_seconds)
        self._progress: dict[str, tuple[float | None, float | None]] = {}
        os.makedirs(LOGS_DIR, exist_ok=True)

    async def start(self) -> None:
        self._running = True
        self._loop_task = asyncio.create_task(self._run_loop())
        logger.info("Scheduler started (poll interval: %.1fs)", self.poll_interval)

    async def stop(self) -> None:
        self._running = False
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        for task_id in list(self._processes):
            await self._kill_process(task_id)
        self._progress.clear()
        logger.info("Scheduler stopped")

    async def _run_loop(self) -> None:
        while self._running:
            try:
                await self._dispatch_pending_tasks()
                await self._check_running_tasks()
            except Exception:
                logger.exception("Error in scheduler loop")
            await asyncio.sleep(self.poll_interval)

    async def _dispatch_pending_tasks(self) -> None:
        pending = await get_pending_tasks(self.db)
        if not pending:
            return
        for row in pending:
            preferred = row["preferred_gpu_id"]
            if preferred is not None:
                # User pinned a specific GPU — only use it if available
                if not self.gpu_manager.is_gpu_available(preferred):
                    logger.info(
                        "Task %s waiting: preferred GPU %d not available", row["id"], preferred,
                    )
                    continue
                gpu_id = preferred
            else:
                gpu_id = self.gpu_manager.find_available_gpu()
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
            await self._spawn_task(task_id, command, work_dir, gpu_id, conda_env, env_type, username)

    async def _spawn_task(
        self,
        task_id: str,
        command: str,
        work_dir: str,
        gpu_id: int,
        conda_env: str | None = None,
        env_type: str | None = None,
        username: str | None = None,
    ) -> None:
        env = os.environ.copy()

        timestamp = int(time.time())
        log_path = os.path.join(LOGS_DIR, f"{task_id}_{timestamp}.log")
        log_fh = open(log_path, "w")
        try:
            wrapped = _wrap_command(command, conda_env, username, gpu_id, work_dir, env_type)
            process = await asyncio.create_subprocess_shell(
                wrapped,
                env=env,
                stdout=log_fh,
                stderr=asyncio.subprocess.STDOUT,
                preexec_fn=os.setsid,
            )
        except Exception:
            log_fh.close()
            logger.exception("Failed to spawn task %s", task_id)
            await set_task_aborted(self.db, task_id)
            return

        self._processes[task_id] = (process, log_fh, log_path)
        self.gpu_manager.register_task(task_id, gpu_id)
        updated = await set_task_running(self.db, task_id, gpu_id, process.pid, log_path)
        if not updated:
            # Task was aborted or already dispatched in another codepath
            await self._kill_process(task_id)
            logger.info("Task %s was aborted before dispatch completed — cleaned up", task_id)
            return
        logger.info(
            "Task %s dispatched → GPU %d (pid=%d, env=%s/%s, log=%s)",
            task_id, gpu_id, process.pid, env_type or "none", conda_env or "none", log_path,
        )

    async def _check_running_tasks(self) -> None:
        for task_id in list(self._processes):
            process, _, log_path = self._processes[task_id]
            if process.returncode is not None:
                await self._finalize_task(task_id, process.returncode)
            else:
                # Parse log tail for progress/ETA (rsl_rl format)
                self._progress[task_id] = await asyncio.to_thread(_parse_log_progress, log_path)

    async def _finalize_task(self, task_id: str, exit_code: int) -> None:
        updated = await set_task_finished(self.db, task_id, exit_code)
        if not updated:
            logger.info("Task %s was already finalized (likely aborted)", task_id)
            return
        process, log_fh, _ = self._processes.pop(task_id, (None, None, None))
        if process is not None:
            log_fh.close()
        self._progress.pop(task_id, None)
        self.gpu_manager.unregister_task(task_id)
        status = "COMPLETED" if exit_code == 0 else "FAILED"
        logger.info("Task %s %s (exit_code=%d)", task_id, status, exit_code)

    async def _kill_process(self, task_id: str) -> None:
        if task_id not in self._processes:
            return
        process, log_fh, _ = self._processes.pop(task_id)
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
            logger.info("Killed process group for task %s (pgid=%d)", task_id, pgid)
        except (ProcessLookupError, OSError):
            pass
        log_fh.close()
        self.gpu_manager.unregister_task(task_id)
        self._progress.pop(task_id, None)

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
        logger.info(
            "Task submitted: %s (user=%s, cmd=%r, env=%s/%s, gpu=%s, priority=%d)",
            task_id, username, command, env_type, conda_env, preferred_gpu_id, priority,
        )
        return task_id

    async def abort_task(self, task_id: str) -> bool:
        row = await get_task_by_id(self.db, task_id)
        if row is None:
            return False
        if row["state"] == "RUNNING":
            await self._kill_process(task_id)
        elif row["state"] == "PENDING":
            pass  # No process to kill, just update state
        else:
            return False
        await set_task_aborted(self.db, task_id)
        logger.info("Task %s aborted (was %s)", task_id, row["state"])
        return True

    async def get_tasks(self, state: str | None = None, username: str | None = None) -> list[TaskStatus]:
        rows = await get_tasks(self.db, state, username)
        return [row_to_status(r) for r in rows]

    def get_progress(self, task_id: str) -> tuple[float | None, float | None]:
        return self._progress.get(task_id, (None, None))
