from __future__ import annotations

import asyncio
import logging
import os
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
    set_task_finished,
    set_task_running,
    TaskStatus,
)

logger = logging.getLogger("scheduler")

POLL_INTERVAL_SECONDS = 3.0
LOGS_DIR = "logs"


def _wrap_command(command: str, conda_env: str | None) -> str:
    if conda_env is None:
        return command
    return f"conda run -n {conda_env} --no-capture-output --live-stream {command}"


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
        # task_id -> (asyncio.Process, file_handle)
        self._processes: dict[str, tuple[asyncio.subprocess.Process, object]] = {}
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
        for row in pending:
            preferred = row["preferred_gpu_id"]
            if preferred is not None:
                # User pinned a specific GPU — only use it if available
                if not self.gpu_manager.is_gpu_available(preferred):
                    continue
                gpu_id = preferred
            else:
                gpu_id = self.gpu_manager.find_available_gpu()
                if gpu_id is None:
                    break
            task_id = row["id"]
            command = row["command"]
            work_dir = row["work_dir"]
            conda_env = row["conda_env"]
            await self._spawn_task(task_id, command, work_dir, gpu_id, conda_env)

    async def _spawn_task(
        self,
        task_id: str,
        command: str,
        work_dir: str,
        gpu_id: int,
        conda_env: str | None = None,
    ) -> None:
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        wrapped = _wrap_command(command, conda_env)
        timestamp = int(time.time())
        log_path = os.path.join(LOGS_DIR, f"{task_id}_{timestamp}.log")
        log_fh = open(log_path, "w")

        process = await asyncio.create_subprocess_shell(
            wrapped,
            cwd=work_dir,
            env=env,
            stdout=log_fh,
            stderr=asyncio.subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        self._processes[task_id] = (process, log_fh)
        self.gpu_manager.register_task(task_id, gpu_id)
        await set_task_running(self.db, task_id, gpu_id, process.pid, log_path)
        logger.info(
            "Task %s dispatched → GPU %d (pid=%d, conda=%s, log=%s)",
            task_id, gpu_id, process.pid, conda_env or "none", log_path,
        )

    async def _check_running_tasks(self) -> None:
        for task_id in list(self._processes):
            process, log_fh = self._processes[task_id]
            if process.returncode is not None:
                await self._finalize_task(task_id, process.returncode)

    async def _finalize_task(self, task_id: str, exit_code: int) -> None:
        process, log_fh = self._processes.pop(task_id)
        log_fh.close()
        self.gpu_manager.unregister_task(task_id)
        await set_task_finished(self.db, task_id, exit_code)
        status = "COMPLETED" if exit_code == 0 else "FAILED"
        logger.info("Task %s %s (exit_code=%d)", task_id, status, exit_code)

    async def _kill_process(self, task_id: str) -> None:
        if task_id not in self._processes:
            return
        process, log_fh = self._processes.pop(task_id)
        try:
            pgid = os.getpgid(process.pid)
            os.killpg(pgid, signal.SIGKILL)
            logger.info("Killed process group for task %s (pgid=%d)", task_id, pgid)
        except (ProcessLookupError, OSError):
            pass
        log_fh.close()
        self.gpu_manager.unregister_task(task_id)

    async def submit_task(
        self, command: str, work_dir: str, priority: int = 0,
        conda_env: str | None = None, preferred_gpu_id: int | None = None,
    ) -> str:
        from models import insert_task, TaskSubmit
        submit = TaskSubmit(
            command=command, work_dir=work_dir, priority=priority,
            conda_env=conda_env, preferred_gpu_id=preferred_gpu_id,
        )
        task_id = await insert_task(self.db, submit)
        logger.info(
            "Task submitted: %s (cmd=%r, conda=%s, gpu=%s, priority=%d)",
            task_id, command, conda_env, preferred_gpu_id, priority,
        )
        return task_id

    async def abort_task(self, task_id: str) -> bool:
        row = await get_task_by_id(self.db, task_id)
        if row is None:
            return False
        if row["state"] != "RUNNING":
            return False
        await self._kill_process(task_id)
        await set_task_finished(self.db, task_id, -9)
        logger.info("Task %s aborted", task_id)
        return True

    async def get_tasks(self, state: str | None = None) -> list[TaskStatus]:
        rows = await get_tasks(self.db, state)
        return [row_to_status(r) for r in rows]
