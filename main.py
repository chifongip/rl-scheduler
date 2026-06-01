from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hardware import GpuManager
from models import (
    DB_PATH,
    TaskSubmit,
    get_task_by_id,
    init_db,
    row_to_status,
)
from scheduler import Scheduler

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

# Global references, initialized in lifespan
db: aiosqlite.Connection = None  # type: ignore[assignment]
gpu_manager: GpuManager = None  # type: ignore[assignment]
scheduler: Scheduler = None  # type: ignore[assignment]


@asynccontextmanager
async def lifespan(app: FastAPI):
    global db, gpu_manager, scheduler
    logger.info("Starting rl-scheduler...")
    db = await init_db(DB_PATH)
    gpu_manager = GpuManager()
    scheduler = Scheduler(db, gpu_manager)
    await scheduler.start()
    logger.info("rl-scheduler ready")
    yield
    logger.info("Shutting down rl-scheduler...")
    await scheduler.stop()
    gpu_manager.shutdown()
    await db.close()
    logger.info("rl-scheduler stopped")


app = FastAPI(title="rl-scheduler", lifespan=lifespan)


@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")


# --- Response models ---

class SubmitResponse(BaseModel):
    task_id: str
    state: str


class AbortResponse(BaseModel):
    success: bool
    task_id: str


class TaskListResponse(BaseModel):
    tasks: list


class GpuListResponse(BaseModel):
    gpus: list


class CondaEnvListResponse(BaseModel):
    environments: list[str]


class HealthResponse(BaseModel):
    status: str
    gpus_managed: int
    tasks_running: int
    tasks_pending: int


class WorkdirListResponse(BaseModel):
    directories: list[str]


# --- Endpoints ---

@app.post("/tasks/submit", response_model=SubmitResponse)
async def submit_task(body: TaskSubmit):
    # Validate work_dir exists
    if not os.path.isdir(body.work_dir):
        raise HTTPException(status_code=400, detail=f"work_dir does not exist: {body.work_dir}")

    # Validate preferred_gpu_id if provided
    if body.preferred_gpu_id is not None:
        if body.preferred_gpu_id not in gpu_manager.managed_gpu_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GPU ID {body.preferred_gpu_id}. Managed GPUs: {gpu_manager.managed_gpu_ids}",
            )

    # Validate conda_env if provided
    if body.conda_env is not None:
        if not shutil.which("conda"):
            raise HTTPException(status_code=400, detail="conda is not installed or not on PATH")
        try:
            result = subprocess.run(
                ["conda", "env", "list", "--json"],
                capture_output=True, text=True, timeout=10,
            )
            env_data = json.loads(result.stdout)
            env_names = [os.path.basename(p) for p in env_data.get("envs", [])]
            # "base" is always valid even if listed as prefix
            if body.conda_env not in env_names and body.conda_env != "base":
                raise HTTPException(
                    status_code=400,
                    detail=f"Conda environment '{body.conda_env}' not found. Available: {env_names}",
                )
        except FileNotFoundError:
            raise HTTPException(status_code=400, detail="conda command not found")
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
            raise HTTPException(status_code=500, detail=f"Failed to query conda environments: {e}")

    task_id = await scheduler.submit_task(
        command=body.command,
        work_dir=body.work_dir,
        priority=body.priority,
        conda_env=body.conda_env,
        preferred_gpu_id=body.preferred_gpu_id,
    )
    return SubmitResponse(task_id=task_id, state="PENDING")


@app.get("/tasks/status", response_model=TaskListResponse)
async def get_task_status(state: str | None = None):
    tasks = await scheduler.get_tasks(state=state)
    return TaskListResponse(tasks=[t.model_dump() for t in tasks])


@app.post("/tasks/{task_id}/abort", response_model=AbortResponse)
async def abort_task(task_id: str):
    row = await get_task_by_id(db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if row["state"] != "RUNNING":
        raise HTTPException(status_code=400, detail=f"Task is not RUNNING (state={row['state']})")
    success = await scheduler.abort_task(task_id)
    return AbortResponse(success=success, task_id=task_id)


@app.get("/gpus", response_model=GpuListResponse)
async def get_gpus():
    statuses = gpu_manager.get_all_gpu_status()
    return GpuListResponse(gpus=[s.model_dump() for s in statuses])


@app.get("/conda/envs", response_model=CondaEnvListResponse)
async def list_conda_envs():
    try:
        result = subprocess.run(
            ["conda", "env", "list", "--json"],
            capture_output=True, text=True, timeout=10,
        )
        env_data = json.loads(result.stdout)
        env_names = sorted({os.path.basename(p) for p in env_data.get("envs", [])})
        return CondaEnvListResponse(environments=env_names)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="conda is not installed or not on PATH")
    except (subprocess.TimeoutExpired, json.JSONDecodeError) as e:
        raise HTTPException(status_code=500, detail=f"Failed to query conda: {e}")


@app.get("/health", response_model=HealthResponse)
async def health():
    running = await scheduler.get_tasks(state="RUNNING")
    pending = await scheduler.get_tasks(state="PENDING")
    return HealthResponse(
        status="ok",
        gpus_managed=len(gpu_manager.managed_gpu_ids),
        tasks_running=len(running),
        tasks_pending=len(pending),
    )


@app.get("/workdirs", response_model=WorkdirListResponse)
async def list_workdirs():
    root = os.environ.get("RLS_WORKDIR_ROOT", os.path.expanduser("~"))
    dirs = []
    try:
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                dirs.append(full)
    except OSError:
        pass
    return WorkdirListResponse(directories=dirs)


@app.get("/tasks/{task_id}/log")
async def get_task_log(task_id: str):
    row = await get_task_by_id(db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    log_path = row["log_path"]
    if log_path is None or not os.path.isfile(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")
    with open(log_path, "r") as f:
        content = f.read()
    return PlainTextResponse(content)


# Need shutil for conda check
import shutil

# Static files mount must be after all API routes to avoid path conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
