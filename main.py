from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager

import aiosqlite
from fastapi import FastAPI, HTTPException
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from hardware import GpuManager
from models import (
    DB_PATH,
    FanConfig,
    TaskSubmit,
    delete_all_tasks,
    delete_task,
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


class VenvInfo(BaseModel):
    name: str
    path: str


class VenvScanResponse(BaseModel):
    venvs: list[VenvInfo]


class HealthResponse(BaseModel):
    status: str
    gpus_managed: int
    tasks_running: int
    tasks_pending: int


class WorkdirListResponse(BaseModel):
    directories: list[str]


class UsersListResponse(BaseModel):
    users: list[str]


# --- Helpers ---

def _scan_user_conda_envs(username: str) -> set[str]:
    user_home = f"/home/{username}"
    envs: set[str] = set()
    for conda_dir in ("anaconda3", "miniconda3", "miniforge3"):
        envs_path = os.path.join(user_home, conda_dir, "envs")
        if os.path.isdir(envs_path):
            for entry in os.listdir(envs_path):
                full = os.path.join(envs_path, entry)
                if os.path.isdir(full) and not entry.startswith("."):
                    envs.add(entry)
    conda_envs_file = os.path.join(user_home, ".conda", "environments.txt")
    if os.path.isfile(conda_envs_file):
        try:
            with open(conda_envs_file) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        name = os.path.basename(line)
                        if name and not name.startswith("."):
                            envs.add(name)
        except OSError:
            pass
    envs.add("base")
    return envs


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

    # Validate env_type
    if body.env_type is not None and body.env_type not in ("conda", "venv"):
        raise HTTPException(status_code=400, detail=f"Invalid env_type: {body.env_type}. Must be 'conda' or 'venv'.")

    # Validate conda_env if provided
    if body.conda_env is not None:
        if body.env_type == "venv":
            # For venv, conda_env holds the full path to the venv directory
            user_home = f"/home/{body.username}"
            real_home = os.path.realpath(user_home)
            real_env = os.path.realpath(body.conda_env)
            if not real_env.startswith(real_home + os.sep) and real_env != real_home:
                raise HTTPException(status_code=403, detail="venv must be within user's home directory")
            python_path = os.path.join(body.conda_env, "bin", "python")
            if not os.path.isfile(python_path):
                raise HTTPException(
                    status_code=400,
                    detail=f"venv not found or missing bin/python: {body.conda_env}",
                )
        else:
            # For conda (or unset), scan user's filesystem for conda installations
            user_home = f"/home/{body.username}"
            if not os.path.isdir(user_home):
                raise HTTPException(status_code=400, detail=f"User home not found: {user_home}")
            env_names = _scan_user_conda_envs(body.username)
            if body.conda_env not in env_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"Conda environment '{body.conda_env}' not found for user '{body.username}'. Available: {sorted(env_names)}",
                )

    task_id = await scheduler.submit_task(
        username=body.username,
        command=body.command,
        work_dir=body.work_dir,
        priority=body.priority,
        conda_env=body.conda_env,
        env_type=body.env_type,
        preferred_gpu_id=body.preferred_gpu_id,
    )
    return SubmitResponse(task_id=task_id, state="PENDING")


@app.get("/tasks/status", response_model=TaskListResponse)
async def get_task_status(state: str | None = None, username: str | None = None):
    tasks = await scheduler.get_tasks(state=state, username=username)
    for t in tasks:
        if t.state == "RUNNING":
            fraction, eta_seconds = scheduler.get_progress(t.id)
            t.progress = fraction
            t.eta = eta_seconds
    return TaskListResponse(tasks=[t.model_dump() for t in tasks])


@app.post("/tasks/{task_id}/abort", response_model=AbortResponse)
async def abort_task(task_id: str, username: str | None = None):
    row = await get_task_by_id(db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if row["state"] not in ("PENDING", "RUNNING"):
        raise HTTPException(status_code=400, detail=f"Cannot abort task in state {row['state']}")
    if username and row["username"] != username:
        raise HTTPException(status_code=403, detail=f"Task belongs to user '{row['username']}', not '{username}'")
    success = await scheduler.abort_task(task_id)
    return AbortResponse(success=success, task_id=task_id)


@app.delete("/tasks")
async def delete_all_tasks_endpoint(username: str | None = None):
    count = await delete_all_tasks(db, username)
    return {"deleted": count}


@app.delete("/tasks/{task_id}")
async def delete_task_endpoint(task_id: str):
    row = await get_task_by_id(db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if row["state"] in ("RUNNING", "PENDING"):
        raise HTTPException(status_code=400, detail=f"Cannot delete task in state {row['state']}")
    deleted = await delete_task(db, task_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete task")
    return {"success": True, "task_id": task_id}


@app.get("/gpus", response_model=GpuListResponse)
async def get_gpus():
    statuses = gpu_manager.get_all_gpu_status()
    return GpuListResponse(gpus=[s.model_dump() for s in statuses])


@app.post("/gpus/{gpu_id}/fan")
async def set_gpu_fan(gpu_id: int, body: FanConfig):
    if gpu_id not in gpu_manager.managed_gpu_ids:
        raise HTTPException(status_code=400, detail=f"Invalid GPU ID {gpu_id}")
    try:
        if body.mode == "auto":
            result = gpu_manager.set_fan_auto(gpu_id)
        elif body.mode == "manual":
            if body.speed is None:
                raise HTTPException(status_code=400, detail="speed is required when mode is 'manual'")
            result = gpu_manager.set_fan_speed(gpu_id, body.speed)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode}. Must be 'auto' or 'manual'.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    return result


@app.get("/users", response_model=UsersListResponse)
async def list_users():
    users = []
    try:
        with open("/etc/passwd") as f:
            for line in f:
                parts = line.strip().split(":")
                if len(parts) >= 6:
                    uid = int(parts[2])
                    home = parts[5]
                    username = parts[0]
                    if uid >= 1000 and os.path.isdir(home):
                        users.append(username)
    except OSError:
        pass
    return UsersListResponse(users=sorted(users))


@app.get("/conda/envs/{username}", response_model=CondaEnvListResponse)
async def list_conda_envs(username: str):
    user_home = f"/home/{username}"
    if not os.path.isdir(user_home):
        raise HTTPException(status_code=404, detail=f"User home directory not found: {user_home}")
    envs = _scan_user_conda_envs(username)
    return CondaEnvListResponse(environments=sorted(envs))


@app.get("/envs/scan-venvs", response_model=VenvScanResponse)
async def scan_venvs(path: str, username: str):
    """Scan a directory for Python venvs (directories containing pyvenv.cfg)."""
    user_home = f"/home/{username}"
    real_home = os.path.realpath(user_home)
    real_path = os.path.realpath(path)
    if not real_path.startswith(real_home + os.sep) and real_path != real_home:
        raise HTTPException(status_code=403, detail="Path must be within user's home directory")
    if not os.path.isdir(path):
        raise HTTPException(status_code=404, detail=f"Directory not found: {path}")
    venvs = []
    try:
        for entry in os.listdir(real_path):
            full = os.path.join(real_path, entry)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "pyvenv.cfg")):
                python_path = os.path.join(full, "bin", "python")
                if os.path.isfile(python_path):
                    venvs.append(VenvInfo(name=entry, path=full))
    except PermissionError:
        pass
    return VenvScanResponse(venvs=sorted(venvs, key=lambda v: v.name))


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


@app.get("/workdirs/{username}", response_model=WorkdirListResponse)
async def list_workdirs(username: str):
    root = f"/home/{username}"
    if not os.path.isdir(root):
        raise HTTPException(status_code=404, detail=f"User home directory not found: {root}")
    try:
        dirs = []
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                dirs.append(full)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: cannot read {root}")
    return WorkdirListResponse(directories=dirs)


@app.get("/workdirs/{username}/browse", response_model=WorkdirListResponse)
async def browse_workdirs(username: str, path: str = ""):
    user_home = f"/home/{username}"
    if path:
        root = os.path.join(user_home, path)
    else:
        root = user_home
    real_home = os.path.realpath(user_home)
    real_root = os.path.realpath(root)
    if not real_root.startswith(real_home + os.sep) and real_root != real_home:
        raise HTTPException(status_code=403, detail="Path must be within user's home directory")
    if not os.path.isdir(root):
        raise HTTPException(status_code=404, detail=f"Directory not found: {root}")
    try:
        dirs = []
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                dirs.append(os.path.join(path, entry) if path else entry)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: {root}")
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


# Static files mount must be after all API routes to avoid path conflicts
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
