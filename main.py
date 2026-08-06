from __future__ import annotations

import asyncio
import logging
import os
import pwd
import secrets
from contextlib import asynccontextmanager
from collections.abc import Callable

import aiosqlite
from fastapi import APIRouter, FastAPI, Header, HTTPException, Query, Request, Response
from fastapi.responses import PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from admin_sessions import (
    AdminSessionStore,
    DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS,
    parse_admin_session_timeout,
)
from environments import get_user_home, is_within, list_conda_environments, resolve_python
from hardware import GpuManager
from models import (
    DB_PATH,
    FanConfig,
    TaskSubmit,
    TaskStatus,
    count_tasks,
    delete_all_tasks,
    delete_task,
    get_task_by_id,
    init_db,
    row_to_status,
)
from scheduler import Scheduler
from systemd_runner import SupervisorUnavailable

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("main")

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting rl-scheduler...")
    app.state.admin_password = os.environ.get("ADMIN_PASSWORD", "")
    raw_timeout = os.environ.get("ADMIN_SESSION_TIMEOUT_SECONDS")
    timeout = parse_admin_session_timeout(raw_timeout)
    invalid_timeout = False
    if raw_timeout is not None:
        try:
            invalid_timeout = int(raw_timeout) != timeout
        except ValueError:
            invalid_timeout = True
    if invalid_timeout:
        logger.warning(
            "Invalid ADMIN_SESSION_TIMEOUT_SECONDS=%r; using %d seconds",
            raw_timeout,
            DEFAULT_ADMIN_SESSION_TIMEOUT_SECONDS,
        )
    app.state.admin_sessions = app.state.admin_session_store_factory(timeout)
    app.state.db = await init_db(app.state.db_path)
    app.state.gpu_manager = app.state.gpu_factory()
    app.state.scheduler = app.state.scheduler_factory(
        app.state.db, app.state.gpu_manager
    )
    await app.state.scheduler.start()
    logger.info("rl-scheduler ready")
    yield
    logger.info("Shutting down rl-scheduler...")
    await app.state.scheduler.stop()
    await asyncio.to_thread(app.state.gpu_manager.shutdown)
    await app.state.db.close()
    logger.info("rl-scheduler stopped")


router = APIRouter()


@router.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/static/index.html")


# --- Response models ---

class SubmitResponse(BaseModel):
    task_id: str
    state: str


class AbortResponse(BaseModel):
    success: bool
    task_id: str


class AdminSessionRequest(BaseModel):
    password: str


class AdminSessionResponse(BaseModel):
    token: str
    expires_in: int


class TaskListResponse(BaseModel):
    tasks: list[TaskStatus]
    total: int
    limit: int | None = None
    offset: int = 0


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
    tasks_starting: int
    supervisor_available: bool
    admin_enabled: bool


class WorkdirListResponse(BaseModel):
    directories: list[str]


class UsersListResponse(BaseModel):
    users: list[str]
    home_directories: dict[str, str]


# --- Helpers ---

def _scan_user_conda_envs(username: str) -> set[str]:
    return set(list_conda_environments(username))


def _require_admin(
    request: Request,
    username: str | None,
    admin_password: str | None,
    header_password: str | None = None,
    header_token: str | None = None,
) -> str | None:
    if username:
        return None
    configured_password = request.app.state.admin_password
    if not configured_password:
        raise HTTPException(status_code=403, detail="Admin mode is disabled (no ADMIN_PASSWORD set)")
    if header_token is not None:
        if request.app.state.admin_sessions.validate(header_token):
            return header_token
        raise HTTPException(status_code=403, detail="Invalid or expired admin session")
    supplied_password = header_password if header_password is not None else admin_password
    if supplied_password is None or not secrets.compare_digest(supplied_password, configured_password):
        raise HTTPException(status_code=403, detail="Invalid admin password")
    return None


def _touch_admin_session(request: Request, token: str | None) -> None:
    if token is not None:
        request.app.state.admin_sessions.touch(token)


# --- Endpoints ---

@router.post("/admin/session", response_model=AdminSessionResponse)
async def create_admin_session(body: AdminSessionRequest, request: Request, response: Response):
    configured_password = request.app.state.admin_password
    if not configured_password:
        raise HTTPException(status_code=403, detail="Admin mode is disabled (no ADMIN_PASSWORD set)")
    if not secrets.compare_digest(body.password, configured_password):
        raise HTTPException(status_code=403, detail="Invalid admin password")
    token = request.app.state.admin_sessions.create()
    response.headers["Cache-Control"] = "no-store"
    return AdminSessionResponse(
        token=token,
        expires_in=request.app.state.admin_sessions.timeout_seconds,
    )


@router.delete("/admin/session")
async def delete_admin_session(
    request: Request,
    x_admin_token: str | None = Header(default=None),
):
    if x_admin_token is None or not request.app.state.admin_sessions.validate(x_admin_token):
        raise HTTPException(status_code=403, detail="Invalid or expired admin session")
    request.app.state.admin_sessions.revoke(x_admin_token)
    return {"success": True}

@router.post("/tasks/submit", response_model=SubmitResponse)
async def submit_task(body: TaskSubmit, request: Request):
    if not await request.app.state.scheduler.is_supervisor_available():
        raise HTTPException(
            status_code=503,
            detail="Task supervisor is unavailable; submission is temporarily disabled",
        )
    user_home = get_user_home(body.username)
    if user_home is None:
        raise HTTPException(status_code=400, detail=f"System user not found: {body.username}")
    # Validate work_dir exists
    if not os.path.isdir(body.work_dir):
        raise HTTPException(status_code=400, detail=f"work_dir does not exist: {body.work_dir}")

    # Validate preferred_gpu_id if provided
    if body.preferred_gpu_id is not None:
        if body.preferred_gpu_id not in request.app.state.gpu_manager.managed_gpu_ids:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid GPU ID {body.preferred_gpu_id}. Managed GPUs: {request.app.state.gpu_manager.managed_gpu_ids}",
            )

    # Validate env_type
    if body.env_type is not None and body.env_type not in ("conda", "venv"):
        raise HTTPException(status_code=400, detail=f"Invalid env_type: {body.env_type}. Must be 'conda' or 'venv'.")

    # Validate conda_env if provided
    if body.conda_env is not None:
        if body.env_type == "venv":
            # For venv, conda_env holds the full path to the venv directory
            if not is_within(body.conda_env, user_home):
                raise HTTPException(status_code=403, detail="venv must be within user's home directory")
            if resolve_python(body.username, body.conda_env, "venv") is None:
                raise HTTPException(
                    status_code=400,
                    detail=f"venv not found or missing bin/python: {body.conda_env}",
                )
        else:
            # For conda (or unset), scan user's filesystem for conda installations
            env_names = _scan_user_conda_envs(body.username)
            if body.conda_env not in env_names:
                raise HTTPException(
                    status_code=400,
                    detail=f"Conda environment '{body.conda_env}' not found for user '{body.username}'. Available: {sorted(env_names)}",
                )

    task_id = await request.app.state.scheduler.submit_task(
        username=body.username,
        command=body.command,
        work_dir=body.work_dir,
        priority=body.priority,
        conda_env=body.conda_env,
        env_type=body.env_type,
        preferred_gpu_id=body.preferred_gpu_id,
    )
    return SubmitResponse(task_id=task_id, state="PENDING")


@router.get("/tasks/status", response_model=TaskListResponse)
async def get_task_status(
    request: Request,
    state: str | None = None,
    username: str | None = None,
    limit: int | None = Query(default=None, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    q: str | None = Query(default=None, max_length=200),
):
    tasks = await request.app.state.scheduler.get_tasks(
        state=state, username=username, limit=limit, offset=offset, query=q
    )
    for t in tasks:
        if t.state == "RUNNING":
            fraction, eta_seconds = request.app.state.scheduler.get_progress(t.id)
            t.progress = fraction
            t.eta = eta_seconds
    total = await count_tasks(request.app.state.db, state, username, q)
    return TaskListResponse(tasks=tasks, total=total, limit=limit, offset=offset)


@router.post("/tasks/{task_id}/abort", response_model=AbortResponse)
async def abort_task(
    task_id: str,
    request: Request,
    username: str | None = None,
    admin_password: str | None = None,
    x_admin_password: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
):
    row = await get_task_by_id(request.app.state.db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if row["state"] not in ("PENDING", "STARTING", "RUNNING"):
        raise HTTPException(status_code=400, detail=f"Cannot abort task in state {row['state']}")
    if username and row["username"] != username:
        raise HTTPException(status_code=403, detail=f"Task belongs to user '{row['username']}', not '{username}'")
    session_token = _require_admin(
        request, username, admin_password, x_admin_password, x_admin_token
    )
    try:
        success = await request.app.state.scheduler.abort_task(task_id)
    except SupervisorUnavailable as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    if success:
        _touch_admin_session(request, session_token)
    return AbortResponse(success=success, task_id=task_id)


@router.delete("/tasks")
async def delete_all_tasks_endpoint(
    request: Request,
    username: str | None = None,
    admin_password: str | None = None,
    x_admin_password: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
):
    session_token = _require_admin(
        request, username, admin_password, x_admin_password, x_admin_token
    )
    count = await delete_all_tasks(request.app.state.db, username)
    _touch_admin_session(request, session_token)
    return {"deleted": count}


@router.delete("/tasks/{task_id}")
async def delete_task_endpoint(
    task_id: str,
    request: Request,
    username: str | None = None,
    admin_password: str | None = None,
    x_admin_password: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
):
    row = await get_task_by_id(request.app.state.db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    if row["state"] in ("RUNNING", "STARTING", "PENDING"):
        raise HTTPException(status_code=400, detail=f"Cannot delete task in state {row['state']}")
    if username and row["username"] != username:
        raise HTTPException(status_code=403, detail=f"Task belongs to user '{row['username']}', not '{username}'")
    session_token = _require_admin(
        request, username, admin_password, x_admin_password, x_admin_token
    )
    deleted = await delete_task(request.app.state.db, task_id)
    if not deleted:
        raise HTTPException(status_code=500, detail="Failed to delete task")
    _touch_admin_session(request, session_token)
    return {"success": True, "task_id": task_id}


@router.get("/gpus", response_model=GpuListResponse)
async def get_gpus(request: Request):
    statuses = await asyncio.to_thread(request.app.state.gpu_manager.get_all_gpu_status)
    return GpuListResponse(gpus=[s.model_dump() for s in statuses])


@router.post("/gpus/{gpu_id}/fan")
async def set_gpu_fan(
    gpu_id: int,
    body: FanConfig,
    request: Request,
    admin_password: str | None = None,
    x_admin_password: str | None = Header(default=None),
    x_admin_token: str | None = Header(default=None),
):
    session_token = _require_admin(
        request, None, admin_password, x_admin_password, x_admin_token
    )
    gpu_manager = request.app.state.gpu_manager
    if gpu_id not in gpu_manager.managed_gpu_ids:
        raise HTTPException(status_code=400, detail=f"Invalid GPU ID {gpu_id}")
    try:
        if body.mode == "auto":
            result = await asyncio.to_thread(gpu_manager.set_fan_auto, gpu_id)
        elif body.mode == "manual":
            if body.speed is None:
                raise HTTPException(status_code=400, detail="speed is required when mode is 'manual'")
            result = await asyncio.to_thread(gpu_manager.set_fan_speed, gpu_id, body.speed)
        else:
            raise HTTPException(status_code=400, detail=f"Invalid mode: {body.mode}. Must be 'auto' or 'manual'.")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    _touch_admin_session(request, session_token)
    return result


@router.get("/users", response_model=UsersListResponse)
async def list_users():
    entries = [entry for entry in pwd.getpwall() if entry.pw_uid >= 1000 and os.path.isdir(entry.pw_dir)]
    users = sorted(entry.pw_name for entry in entries)
    homes = {entry.pw_name: entry.pw_dir for entry in entries}
    return UsersListResponse(users=users, home_directories=homes)


@router.get("/conda/envs/{username}", response_model=CondaEnvListResponse)
async def list_conda_envs(username: str):
    user_home = get_user_home(username)
    if user_home is None:
        raise HTTPException(status_code=404, detail=f"User home directory not found: {username}")
    envs = _scan_user_conda_envs(username)
    return CondaEnvListResponse(environments=sorted(envs))


@router.get("/envs/scan-venvs", response_model=VenvScanResponse)
async def scan_venvs(path: str, username: str):
    """Scan a directory for Python venvs (directories containing pyvenv.cfg)."""
    user_home = get_user_home(username)
    if user_home is None:
        raise HTTPException(status_code=404, detail=f"System user not found: {username}")
    real_path = os.path.realpath(path)
    if not is_within(real_path, user_home):
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


@router.get("/health", response_model=HealthResponse)
async def health(request: Request):
    running, starting, pending = await asyncio.gather(
        count_tasks(request.app.state.db, state="RUNNING"),
        count_tasks(request.app.state.db, state="STARTING"),
        count_tasks(request.app.state.db, state="PENDING"),
    )
    supervisor_available = request.app.state.scheduler.supervisor_available
    return HealthResponse(
        status="ok" if supervisor_available else "degraded",
        gpus_managed=len(request.app.state.gpu_manager.managed_gpu_ids),
        tasks_running=running,
        tasks_pending=pending,
        tasks_starting=starting,
        supervisor_available=supervisor_available,
        admin_enabled=bool(request.app.state.admin_password),
    )


@router.get("/workdirs/{username}", response_model=WorkdirListResponse)
async def list_workdirs(username: str):
    root = get_user_home(username)
    if root is None:
        raise HTTPException(status_code=404, detail=f"User home directory not found: {username}")
    try:
        dirs = []
        for entry in sorted(os.listdir(root)):
            full = os.path.join(root, entry)
            if os.path.isdir(full) and not entry.startswith("."):
                dirs.append(full)
    except PermissionError:
        raise HTTPException(status_code=403, detail=f"Permission denied: cannot read {root}")
    return WorkdirListResponse(directories=dirs)


@router.get("/workdirs/{username}/browse", response_model=WorkdirListResponse)
async def browse_workdirs(username: str, path: str = ""):
    user_home = get_user_home(username)
    if user_home is None:
        raise HTTPException(status_code=404, detail=f"System user not found: {username}")
    if path:
        root = os.path.join(user_home, path)
    else:
        root = user_home
    real_root = os.path.realpath(root)
    if not is_within(real_root, user_home):
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


@router.get("/tasks/{task_id}/log")
async def get_task_log(
    task_id: str,
    request: Request,
    tail_bytes: int | None = Query(default=None, ge=1, le=262_144),
    offset: int | None = Query(default=None, ge=0),
    limit_bytes: int = Query(default=262_144, ge=1, le=262_144),
):
    if tail_bytes is not None and offset is not None:
        raise HTTPException(status_code=400, detail="tail_bytes and offset are mutually exclusive")
    row = await get_task_by_id(request.app.state.db, task_id)
    if row is None:
        raise HTTPException(status_code=404, detail=f"Task not found: {task_id}")
    log_path = row["log_path"]
    if log_path is None or not os.path.isfile(log_path):
        raise HTTPException(status_code=404, detail="Log file not found")
    if tail_bytes is None and offset is None:
        with open(log_path, "r", errors="replace") as file_handle:
            return PlainTextResponse(file_handle.read())

    size = os.path.getsize(log_path)
    start = max(0, size - tail_bytes) if tail_bytes is not None else min(offset or 0, size)
    read_limit = tail_bytes if tail_bytes is not None else limit_bytes
    with open(log_path, "rb") as file_handle:
        file_handle.seek(start)
        raw = file_handle.read(read_limit)
    next_offset = start + len(raw)
    headers = {
        "X-Log-Next-Offset": str(next_offset),
        "X-Log-Size": str(size),
        "X-Log-Truncated": "true" if start > 0 or next_offset < size else "false",
        "Access-Control-Expose-Headers": "X-Log-Next-Offset, X-Log-Size, X-Log-Truncated",
    }
    return PlainTextResponse(raw.decode("utf-8", errors="replace"), headers=headers)


def create_app(
    db_path: str = DB_PATH,
    gpu_factory: Callable[[], GpuManager] = GpuManager,
    scheduler_factory: Callable[[aiosqlite.Connection, GpuManager], Scheduler] = Scheduler,
    admin_session_store_factory: Callable[[int], AdminSessionStore] = AdminSessionStore,
) -> FastAPI:
    application = FastAPI(title="rl-scheduler", lifespan=lifespan)
    application.state.db_path = db_path
    application.state.gpu_factory = gpu_factory
    application.state.scheduler_factory = scheduler_factory
    application.state.admin_session_store_factory = admin_session_store_factory
    application.include_router(router)
    # Static files mount must follow API routes to avoid path conflicts.
    application.mount("/static", StaticFiles(directory="static"), name="static")
    return application


app = create_app()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
