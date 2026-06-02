# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server (requires sudo for per-user subprocess execution)
sudo -E python3 main.py
# or: ./start.sh

# Server starts on http://0.0.0.0:8000, dashboard at http://localhost:8000
```

There is no test suite or linter configured.

## Architecture

A single-process FastAPI server that manages GPU workloads across multiple users on a shared multi-GPU machine. A background asyncio loop (the `Scheduler`) polls every 3s to dispatch queued tasks to available GPUs.

**4 modules, layered bottom-up:**

- `hardware.py` — `GpuManager` wraps pynvml (NVML). GPU availability = no scheduler-managed task on it AND no non-MPS compute processes from other users. Collects temperature, VRAM usage (sum of per-process memory, matching nvidia-smi), active task ID.
- `models.py` — SQLite schema (`tasks` table with `env_type` column for conda/venv), Pydantic models (`TaskSubmit`, `TaskStatus`, `GpuStatus`), DB helper functions. DB write helpers (`set_task_running`, `set_task_finished`, `set_task_aborted`) all return `bool` and include state-transition guards (e.g. `WHERE state = 'PENDING'`) to prevent races between dispatch and abort. On `init_db`, any tasks stuck in `RUNNING` state from a prior crash are moved to `FAILED` (exit_code=-1). Schema migrations add missing columns (e.g. `env_type`) on startup.
- `scheduler.py` — `Scheduler` is the async orchestrator. `_run_loop` calls `_dispatch_pending_tasks` (sorted by priority DESC, created_at ASC) then `_check_running_tasks`. Tasks spawn via `asyncio.create_subprocess_shell` with `preexec_fn=os.setsid` (process group for clean kill). Commands are wrapped with `sudo -u <user> bash -l -c` for per-user execution, and `CUDA_VISIBLE_DEVICES=<gpu_id>` for GPU isolation. Python envs (conda or venv) are resolved via `_find_python` — conda locates the env's python in `/home/<user>/<conda_dir>/envs/<name>/bin/python`, venv uses the stored full path under the user's home — and `python`/`python3` in the command string is replaced via regex. `_spawn_task` checks `set_task_running` return value — if an abort sneaked in mid-dispatch, it kills the spawned process and cleans up.
- `main.py` — FastAPI app with lifespan-managed globals (`db`, `gpu_manager`, `scheduler`). REST endpoints: task submit/status/abort/log, GPU telemetry, user listing (parsed from `/etc/passwd`), per-user conda env scanning, venv scanning (by directory, with path traversal guard), per-user workdir browsing with path traversal guard. Static files (`static/index.html`) mounted after all API routes.

**Frontend:** Single-page dashboard in `static/index.html` — vanilla JS + Tailwind CSS CDN. Polls `/health`, `/gpus`, `/tasks/status` every 3s. Includes task submission form (with user selector, GPU selector, Python env picker combining conda and venv options, workdir browser modal), GPU cards, filterable task table, log viewer modal (works for both running and completed tasks), and directory browser modal.

**Task lifecycle:** `PENDING → RUNNING → COMPLETED` (exit 0) or `FAILED` (non-zero exit or abort). Abort sends SIGKILL to the entire process group. State transitions use SQL guards to prevent concurrent abort/dispatch races.

**Database:** SQLite via aiosqlite, file `scheduler.db` (in `.gitignore`). Logs go to `logs/` (also gitignored). `.vscode/` and `notes/` are also gitignored.
