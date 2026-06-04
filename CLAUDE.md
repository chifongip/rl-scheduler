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

## Configuration

Environment variables (set in `start.sh` or at runtime):
- `ADMIN_PASSWORD` — password for admin-mode operations (abort/delete when no user is selected). If not set, admin mode is disabled (returns 403).

## Architecture

A single-process FastAPI server that manages GPU workloads across multiple users on a shared multi-GPU machine. A background asyncio loop (the `Scheduler`) polls every 3s to dispatch queued tasks to available GPUs.

**4 modules, layered bottom-up:**

- `hardware.py` — `GpuManager` wraps pynvml (NVML). GPU availability = no scheduler-managed task on it AND no non-MPS compute processes from other users (`_count_external_compute_procs` helper). Collects temperature, VRAM usage (sum of per-process memory, matching nvidia-smi), active task ID, `external_process_count`, and fan status (speed, mode, num fans). GPUs are sorted by PCI Bus-ID to match nvidia-smi order. Fan control via `_v2` NVML APIs with lazy capability probing (`_probe_fan_support`). Fans are reset to automatic on shutdown.
- `models.py` — SQLite schema (`tasks` table with `env_type` column for conda/venv), Pydantic models (`TaskSubmit`, `TaskStatus` with `progress`/`eta`/`duration`, `GpuStatus` with `external_process_count` and fan fields, `FanConfig`), DB helper functions. DB write helpers (`set_task_running`, `set_task_finished`, `set_task_aborted`) all return `bool` and include state-transition guards (e.g. `WHERE state = 'PENDING'`) to prevent races between dispatch and abort. Delete helpers (`delete_task`, `delete_all_tasks`) remove both DB rows and log files. On `init_db`, any tasks stuck in `RUNNING` state from a prior crash are moved to `FAILED` (exit_code=-1). Schema migrations add missing columns (e.g. `env_type`) on startup.
- `scheduler.py` — `Scheduler` is the async orchestrator. `_run_loop` calls `_dispatch_pending_tasks` (sorted by priority DESC, created_at ASC) then `_check_running_tasks`. Tasks spawn via `asyncio.create_subprocess_shell` with `preexec_fn=os.setsid` (process group for clean kill). Commands are wrapped with `sudo -u <user> bash -l -c` for per-user execution, `CUDA_DEVICE_ORDER=PCI_BUS_ID` and `CUDA_VISIBLE_DEVICES=<gpu_id>` for GPU isolation. Python envs (conda or venv) are resolved via `_find_python` — conda locates the env's python in `/home/<user>/<conda_dir>/envs/<name>/bin/python` (checks anaconda3, miniconda3, miniforge3 and `~/.conda/environments.txt`), venv uses the stored full path under the user's home — and `python`/`python3` in the command string is replaced via regex. `_parse_log_progress` parses rsl_rl-format logs for `Learning iteration X/Y` and `ETA: HH:MM:SS` to report progress/ETA. `_spawn_task` checks `set_task_running` return value — if an abort sneaked in mid-dispatch, it kills the spawned process and cleans up.
- `main.py` — FastAPI app with lifespan-managed globals (`db`, `gpu_manager`, `scheduler`). REST endpoints: task submit/status/abort/delete/log, GPU telemetry (includes `external_process_count` and fan data), GPU fan control (`POST /gpus/{gpu_id}/fan`), user listing (parsed from `/etc/passwd`), per-user conda env scanning (also reads `~/.conda/environments.txt`), venv scanning (by directory, with path traversal guard), per-user workdir browsing with path traversal guard. Abort and delete endpoints require admin password when no user is selected (`_require_admin` helper). Static files (`static/index.html`) mounted after all API routes.

**Frontend:** Single-page dashboard in `static/index.html` — vanilla JS + Tailwind CSS CDN. Polls `/health`, `/gpus`, `/tasks/status` every 3s (pauses when tab is hidden). Selected user persisted in `localStorage`. Includes task submission form (with user selector, GPU selector, Python env picker combining conda and venv options, workdir browser modal), GPU cards (show "Running"/"Busy"/"Idle" badges, fan speed/mode, clickable for fan control modal), filterable task table (state filter tabs, DOM-diff updates to preserve expanded command state), log viewer modal (works for both running and completed tasks), progress bars with ETA for running tasks, delete buttons (individual and bulk, with admin password prompt in admin mode), and directory browser modal.

**Admin password:** When `ADMIN_PASSWORD` is set, abort/delete operations without a selected user require the password. Cross-user operations (User A managing User B's tasks) also require admin credentials. The password is prompted via `prompt()` and not cached between operations.

**Task lifecycle:** `PENDING → RUNNING → COMPLETED` (exit 0) or `FAILED` (non-zero exit or abort). Abort sends SIGKILL to the entire process group. State transitions use SQL guards to prevent concurrent abort/dispatch races.

**Database:** SQLite via aiosqlite, file `scheduler.db` (in `.gitignore`). Logs go to `logs/` (also gitignored). `.vscode/` and `notes/` are also gitignored.
