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

# Install development dependencies and run tests
pip install -r requirements-dev.txt
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q
```

The pytest suite covers database transitions/migrations, scheduler helpers, and API compatibility. There is no linter configured.

## Configuration

Environment variables:
- `ADMIN_PASSWORD` — password for admin-mode task operations and GPU fan control. `start.sh` defaults it to `admin`; deployments must override that value. If the application receives no password, admin operations are disabled.
- `ADMIN_SESSION_TIMEOUT_SECONDS` — sliding admin-session inactivity timeout, from 1 to 86400 seconds (default: 300). Invalid values fall back to the default.

## Architecture

A FastAPI control plane for systemd-managed GPU workloads. A background asyncio loop polls every 3s to reconcile persistent transient units and dispatch queued tasks.

**4 modules, layered bottom-up:**

- `hardware.py` — `GpuManager` wraps pynvml (NVML). GPU availability requires scheduling to be enabled, no scheduler-managed task, and no non-MPS compute processes from other users. It collects telemetry and fan status, and sorts GPUs by PCI Bus-ID.
- `models.py` — SQLite schema and guarded task transitions. `runner_unit` links active records to systemd; GPU scheduling settings persist by PCI Bus-ID; migrations are additive.
- `scheduler.py` — Reconciles database rows with retained systemd status, restores GPU tracking after restart, excludes disabled GPUs from dispatch, and leaves active units running on shutdown.
- `systemd_runner.py` — Argument-safe `systemd-run`/`systemctl` adapter. Units retain exit status, append to task logs, and use control-group termination for abort.
- `environments.py` — Central user-home, conda/venv discovery, containment, and Python resolution used by both validation and dispatch.
- `main.py` — FastAPI app factory with lifespan-managed state and injectable GPU/scheduler factories. REST endpoints provide compatible task pagination/search and bounded byte-offset log reads. Blocking GPU operations are moved to worker threads; privileged operations accept process-local admin-session tokens or legacy password credentials.

**Frontend:** The zero-build vanilla JS/Tailwind dashboard uses one non-overlapping 3-second refresh cycle, bounded server-side task pages/search, user scope, and incremental live-log reads. Polling pauses in hidden tabs.

**Admin sessions:** When `ADMIN_PASSWORD` is set, admin-mode abort/delete, cross-user actions, and all fan controls require authorization. The dashboard exchanges the password for a random token, retains it only in page memory, and renews its inactivity timeout after successful privileged operations. Lock, page refresh, service restart, or expiry requires authentication again. `X-Admin-Password` and the legacy query parameter remain accepted for API compatibility.

**GPU scheduling controls:** Selecting a GPU card opens a control modal with a scheduling toggle above the fan controls. Each scheduling change requires the raw administrator password; session tokens are deliberately not accepted. Disabled GPUs remain visible but cannot receive automatic or explicitly pinned new tasks. The setting persists by PCI Bus-ID; disabling does not terminate an active task, and enabling wakes pending dispatch immediately.

**Task lifecycle:** `PENDING → STARTING → RUNNING → COMPLETED` (exit 0) or `FAILED`. Closing the app preserves units; explicit abort sends SIGKILL to the complete systemd control group. SQL guards and a lifecycle lock prevent dispatch/abort races.

**Database:** SQLite via aiosqlite, file `scheduler.db` (in `.gitignore`). Logs go to `logs/` (also gitignored). `.vscode/` and `notes/` are also gitignored.
