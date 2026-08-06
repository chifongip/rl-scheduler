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
- `ADMIN_PASSWORD` — password for admin-mode task operations and GPU fan control. The dashboard sends it in `X-Admin-Password`; the legacy query parameter remains accepted. If unset, admin operations are disabled.

## Architecture

A FastAPI control plane for systemd-managed GPU workloads. A background asyncio loop polls every 3s to reconcile persistent transient units and dispatch queued tasks.

**4 modules, layered bottom-up:**

- `hardware.py` — `GpuManager` wraps pynvml (NVML). GPU availability = no scheduler-managed task on it AND no non-MPS compute processes from other users (`_count_external_compute_procs` helper). Collects temperature, VRAM usage (sum of per-process memory, matching nvidia-smi), active task ID, `external_process_count`, and fan status (speed, mode, num fans). GPUs are sorted by PCI Bus-ID to match nvidia-smi order. Fan control via `_v2` NVML APIs with lazy capability probing (`_probe_fan_support`). Fans are reset to automatic on shutdown.
- `models.py` — SQLite schema and guarded `PENDING → STARTING → RUNNING → COMPLETED/FAILED` transitions. `runner_unit` links active records to systemd; migrations are additive.
- `scheduler.py` — Reconciles database rows with retained systemd status, restores GPU tracking after restart, pauses safely during supervisor outages, and leaves active units running on shutdown.
- `systemd_runner.py` — Argument-safe `systemd-run`/`systemctl` adapter. Units retain exit status, append to task logs, and use control-group termination for abort.
- `environments.py` — Central user-home, conda/venv discovery, containment, and Python resolution used by both validation and dispatch.
- `main.py` — FastAPI app factory with lifespan-managed state and injectable GPU/scheduler factories. REST endpoints provide compatible task pagination/search and bounded byte-offset log reads. Blocking GPU operations are moved to worker threads; admin fan and cross-user operations accept `X-Admin-Password`.

**Frontend:** The zero-build vanilla JS/Tailwind dashboard uses one non-overlapping 3-second refresh cycle, bounded server-side task pages/search, user scope, and incremental live-log reads. Polling pauses in hidden tabs.

**Admin password:** When `ADMIN_PASSWORD` is set, admin-mode abort/delete, cross-user actions, and all fan controls require it. The dashboard prompts when needed, sends the value in a header, and clears it after each successful mutation.

**Task lifecycle:** `PENDING → STARTING → RUNNING → COMPLETED` (exit 0) or `FAILED`. Closing the app preserves units; explicit abort sends SIGKILL to the complete systemd control group. SQL guards and a lifecycle lock prevent dispatch/abort races.

**Database:** SQLite via aiosqlite, file `scheduler.db` (in `.gitignore`). Logs go to `logs/` (also gitignored). `.vscode/` and `notes/` are also gitignored.
