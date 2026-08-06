# Repository Guidelines

## Project Structure & Module Organization

The service uses top-level Python modules: `main.py` defines FastAPI routes, `scheduler.py` reconciles queued jobs, `systemd_runner.py` owns transient-service execution, `models.py` provides SQLite/Pydantic state, `environments.py` resolves Python environments, and `hardware.py` wraps NVML. The dashboard is `static/index.html`; `doc/` holds documentation assets. Do not commit generated `logs/`, `scheduler.db`, or caches.

## Build, Test, and Development Commands

- `pip install -r requirements.txt` installs runtime dependencies; `pip install -r requirements-dev.txt` adds test tools.
- `sudo -E python3 main.py` starts the API and dashboard on port 8000 with the privileges needed for per-user jobs and fan control.
- `./start.sh` is the convenience launcher; review its environment settings before use.
- `curl http://localhost:8000/health` performs a basic health check.

There is no separate build step. NVIDIA drivers and working `nvidia-smi`/NVML support are required for normal operation.

## Coding Style & Naming Conventions

Follow the existing Python style: four-space indentation, type hints, `snake_case` functions and variables, `PascalCase` classes, and `UPPER_SNAKE_CASE` module constants. Keep database and server operations asynchronous. Prefer small helpers and explicit state-transition guards when modifying concurrent scheduler behavior. No formatter or linter is currently configured; keep changes consistent with surrounding code.

## Testing Guidelines

Tests use pytest under `tests/`; run `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest -q`. Add focused `test_*.py` coverage for affected API endpoints, scheduler transitions, and failure paths. For dashboard changes, also exercise the workflow in a browser and include before/after screenshots when visual behavior changes. No coverage threshold is currently enforced.

## Commit & Pull Request Guidelines

Use short, imperative commit subjects consistent with history, such as `fix: prevent duplicate dispatch` or `add task retry support`. Pull requests should explain the behavior change, note validation performed, link relevant issues, and call out database or configuration impacts.

## Security & Configuration Tips

Provide `ADMIN_PASSWORD` through the environment; never commit real credentials. Treat root execution, shell commands, user-supplied paths, process-group termination, and GPU fan controls as security-sensitive. Preserve path-containment checks and authorization rules when changing these areas.
