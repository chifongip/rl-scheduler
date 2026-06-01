# rl-scheduler

A GPU workload manager that prevents resource contention on multi-GPU machines. It queues ML/RL training jobs and dispatches them to isolated GPUs, ensuring each physical GPU runs exactly one training script at a time.

## Features

- **GPU isolation** via `CUDA_VISIBLE_DEVICES` — each job sees only its assigned GPU
- **Priority queue** — higher-priority tasks dispatch first
- **Multi-user aware** — per-user conda environments and working directories
- **Conda environment support** — specify a conda env per task, python binary resolved automatically
- **Manual GPU selection** — pin a task to a specific GPU or let the scheduler auto-assign
- **Persistent queue** — SQLite-backed task state survives server restarts
- **Crash recovery** — orphaned tasks from a previous crash are marked `FAILED` on startup
- **Live dashboard** — web UI with GPU monitoring, task submission, log viewing, and abort
- **Process group management** — abort kills the entire process tree, not just the parent
- **Live log viewing** — view stdout/stderr while a task is still running

## Screenshots

<p align="center">
  <img src="doc/dashboard.png" width="100%">
</p>

## Project Structure

```
rl-scheduler/
├── main.py           # FastAPI app, REST endpoints, static file serving
├── models.py         # Pydantic models, SQLite schema, DB helpers
├── scheduler.py      # Async orchestrator loop, task dispatch, subprocess management
├── hardware.py       # pynvml GPU telemetry and availability checks
├── requirements.txt  # Python dependencies
├── start.sh          # Convenience launcher (sudo python3 main.py)
├── static/
│   └── index.html    # Single-page dashboard (Tailwind CSS + vanilla JS)
├── logs/             # Auto-created; per-task stdout/stderr logs
└── scheduler.db      # SQLite database (created at runtime)
```

## Requirements

- Python 3.10+
- NVIDIA GPU with drivers installed
- `nvidia-smi` working (for pynvml)
- conda (optional, only needed if using `conda_env`)

## Installation

```bash
pip install -r requirements.txt
```

## Usage

Start the server:

```bash
sudo -E python3 main.py
# or: ./start.sh
```

The server runs on `http://0.0.0.0:8000`. Opening this URL in a browser loads the dashboard.

### Quick test via curl

```bash
# Submit a task
curl -X POST http://localhost:8000/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{"username": "your-user", "command": "echo hello && sleep 2", "work_dir": "/tmp"}'

# Check task status
curl http://localhost:8000/tasks/status

# View GPU status
curl http://localhost:8000/gpus
```

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks/submit` | Submit a new task |
| `GET`  | `/tasks/status` | List tasks (optional `?state=` and `?username=` filters) |
| `POST` | `/tasks/{task_id}/abort` | Force-kill a running or pending task |
| `GET`  | `/tasks/{task_id}/log` | Get task log output (works while running) |
| `GET`  | `/gpus` | Live GPU telemetry (temp, VRAM, active task) |
| `GET`  | `/users` | List system users (uid ≥ 1000 with valid home dir) |
| `GET`  | `/conda/envs/{username}` | List conda environments available to a user |
| `GET`  | `/workdirs/{username}` | List top-level directories in user's home |
| `GET`  | `/workdirs/{username}/browse` | Browse subdirectories in user's home (`?path=`) |
| `GET`  | `/health` | System health summary |

### Submit a task

```bash
curl -X POST http://localhost:8000/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{
    "username": "your-user",
    "command": "python train.py --epochs 100",
    "work_dir": "/home/your-user/project",
    "conda_env": "my-env",
    "preferred_gpu_id": 0,
    "priority": 10
  }'
```

Fields:
- `username` **(required)** — the system user to run the command as
- `command` **(required)** — the shell command to execute
- `work_dir` (default: `.`) — working directory for the subprocess
- `conda_env` (default: null) — conda environment name; `python`/`python3` in the command is replaced with the env's python binary
- `preferred_gpu_id` (default: null) — pin to a specific GPU, or null for auto-assign
- `priority` (default: 0) — higher values dispatch first

### List tasks

```bash
# All tasks
curl http://localhost:8000/tasks/status

# Filter by state
curl "http://localhost:8000/tasks/status?state=RUNNING"

# Filter by user
curl "http://localhost:8000/tasks/status?username=alice"

# Combined
curl "http://localhost:8000/tasks/status?state=RUNNING&username=alice"
```

### Abort a task

```bash
# Abort by task ID (optionally scoped to a user)
curl -X POST "http://localhost:8000/tasks/{task_id}/abort?username=your-user"
```

### View task logs

```bash
curl http://localhost:8000/tasks/{task_id}/log
```

Works while the task is running — output is streamed to the log file in real time.

## Task Lifecycle

```
PENDING  ──dispatch──>  RUNNING  ──exit 0──>  COMPLETED
    │                       │
    └──abort──────────────> FAILED
```

- **PENDING** — queued, waiting for a free GPU
- **RUNNING** — subprocess is active on an assigned GPU
- **COMPLETED** — subprocess exited with code 0
- **FAILED** — subprocess exited with a non-zero code, or was aborted

## GPU Isolation

Each task gets `CUDA_VISIBLE_DEVICES=<gpu_id>` injected into its environment. This makes the assigned GPU appear as device 0 to the training script, fully isolating it from other GPUs and other tasks.

A GPU is considered "available" when:
1. No scheduler-managed task is currently running on it
2. No non-MPS compute processes are running on it (detected via NVML)

## Conda Support

If `conda_env` is specified, the scheduler locates the target environment's python binary at `/home/<user>/<conda_dir>/envs/<env>/bin/python` (checking miniconda3, anaconda3, and miniforge3). It then replaces `python` or `python3` in the command with the full path to the env's python. The command runs under `bash -l` so conda-initialized shell environments are picked up.

## Crash Recovery

If the scheduler crashes while tasks are running, those tasks cannot be reattached (the subprocess handles are lost). On the next startup, any tasks left in `RUNNING` state are automatically transitioned to `FAILED` with `exit_code = -1`.
