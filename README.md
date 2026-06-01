# rl-scheduler

A GPU workload manager that prevents resource contention on multi-GPU machines. It queues ML/RL training jobs and dispatches them to isolated GPUs, ensuring each physical GPU runs exactly one training script at a time.

## Features

- **GPU isolation** via `CUDA_VISIBLE_DEVICES` — each job sees only its assigned GPU
- **Priority queue** — higher-priority tasks dispatch first
- **Conda environment support** — specify a conda env per task, activated automatically
- **Manual GPU selection** — pin a task to a specific GPU or let the scheduler auto-assign
- **Persistent queue** — SQLite-backed task state survives server restarts
- **Crash recovery** — orphaned tasks from a previous crash are marked FAILED on startup
- **Live dashboard** — web UI with GPU monitoring, task submission, log viewing, and abort
- **Process group management** — abort kills the entire process tree, not just the parent

## Project Structure

```
rl-scheduler/
├── main.py           # FastAPI app, REST endpoints, static file serving
├── models.py         # Pydantic models, SQLite schema, DB helpers
├── scheduler.py      # Async orchestrator loop, task dispatch, subprocess management
├── hardware.py       # pynvml GPU telemetry and availability checks
├── requirements.txt  # Python dependencies
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
python main.py
```

The server runs on `http://0.0.0.0:8000`. Opening this URL in a browser loads the dashboard.

### Quick test via curl

```bash
# Submit a task
curl -X POST http://localhost:8000/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{"command": "echo hello && sleep 2", "work_dir": "/tmp"}'

# Check task status
curl http://localhost:8000/tasks/status

# View GPU status
curl http://localhost:8000/gpus
```

## Configuration

| Environment Variable | Default | Description |
|---------------------|---------|-------------|
| `RLS_WORKDIR_ROOT`  | `~`     | Root directory for the working directory dropdown in the UI |

## REST API

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/tasks/submit` | Submit a new task |
| `GET`  | `/tasks/status` | List tasks (optional `?state=` filter) |
| `POST` | `/tasks/{task_id}/abort` | Force-kill a running task |
| `GET`  | `/tasks/{task_id}/log` | Get task log output |
| `GET`  | `/gpus` | Live GPU telemetry (temp, VRAM, active task) |
| `GET`  | `/conda/envs` | List available conda environments |
| `GET`  | `/workdirs` | List directories under `RLS_WORKDIR_ROOT` |
| `GET`  | `/health` | System health summary |

### Submit a task

```bash
curl -X POST http://localhost:8000/tasks/submit \
  -H "Content-Type: application/json" \
  -d '{
    "command": "python train.py --epochs 100",
    "work_dir": "/home/user/project",
    "conda_env": "my-env",
    "preferred_gpu_id": 0,
    "priority": 10
  }'
```

All fields except `command` are optional:
- `work_dir` — working directory for the subprocess (default: `.`)
- `conda_env` — conda environment name to activate before running (default: null)
- `preferred_gpu_id` — pin to a specific GPU, or null for auto-assign (default: null)
- `priority` — higher values dispatch first (default: 0)

### List tasks

```bash
# All tasks
curl http://localhost:8000/tasks/status

# Filter by state
curl "http://localhost:8000/tasks/status?state=RUNNING"
```

### Abort a task

```bash
curl -X POST http://localhost:8000/tasks/{task_id}/abort
```

### View task logs

```bash
curl http://localhost:8000/tasks/{task_id}/log
```

## Task Lifecycle

```
PENDING  ──dispatch──>  RUNNING  ──exit 0──>  COMPLETED
                          │
                          └──exit non-zero──>  FAILED
                          └──abort──────────>  FAILED
```

- **PENDING** — queued, waiting for a free GPU
- **RUNNING** — subprocess is active on an assigned GPU
- **COMPLETED** — subprocess exited with code 0
- **FAILED** — subprocess exited with a non-zero code, or was aborted

## GPU Isolation

Each task gets `CUDA_VISIBLE_DEVICES=<gpu_id>` injected into its environment. This makes the assigned GPU appear as device 0 to the training script, fully isolating it from other GPUs and other tasks.

A GPU is considered "available" when:
1. No scheduler-managed task is currently running on it
2. Its VRAM usage is below 500 MB (prevents overriding manually-started jobs)

## Conda Support

If `conda_env` is specified, the command is wrapped with:

```
conda run -n <env> --no-capture-output --live-stream <command>
```

This activates the environment cleanly without shell sourcing, passes through stdout/stderr for log capture, and deactivates on exit. The environment must already exist on the system.

## Crash Recovery

If the scheduler crashes while tasks are running, those tasks cannot be reattached (the subprocess handles are lost). On the next startup, any tasks left in `RUNNING` state are automatically transitioned to `FAILED` with `exit_code = -1`.
