from __future__ import annotations

import os
import time
import uuid
from typing import Literal

import aiosqlite
from pydantic import BaseModel, Field

DB_PATH = "scheduler.db"

CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS tasks (
    id                TEXT PRIMARY KEY,
    username          TEXT NOT NULL DEFAULT '',
    command           TEXT NOT NULL,
    work_dir          TEXT NOT NULL,
    conda_env         TEXT,
    env_type          TEXT,
    preferred_gpu_id  INTEGER,
    priority          INTEGER NOT NULL DEFAULT 0,
    state             TEXT NOT NULL DEFAULT 'PENDING',
    gpu_id            INTEGER,
    pid               INTEGER,
    exit_code         INTEGER,
    log_path          TEXT,
    created_at        REAL NOT NULL,
    started_at        REAL,
    finished_at       REAL,
    status_reason     TEXT,
    runner_unit       TEXT
);
"""

CREATE_GPU_SETTINGS_SQL = """
CREATE TABLE IF NOT EXISTS gpu_settings (
    bus_id              TEXT PRIMARY KEY,
    scheduling_enabled  INTEGER NOT NULL DEFAULT 1 CHECK (scheduling_enabled IN (0, 1)),
    updated_at          REAL NOT NULL
);
"""

INSERT_SQL = """
INSERT INTO tasks (id, username, command, work_dir, conda_env, env_type, preferred_gpu_id, priority, state, created_at)
VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'PENDING', ?);
"""

SELECT_ALL_SQL = "SELECT * FROM tasks ORDER BY created_at DESC;"

SELECT_BY_STATE_SQL = "SELECT * FROM tasks WHERE state = ? ORDER BY created_at DESC;"

SELECT_BY_USERNAME_SQL = "SELECT * FROM tasks WHERE username = ? ORDER BY created_at DESC;"

SELECT_BY_STATE_AND_USERNAME_SQL = "SELECT * FROM tasks WHERE state = ? AND username = ? ORDER BY created_at DESC;"

SELECT_BY_ID_SQL = "SELECT * FROM tasks WHERE id = ?;"

SELECT_PENDING_SQL = """
SELECT * FROM tasks WHERE state = 'PENDING'
ORDER BY priority DESC, created_at ASC;
"""

UPDATE_TO_RUNNING_SQL = """
UPDATE tasks SET state = 'RUNNING', pid = ?, status_reason = NULL
WHERE id = ? AND state = 'STARTING';
"""

CLAIM_STARTING_SQL = """
UPDATE tasks SET state = 'STARTING', gpu_id = ?, started_at = ?, log_path = ?,
runner_unit = ?, status_reason = NULL
WHERE id = ? AND state = 'PENDING';
"""

RESET_STARTING_SQL = """
UPDATE tasks SET state = 'PENDING', gpu_id = NULL, pid = NULL, started_at = NULL,
log_path = NULL, runner_unit = NULL, status_reason = ?
WHERE id = ? AND state = 'STARTING';
"""

UPDATE_FINISHED_SQL = """
UPDATE tasks SET state = ?, exit_code = ?, finished_at = ?, status_reason = ?
WHERE id = ? AND state IN ('STARTING', 'RUNNING');
"""

SET_ABORTED_SQL = """
UPDATE tasks SET state = 'FAILED', exit_code = -9, finished_at = ?, status_reason = ?
WHERE id = ? AND state IN ('PENDING', 'STARTING', 'RUNNING');
"""

FAIL_PENDING_SQL = """
UPDATE tasks SET state = 'FAILED', exit_code = -1, finished_at = ?, status_reason = ?,
log_path = COALESCE(?, log_path)
WHERE id = ? AND state = 'PENDING';
"""


# --- Pydantic models ---

class TaskSubmit(BaseModel):
    username: str = Field(..., min_length=1)
    command: str = Field(..., min_length=1)
    work_dir: str = "."
    conda_env: str | None = None
    env_type: str | None = None
    preferred_gpu_id: int | None = None
    priority: int = 0


class TaskStatus(BaseModel):
    id: str
    username: str
    command: str
    work_dir: str
    conda_env: str | None
    env_type: str | None
    preferred_gpu_id: int | None
    priority: int
    state: str
    gpu_id: int | None
    pid: int | None
    exit_code: int | None
    log_path: str | None
    created_at: float
    started_at: float | None
    finished_at: float | None
    duration: float | None
    progress: float | None = None
    eta: float | None = None
    status_reason: str | None = None
    runner_unit: str | None = None


class GpuStatus(BaseModel):
    gpu_id: int
    name: str
    temperature_c: int
    memory_used_mb: int
    memory_total_mb: int
    memory_utilization_pct: float
    active_task_id: str | None
    external_process_count: int = 0
    scheduling_enabled: bool = True
    fan_speed_pct: int | None = None
    fan_mode: str | None = None
    num_fans: int | None = None


class FanConfig(BaseModel):
    mode: Literal["auto", "manual"] = "auto"
    speed: int | None = Field(default=None, ge=0, le=100)


# --- DB helpers ---

async def init_db(db_path: str = DB_PATH) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    db.row_factory = aiosqlite.Row
    await db.execute(CREATE_TABLE_SQL)
    await db.execute(CREATE_GPU_SETTINGS_SQL)
    cursor = await db.execute("PRAGMA table_info(tasks)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "env_type" not in columns:
        await db.execute("ALTER TABLE tasks ADD COLUMN env_type TEXT")
    if "status_reason" not in columns:
        await db.execute("ALTER TABLE tasks ADD COLUMN status_reason TEXT")
    if "runner_unit" not in columns:
        await db.execute("ALTER TABLE tasks ADD COLUMN runner_unit TEXT")
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_pending ON tasks(state, priority DESC, created_at ASC)"
    )
    await db.execute(
        "CREATE INDEX IF NOT EXISTS idx_tasks_user_state ON tasks(username, state, created_at DESC)"
    )
    await db.commit()
    return db


async def get_disabled_gpu_bus_ids(db: aiosqlite.Connection) -> set[str]:
    cursor = await db.execute(
        "SELECT bus_id FROM gpu_settings WHERE scheduling_enabled = 0"
    )
    return {row[0] for row in await cursor.fetchall()}


async def set_gpu_scheduling_enabled(
    db: aiosqlite.Connection, bus_id: str, enabled: bool
) -> None:
    await db.execute(
        """INSERT INTO gpu_settings (bus_id, scheduling_enabled, updated_at)
        VALUES (?, ?, ?)
        ON CONFLICT(bus_id) DO UPDATE SET
            scheduling_enabled = excluded.scheduling_enabled,
            updated_at = excluded.updated_at""",
        (bus_id, int(enabled), time.time()),
    )
    await db.commit()


async def insert_task(db: aiosqlite.Connection, submit: TaskSubmit) -> str:
    task_id = uuid.uuid4().hex[:12]
    await db.execute(
        INSERT_SQL,
        (task_id, submit.username, submit.command, submit.work_dir, submit.conda_env, submit.env_type, submit.preferred_gpu_id, submit.priority, time.time()),
    )
    await db.commit()
    return task_id


async def get_pending_tasks(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute(SELECT_PENDING_SQL)
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def get_managed_tasks(db: aiosqlite.Connection) -> list[dict]:
    cursor = await db.execute(
        "SELECT * FROM tasks WHERE state IN ('STARTING', 'RUNNING') ORDER BY started_at ASC"
    )
    return [dict(row) for row in await cursor.fetchall()]


async def claim_task_starting(
    db: aiosqlite.Connection,
    task_id: str,
    gpu_id: int,
    log_path: str,
    runner_unit: str,
) -> bool:
    cursor = await db.execute(
        CLAIM_STARTING_SQL, (gpu_id, time.time(), log_path, runner_unit, task_id)
    )
    await db.commit()
    return cursor.rowcount > 0


async def set_task_running(db: aiosqlite.Connection, task_id: str, pid: int) -> bool:
    cursor = await db.execute(UPDATE_TO_RUNNING_SQL, (pid, task_id))
    await db.commit()
    return cursor.rowcount > 0


async def reset_starting_task(
    db: aiosqlite.Connection, task_id: str, reason: str
) -> bool:
    cursor = await db.execute(RESET_STARTING_SQL, (reason[:500], task_id))
    await db.commit()
    return cursor.rowcount > 0


async def clear_runner_unit(db: aiosqlite.Connection, task_id: str) -> None:
    await db.execute("UPDATE tasks SET runner_unit = NULL WHERE id = ?", (task_id,))
    await db.commit()


async def set_task_finished(
    db: aiosqlite.Connection,
    task_id: str,
    exit_code: int,
    status_reason: str | None = None,
) -> bool:
    state = "COMPLETED" if exit_code == 0 else "FAILED"
    cursor = await db.execute(
        UPDATE_FINISHED_SQL, (state, exit_code, time.time(), status_reason, task_id)
    )
    await db.commit()
    return cursor.rowcount > 0


async def set_task_aborted(
    db: aiosqlite.Connection, task_id: str, reason: str = "aborted_by_user"
) -> bool:
    cursor = await db.execute(SET_ABORTED_SQL, (time.time(), reason, task_id))
    await db.commit()
    return cursor.rowcount > 0


async def fail_pending_task(
    db: aiosqlite.Connection,
    task_id: str,
    reason: str,
    log_path: str | None = None,
) -> bool:
    cursor = await db.execute(
        FAIL_PENDING_SQL, (time.time(), reason[:500], log_path, task_id)
    )
    await db.commit()
    return cursor.rowcount > 0


async def delete_task(db: aiosqlite.Connection, task_id: str) -> bool:
    row = await get_task_by_id(db, task_id)
    if row is None:
        return False
    cursor = await db.execute(
        "DELETE FROM tasks WHERE id = ? AND state NOT IN ('RUNNING', 'STARTING', 'PENDING')",
        (task_id,),
    )
    await db.commit()
    if cursor.rowcount > 0:
        log_path = row.get("log_path")
        if log_path and os.path.isfile(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass
        return True
    return False


async def delete_all_tasks(db: aiosqlite.Connection, username: str | None = None) -> int:
    if username:
        cursor = await db.execute(
            "SELECT log_path FROM tasks WHERE state IN ('COMPLETED', 'FAILED') AND username = ?",
            (username,),
        )
    else:
        cursor = await db.execute(
            "SELECT log_path FROM tasks WHERE state IN ('COMPLETED', 'FAILED')"
        )
    rows = await cursor.fetchall()
    if username:
        cursor = await db.execute(
            "DELETE FROM tasks WHERE state IN ('COMPLETED', 'FAILED') AND username = ?",
            (username,),
        )
    else:
        cursor = await db.execute("DELETE FROM tasks WHERE state IN ('COMPLETED', 'FAILED')")
    await db.commit()
    for row in rows:
        log_path = row[0]
        if log_path and os.path.isfile(log_path):
            try:
                os.remove(log_path)
            except OSError:
                pass
    return cursor.rowcount


async def get_task_by_id(db: aiosqlite.Connection, task_id: str) -> dict | None:
    cursor = await db.execute(SELECT_BY_ID_SQL, (task_id,))
    row = await cursor.fetchone()
    if row is None:
        return None
    return dict(row)


async def get_tasks(
    db: aiosqlite.Connection,
    state: str | None = None,
    username: str | None = None,
    limit: int | None = None,
    offset: int = 0,
    query: str | None = None,
) -> list[dict]:
    where, params = _task_filters(state, username, query)
    sql = f"SELECT * FROM tasks{where} ORDER BY created_at DESC"
    if limit is not None:
        sql += " LIMIT ? OFFSET ?"
        params.extend((limit, offset))
    cursor = await db.execute(sql, params)
    rows = await cursor.fetchall()
    return [dict(row) for row in rows]


async def count_tasks(
    db: aiosqlite.Connection,
    state: str | None = None,
    username: str | None = None,
    query: str | None = None,
) -> int:
    where, params = _task_filters(state, username, query)
    cursor = await db.execute(f"SELECT COUNT(*) FROM tasks{where}", params)
    row = await cursor.fetchone()
    return int(row[0])


def _task_filters(
    state: str | None, username: str | None, query: str | None
) -> tuple[str, list[object]]:
    clauses: list[str] = []
    params: list[object] = []
    if state:
        clauses.append("state = ?")
        params.append(state)
    if username:
        clauses.append("username = ?")
        params.append(username)
    if query:
        pattern = f"%{query.strip()}%"
        clauses.append("(id LIKE ? OR username LIKE ? OR command LIKE ? OR work_dir LIKE ?)")
        params.extend((pattern, pattern, pattern, pattern))
    return (" WHERE " + " AND ".join(clauses) if clauses else "", params)


def row_to_status(row: dict) -> TaskStatus:
    now = time.time()
    duration = None
    if row["started_at"]:
        end = row["finished_at"] or now
        duration = round(end - row["started_at"], 2)
    return TaskStatus(
        id=row["id"],
        username=row["username"],
        command=row["command"],
        work_dir=row["work_dir"],
        conda_env=row["conda_env"],
        env_type=row["env_type"],
        preferred_gpu_id=row["preferred_gpu_id"],
        priority=row["priority"],
        state=row["state"],
        gpu_id=row["gpu_id"],
        pid=row["pid"],
        exit_code=row["exit_code"],
        log_path=row["log_path"],
        created_at=row["created_at"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        duration=duration,
        status_reason=row.get("status_reason"),
        runner_unit=row.get("runner_unit"),
    )
