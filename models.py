from __future__ import annotations

import os
import time
import uuid

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
    finished_at       REAL
);
"""

RECOVER_STUCK_SQL = """
UPDATE tasks SET state = 'FAILED', exit_code = -1
WHERE state = 'RUNNING';
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
UPDATE tasks SET state = 'RUNNING', gpu_id = ?, pid = ?, started_at = ?, log_path = ?
WHERE id = ? AND state = 'PENDING';
"""

UPDATE_FINISHED_SQL = """
UPDATE tasks SET state = ?, exit_code = ?, finished_at = ?
WHERE id = ? AND state = 'RUNNING';
"""

SET_ABORTED_SQL = """
UPDATE tasks SET state = 'FAILED', exit_code = -9, finished_at = ?
WHERE id = ? AND state IN ('PENDING', 'RUNNING');
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


class GpuStatus(BaseModel):
    gpu_id: int
    name: str
    temperature_c: int
    memory_used_mb: int
    memory_total_mb: int
    memory_utilization_pct: float
    active_task_id: str | None
    external_process_count: int = 0


# --- DB helpers ---

async def init_db(db_path: str = DB_PATH) -> aiosqlite.Connection:
    db = await aiosqlite.connect(db_path)
    await db.execute(CREATE_TABLE_SQL)
    # Migrate: add env_type column if missing (for existing databases)
    cursor = await db.execute("PRAGMA table_info(tasks)")
    columns = {row[1] for row in await cursor.fetchall()}
    if "env_type" not in columns:
        await db.execute("ALTER TABLE tasks ADD COLUMN env_type TEXT")
    # Recover tasks that were RUNNING when the scheduler last crashed
    cursor = await db.execute(RECOVER_STUCK_SQL)
    if cursor.rowcount > 0:
        import logging
        logging.getLogger("models").warning(
            "Recovered %d stuck RUNNING tasks → FAILED on startup", cursor.rowcount
        )
    await db.commit()
    return db


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
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


async def set_task_running(db: aiosqlite.Connection, task_id: str, gpu_id: int, pid: int, log_path: str) -> bool:
    cursor = await db.execute(UPDATE_TO_RUNNING_SQL, (gpu_id, pid, time.time(), log_path, task_id))
    await db.commit()
    return cursor.rowcount > 0


async def set_task_finished(db: aiosqlite.Connection, task_id: str, exit_code: int) -> bool:
    state = "COMPLETED" if exit_code == 0 else "FAILED"
    cursor = await db.execute(UPDATE_FINISHED_SQL, (state, exit_code, time.time(), task_id))
    await db.commit()
    return cursor.rowcount > 0


async def set_task_aborted(db: aiosqlite.Connection, task_id: str) -> bool:
    cursor = await db.execute(SET_ABORTED_SQL, (time.time(), task_id))
    await db.commit()
    return cursor.rowcount > 0


async def delete_task(db: aiosqlite.Connection, task_id: str) -> bool:
    row = await get_task_by_id(db, task_id)
    if row is None:
        return False
    cursor = await db.execute(
        "DELETE FROM tasks WHERE id = ? AND state NOT IN ('RUNNING', 'PENDING')",
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
    for (log_path,) in rows:
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
    cols = [d[0] for d in cursor.description]
    return dict(zip(cols, row))


async def get_tasks(db: aiosqlite.Connection, state: str | None = None, username: str | None = None) -> list[dict]:
    if state and username:
        cursor = await db.execute(SELECT_BY_STATE_AND_USERNAME_SQL, (state, username))
    elif state:
        cursor = await db.execute(SELECT_BY_STATE_SQL, (state,))
    elif username:
        cursor = await db.execute(SELECT_BY_USERNAME_SQL, (username,))
    else:
        cursor = await db.execute(SELECT_ALL_SQL)
    rows = await cursor.fetchall()
    cols = [d[0] for d in cursor.description]
    return [dict(zip(cols, row)) for row in rows]


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
    )
