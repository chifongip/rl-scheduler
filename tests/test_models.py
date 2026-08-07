import sqlite3

import pytest

from models import (
    TaskSubmit,
    claim_task_starting,
    count_tasks,
    get_disabled_gpu_bus_ids,
    get_tasks,
    init_db,
    insert_task,
    set_task_aborted,
    set_task_finished,
    set_task_running,
    set_gpu_scheduling_enabled,
)


@pytest.mark.asyncio
async def test_task_transitions_search_and_pagination(tmp_path):
    db = await init_db(str(tmp_path / "scheduler.db"))
    first = await insert_task(db, TaskSubmit(username="alice", command="python train.py"))
    await insert_task(db, TaskSubmit(username="bob", command="echo done"))

    assert await count_tasks(db, query="train") == 1
    assert len(await get_tasks(db, limit=1, offset=1)) == 1
    assert await claim_task_starting(db, first, 0, "logs/test.log", "task.service")
    assert await set_task_running(db, first, 123)
    assert not await set_task_running(db, first, 123)
    assert await set_task_finished(db, first, 2, "process_exit_2")
    assert not await set_task_aborted(db, first)
    row = (await get_tasks(db, query="train"))[0]
    assert row["state"] == "FAILED"
    assert row["status_reason"] == "process_exit_2"
    await db.close()


@pytest.mark.asyncio
async def test_migration_preserves_running_tasks_for_reconciliation(tmp_path):
    path = tmp_path / "old.db"
    connection = sqlite3.connect(path)
    connection.execute(
        """CREATE TABLE tasks (
        id TEXT PRIMARY KEY, username TEXT, command TEXT, work_dir TEXT,
        conda_env TEXT, env_type TEXT, preferred_gpu_id INTEGER, priority INTEGER,
        state TEXT, gpu_id INTEGER, pid INTEGER, exit_code INTEGER, log_path TEXT,
        created_at REAL, started_at REAL, finished_at REAL)"""
    )
    connection.execute(
        "INSERT INTO tasks VALUES ('task', 'alice', 'cmd', '.', NULL, NULL, NULL, 0, "
        "'RUNNING', 0, 1, NULL, NULL, 1, 2, NULL)"
    )
    connection.commit()
    connection.close()

    db = await init_db(str(path))
    row = (await get_tasks(db))[0]
    assert row["state"] == "RUNNING"
    assert row["runner_unit"] is None
    await db.close()


@pytest.mark.asyncio
async def test_gpu_scheduling_setting_persists_by_bus_id(tmp_path):
    db = await init_db(str(tmp_path / "scheduler.db"))
    assert await get_disabled_gpu_bus_ids(db) == set()
    await set_gpu_scheduling_enabled(db, "00000000:41:00.0", False)
    assert await get_disabled_gpu_bus_ids(db) == {"00000000:41:00.0"}
    await set_gpu_scheduling_enabled(db, "00000000:41:00.0", True)
    assert await get_disabled_gpu_bus_ids(db) == set()
    await db.close()
