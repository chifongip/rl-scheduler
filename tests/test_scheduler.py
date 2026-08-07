import asyncio

import pytest

from models import (
    TaskSubmit,
    claim_task_starting,
    get_disabled_gpu_bus_ids,
    get_tasks,
    init_db,
    insert_task,
)
from scheduler import Scheduler, _parse_log_progress
from systemd_runner import UnitStatus


class FakeGpuManager:
    managed_gpu_ids = [0, 1]
    bus_id_map = {0: "bus-0", 1: "bus-1"}

    def __init__(self):
        self.active_tasks = {}
        self.disabled_gpu_ids = set()

    def get_available_gpu_ids(self):
        return [gid for gid in self.managed_gpu_ids if self.is_scheduling_enabled(gid)]

    def is_scheduling_enabled(self, gpu_id):
        return gpu_id not in self.disabled_gpu_ids

    def set_scheduling_enabled(self, gpu_id, enabled):
        if enabled:
            self.disabled_gpu_ids.discard(gpu_id)
        else:
            self.disabled_gpu_ids.add(gpu_id)

    def register_task(self, task_id, gpu_id, process_group_id=None):
        self.active_tasks[task_id] = gpu_id

    def unregister_task(self, task_id):
        self.active_tasks.pop(task_id, None)


class FakeRunner:
    available = True

    def __init__(self):
        self.statuses = {}
        self.stopped = []
        self.cleaned = []
        self.launched = []

    @staticmethod
    def unit_name(task_id):
        return f"rl-scheduler-task-{task_id}.service"

    async def is_available(self):
        return self.available

    async def inspect(self, unit):
        return self.statuses.get(
            unit, UnitStatus(unit, "not-found", "inactive", "dead", 0, -1, "unknown")
        )

    async def stop(self, unit):
        self.stopped.append(unit)

    async def cleanup(self, unit):
        self.cleaned.append(unit)

    async def launch(self, **values):
        self.launched.append(values)
        unit = values["unit"]
        return UnitStatus(unit, "loaded", "active", "running", 123, 0, "success")


def test_progress_parser_reads_latest_values(tmp_path):
    log = tmp_path / "train.log"
    log.write_text("Learning iteration 2/10\nETA: 00:01:30\n")
    assert _parse_log_progress(str(log)) == (0.2, 90)


@pytest.mark.asyncio
async def test_submit_wakes_scheduler_and_preserves_priority_order(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await init_db(str(tmp_path / "db.sqlite"))
    scheduler = Scheduler(db, FakeGpuManager(), poll_interval=60)
    scheduler._wake_event.clear()
    await scheduler.submit_task("alice", "echo low", ".", priority=1)
    await scheduler.submit_task("alice", "echo high", ".", priority=9)
    assert scheduler._wake_event.is_set()
    rows = await get_tasks(db, state="PENDING")
    assert {row["command"] for row in rows} == {"echo low", "echo high"}
    cursor = await db.execute("SELECT command FROM tasks WHERE state='PENDING' ORDER BY priority DESC")
    assert [row[0] for row in await cursor.fetchall()] == ["echo high", "echo low"]
    await db.close()


@pytest.mark.asyncio
async def test_disabled_gpu_cannot_launch_pending_task(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await init_db(str(tmp_path / "db.sqlite"))
    task_id = await insert_task(db, TaskSubmit(username="alice", command="train"))
    gpu = FakeGpuManager()
    runner = FakeRunner()
    scheduler = Scheduler(db, gpu, runner=runner)
    gpu.register_task("existing", 0)
    scheduler._wake_event.clear()
    await scheduler.set_gpu_scheduling_enabled(0, False)

    launched = await scheduler._spawn_task(
        task_id, "train", str(tmp_path), 0, username="alice"
    )

    assert gpu.active_tasks["existing"] == 0
    assert scheduler._wake_event.is_set()
    assert await get_disabled_gpu_bus_ids(db) == {"bus-0"}
    assert not launched
    assert runner.launched == []
    assert (await get_tasks(db))[0]["state"] == "PENDING"
    await db.close()


@pytest.mark.asyncio
async def test_auto_select_skips_disabled_gpu(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await init_db(str(tmp_path / "db.sqlite"))
    task_id = await insert_task(db, TaskSubmit(username="alice", command="train"))
    gpu = FakeGpuManager()
    gpu.set_scheduling_enabled(0, False)
    runner = FakeRunner()
    scheduler = Scheduler(db, gpu, runner=runner)

    await scheduler._dispatch_pending_tasks()

    assert runner.launched[0]["gpu_id"] == 1
    row = (await get_tasks(db))[0]
    assert row["id"] == task_id
    assert row["gpu_id"] == 1
    assert row["state"] == "RUNNING"
    await db.close()


@pytest.mark.asyncio
async def test_reconcile_survives_restart_and_records_completion(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await init_db(str(tmp_path / "db.sqlite"))
    task_id = await insert_task(db, TaskSubmit(username="alice", command="train"))
    unit = FakeRunner.unit_name(task_id)
    log = str(tmp_path / "task.log")
    await claim_task_starting(db, task_id, 0, log, unit)
    runner = FakeRunner()
    runner.statuses[unit] = UnitStatus(unit, "loaded", "active", "running", 4321, 0, "success")
    gpu = FakeGpuManager()
    scheduler = Scheduler(db, gpu, runner=runner)

    await scheduler._reconcile_managed_tasks()
    row = (await get_tasks(db))[0]
    assert row["state"] == "RUNNING"
    assert row["pid"] == 4321
    assert gpu.active_tasks[task_id] == 0

    runner.statuses[unit] = UnitStatus(unit, "loaded", "active", "exited", 0, 0, "success")
    await scheduler._reconcile_managed_tasks()
    row = (await get_tasks(db))[0]
    assert row["state"] == "COMPLETED"
    assert unit in runner.cleaned
    await db.close()


@pytest.mark.asyncio
async def test_missing_starting_unit_requeues_and_shutdown_preserves_tasks(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await init_db(str(tmp_path / "db.sqlite"))
    task_id = await insert_task(db, TaskSubmit(username="alice", command="train"))
    unit = FakeRunner.unit_name(task_id)
    await claim_task_starting(db, task_id, 0, str(tmp_path / "missing.log"), unit)
    scheduler = Scheduler(db, FakeGpuManager(), runner=FakeRunner())
    await scheduler._reconcile_managed_tasks()
    assert (await get_tasks(db))[0]["state"] == "PENDING"

    await claim_task_starting(db, task_id, 0, str(tmp_path / "task.log"), unit)
    scheduler.runner.statuses[unit] = UnitStatus(unit, "loaded", "active", "running", 9, 0, "success")
    await scheduler._reconcile_managed_tasks()
    await scheduler.stop()
    assert (await get_tasks(db))[0]["state"] == "RUNNING"
    assert scheduler.runner.stopped == []
    await db.close()


@pytest.mark.asyncio
async def test_abort_stops_managed_unit(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    db = await init_db(str(tmp_path / "db.sqlite"))
    task_id = await insert_task(db, TaskSubmit(username="alice", command="train"))
    unit = FakeRunner.unit_name(task_id)
    await claim_task_starting(db, task_id, 0, str(tmp_path / "task.log"), unit)
    runner = FakeRunner()
    scheduler = Scheduler(db, FakeGpuManager(), runner=runner)
    assert await scheduler.abort_task(task_id)
    assert runner.stopped == [unit]
    assert (await get_tasks(db))[0]["status_reason"] == "aborted_by_user"
    await db.close()
