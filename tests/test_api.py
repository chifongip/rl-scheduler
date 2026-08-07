import os
import pwd
import sqlite3

from fastapi.testclient import TestClient

from admin_sessions import AdminSessionStore
from main import create_app
from models import (
    GpuStatus,
    TaskSubmit,
    count_tasks,
    get_tasks,
    insert_task,
    row_to_status,
    set_gpu_scheduling_enabled,
    set_task_aborted,
)


class FakeGpuManager:
    managed_gpu_ids = [0]
    bus_id_map = {0: "00000000:41:00.0"}

    def __init__(self):
        self.disabled_gpu_ids = set()

    def get_all_gpu_status(self):
        return [
            GpuStatus(
                gpu_id=0,
                name="Fake GPU",
                temperature_c=40,
                memory_used_mb=0,
                memory_total_mb=100,
                memory_utilization_pct=0,
                active_task_id=None,
                scheduling_enabled=self.is_scheduling_enabled(0),
            )
        ]

    def is_scheduling_enabled(self, gpu_id):
        return gpu_id in self.managed_gpu_ids and gpu_id not in self.disabled_gpu_ids

    def set_scheduling_enabled(self, gpu_id, enabled):
        if enabled:
            self.disabled_gpu_ids.discard(gpu_id)
        else:
            self.disabled_gpu_ids.add(gpu_id)

    def set_fan_auto(self, gpu_id):
        return {"fan_supported": True, "fan_mode": "auto"}

    def set_fan_speed(self, gpu_id, speed):
        return {"fan_supported": True, "fan_mode": "manual", "fan_speed_pct": speed}

    def shutdown(self):
        pass


class FakeScheduler:
    def __init__(self, db, gpu_manager):
        self.db = db
        self.gpu_manager = gpu_manager
        self.supervisor_available = True

    async def start(self):
        pass

    async def stop(self):
        pass

    async def is_supervisor_available(self):
        return self.supervisor_available

    async def submit_task(self, **values):
        return await insert_task(self.db, TaskSubmit(**values))

    async def get_tasks(self, state=None, username=None, limit=None, offset=0, query=None):
        rows = await get_tasks(self.db, state, username, limit, offset, query)
        return [row_to_status(row) for row in rows]

    async def abort_task(self, task_id):
        return await set_task_aborted(self.db, task_id)

    async def set_gpu_scheduling_enabled(self, gpu_id, enabled):
        await set_gpu_scheduling_enabled(
            self.db, self.gpu_manager.bus_id_map[gpu_id], enabled
        )
        self.gpu_manager.set_scheduling_enabled(gpu_id, enabled)

    def get_progress(self, task_id):
        return None, None


class UnavailableScheduler(FakeScheduler):
    def __init__(self, db, gpu_manager):
        super().__init__(db, gpu_manager)
        self.supervisor_available = False


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def test_pagination_log_tail_and_admin_header(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    app = create_app(
        db_path=str(tmp_path / "db.sqlite"),
        gpu_factory=FakeGpuManager,
        scheduler_factory=FakeScheduler,
    )
    user = pwd.getpwuid(os.getuid())
    body = {"username": user.pw_name, "command": "echo hello", "work_dir": user.pw_dir}

    with TestClient(app) as client:
        for _ in range(3):
            assert client.post("/tasks/submit", json=body).status_code == 200
        response = client.get("/tasks/status", params={"limit": 2, "offset": 0, "q": "hello"})
        assert response.status_code == 200
        assert response.json()["total"] == 3
        assert len(response.json()["tasks"]) == 2

        assert client.post("/gpus/0/fan", json={"mode": "auto"}).status_code == 403
        assert client.post(
            "/gpus/0/fan",
            json={"mode": "manual", "speed": 50},
            headers={"X-Admin-Password": "secret"},
        ).status_code == 200

        task_id = response.json()["tasks"][0]["id"]

    log = tmp_path / "task.log"
    log.write_text("0123456789")
    connection = sqlite3.connect(tmp_path / "db.sqlite")
    connection.execute("UPDATE tasks SET log_path = ? WHERE id = ?", (str(log), task_id))
    connection.commit()
    connection.close()
    with TestClient(app) as client:
        tail = client.get(f"/tasks/{task_id}/log", params={"tail_bytes": 4})
        assert tail.text == "6789"
        assert tail.headers["X-Log-Next-Offset"] == "10"


def test_unavailable_supervisor_degrades_health_and_rejects_submit(tmp_path):
    app = create_app(
        db_path=str(tmp_path / "db.sqlite"),
        gpu_factory=FakeGpuManager,
        scheduler_factory=UnavailableScheduler,
    )
    user = pwd.getpwuid(os.getuid())
    body = {"username": user.pw_name, "command": "echo hello", "work_dir": user.pw_dir}
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.json()["status"] == "degraded"
        assert health.json()["supervisor_available"] is False
        response = client.post("/tasks/submit", json=body)
        assert response.status_code == 503


def test_admin_session_slides_only_after_success_and_logout(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    monkeypatch.setenv("ADMIN_SESSION_TIMEOUT_SECONDS", "300")
    clock = FakeClock()
    app = create_app(
        db_path=str(tmp_path / "db.sqlite"),
        gpu_factory=FakeGpuManager,
        scheduler_factory=FakeScheduler,
        admin_session_store_factory=lambda timeout: AdminSessionStore(timeout, clock=clock),
    )
    with TestClient(app) as client:
        assert client.post("/admin/session", json={"password": "wrong"}).status_code == 403
        login = client.post("/admin/session", json={"password": "secret"})
        assert login.status_code == 200
        assert login.headers["Cache-Control"] == "no-store"
        assert login.json()["expires_in"] == 300
        token = login.json()["token"]
        headers = {"X-Admin-Token": token}

        clock.advance(250)
        assert client.post("/gpus/0/fan", json={"mode": "auto"}, headers=headers).status_code == 200
        clock.advance(250)
        assert client.post("/gpus/0/fan", json={"mode": "auto"}, headers=headers).status_code == 200

        second = client.post("/admin/session", json={"password": "secret"}).json()["token"]
        second_headers = {"X-Admin-Token": second}
        clock.advance(250)
        assert client.post("/gpus/9/fan", json={"mode": "auto"}, headers=second_headers).status_code == 400
        clock.advance(51)
        assert client.post("/gpus/0/fan", json={"mode": "auto"}, headers=second_headers).status_code == 403

        logout_token = client.post("/admin/session", json={"password": "secret"}).json()["token"]
        logout_headers = {"X-Admin-Token": logout_token}
        assert client.delete("/admin/session", headers=logout_headers).status_code == 200
        assert client.post("/gpus/0/fan", json={"mode": "auto"}, headers=logout_headers).status_code == 403

        restart_token = client.post("/admin/session", json={"password": "secret"}).json()["token"]

    with TestClient(app) as client:
        restart_headers = {"X-Admin-Token": restart_token}
        assert client.post("/gpus/0/fan", json={"mode": "auto"}, headers=restart_headers).status_code == 403


def test_admin_session_is_disabled_without_password(tmp_path, monkeypatch):
    monkeypatch.delenv("ADMIN_PASSWORD", raising=False)
    app = create_app(
        db_path=str(tmp_path / "db.sqlite"),
        gpu_factory=FakeGpuManager,
        scheduler_factory=FakeScheduler,
    )
    with TestClient(app) as client:
        assert client.post("/admin/session", json={"password": "secret"}).status_code == 403


def test_gpu_scheduling_toggle_requires_admin_and_persists(tmp_path, monkeypatch):
    monkeypatch.setenv("ADMIN_PASSWORD", "secret")
    app = create_app(
        db_path=str(tmp_path / "db.sqlite"),
        gpu_factory=FakeGpuManager,
        scheduler_factory=FakeScheduler,
    )
    user = pwd.getpwuid(os.getuid())
    task = {
        "username": user.pw_name,
        "command": "echo hello",
        "work_dir": user.pw_dir,
        "preferred_gpu_id": 0,
    }

    with TestClient(app) as client:
        assert client.put("/gpus/0/scheduling", json={"enabled": False}).status_code == 403
        token = client.post("/admin/session", json={"password": "secret"}).json()["token"]
        assert client.put(
            "/gpus/0/scheduling",
            json={"enabled": False},
            headers={"X-Admin-Token": token},
        ).status_code == 403
        response = client.put(
            "/gpus/0/scheduling",
            json={"enabled": False},
            headers={"X-Admin-Password": "secret"},
        )
        assert response.json() == {"gpu_id": 0, "scheduling_enabled": False}
        assert client.get("/gpus").json()["gpus"][0]["scheduling_enabled"] is False
        rejected = client.post("/tasks/submit", json=task)
        assert rejected.status_code == 400
        assert rejected.json()["detail"] == "GPU 0 is disabled for scheduling"

    with TestClient(app) as client:
        assert client.get("/gpus").json()["gpus"][0]["scheduling_enabled"] is False
