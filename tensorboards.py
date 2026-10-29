from __future__ import annotations

import asyncio
import logging
import os
import re
import socket
import uuid
from dataclasses import asdict, dataclass
from typing import Literal

from pydantic import BaseModel, Field, model_validator

from environments import get_user_home, is_within, resolve_python
from systemd_runner import RunnerError, SupervisorUnavailable, SystemdRunner, TENSORBOARD_UNIT_PREFIX

logger = logging.getLogger(__name__)
ACTIVE_STATES = frozenset({"STARTING", "RUNNING"})
HISTORY_STATES = frozenset({"STOPPED", "FAILED"})


class TensorBoardRequest(BaseModel):
    username: str = Field(min_length=1)
    work_dir: str = Field(min_length=1)
    log_dir: str = Field(min_length=1)
    env_type: Literal['system', 'conda', 'venv'] = 'system'
    conda_env: str | None = None
    port: int | None = Field(default=None, ge=1024, le=65535)

    @model_validator(mode='after')
    def check_environment(self) -> TensorBoardRequest:
        if self.env_type != 'system' and not self.conda_env:
            raise ValueError('Select a Python environment')
        if self.env_type == 'system' and self.conda_env:
            raise ValueError('System Python does not accept an environment name')
        return self


class SessionError(RuntimeError):
    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class TensorBoardSession:
    id: str
    username: str
    work_dir: str
    log_dir: str
    env_type: str
    conda_env: str | None
    python: str
    port: int
    state: str = 'STARTING'
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        # Runtime synchronization is deliberately excluded from the public fields.
        self._lock = asyncio.Lock()

    @property
    def unit(self) -> str:
        return f'{TENSORBOARD_UNIT_PREFIX}{self.id}.service'

    def public(self) -> dict:
        return asdict(self)


class TensorBoardManager:
    def __init__(self, runner: SystemdRunner | None = None, startup_timeout: float = 30.0):
        self.runner = runner or SystemdRunner()
        self.startup_timeout = startup_timeout
        self.sessions: dict[str, TensorBoardSession] = {}
        self.lock = asyncio.Lock()
        self._supervisor_lock = asyncio.Lock()
        self.monitor_task: asyncio.Task | None = None
        self._unconfirmed_stops: set[str] = set()
        self.available = False
        self.closing = False

    async def start(self) -> None:
        self.available = False
        try:
            await self._ensure_supervisor()
        except (RunnerError, OSError, asyncio.TimeoutError):
            self.available = False
            logger.exception('TensorBoard startup cleanup failed')
        self.monitor_task = asyncio.create_task(self._monitor())

    async def close(self) -> None:
        self.closing = True
        if self.monitor_task:
            self.monitor_task.cancel()
            await asyncio.gather(self.monitor_task, return_exceptions=True)
        sessions = [session for session in self.sessions.values()
                    if session.state in ACTIVE_STATES or session.id in self._unconfirmed_stops]
        await asyncio.gather(*(self._close_session(session) for session in sessions))

    async def _close_session(self, session: TensorBoardSession) -> None:
        try:
            await self._stop_session(session)
        except (RunnerError, OSError, asyncio.TimeoutError):
            logger.exception('Could not stop TensorBoard %s', session.id)

    async def _ensure_supervisor(self) -> None:
        # Recovery cleanup must finish once before any sessions can be reserved.
        # Keep this lock separate so supervisor checks never block Stop.
        async with self._supervisor_lock:
            if self.closing or not await self.runner.is_available():
                raise SupervisorUnavailable('TensorBoard supervisor is unavailable')
            if not self.available:
                try:
                    await self.runner.cleanup_tensorboards()
                except (RunnerError, OSError, asyncio.TimeoutError) as exc:
                    raise SupervisorUnavailable('TensorBoard startup cleanup failed') from exc
                self.available = True

    def get(self, session_id: str) -> TensorBoardSession:
        if session_id not in self.sessions:
            raise SessionError('TensorBoard session not found', 404)
        return self.sessions[session_id]

    async def launch(self, body: TensorBoardRequest) -> TensorBoardSession:
        home = get_user_home(body.username)
        if home is None:
            raise SessionError('System user not found')
        work_dir = os.path.realpath(os.path.join(home, body.work_dir))
        log_dir = os.path.realpath(os.path.join(work_dir, body.log_dir))
        for path in (work_dir, log_dir):
            if not is_within(path, home):
                raise SessionError('Directories must be within the selected user’s home', 403)
            if not os.path.isdir(path):
                raise SessionError(f'Directory does not exist: {path}')
        python = '/usr/bin/python3' if body.env_type == 'system' else resolve_python(
            body.username, body.conda_env, body.env_type
        )
        if not python or not os.path.isfile(python):
            raise SessionError('Python environment not found')
        # Do not realpath Python: venv interpreters can be symlinks to system Python.
        await self._ensure_supervisor()
        async with self.lock:
            if self.closing:
                raise SupervisorUnavailable('TensorBoard supervisor is unavailable')
            for session in self.sessions.values():
                if session.state in ACTIVE_STATES and (
                    session.username, session.log_dir, session.python
                ) == (body.username, log_dir, python):
                    if body.port is not None and body.port != session.port:
                        raise SessionError('Session already uses another port', 409)
                    return session
            reserved = {s.port for s in self.sessions.values()
                        if s.state in ACTIVE_STATES or s.id in self._unconfirmed_stops}
            ports = [body.port] if body.port is not None else range(6006, 6100)
            port = next((p for p in ports if p not in reserved and self._port_free(p)), None)
            if port is None:
                raise SessionError('TensorBoard port is unavailable', 409)
            session = TensorBoardSession(uuid.uuid4().hex, body.username, work_dir, log_dir,
                                        body.env_type, body.conda_env, python, port)
            self.sessions[session.id] = session
        return await self._start_session(session)

    async def _start_session(self, session: TensorBoardSession) -> TensorBoardSession:
        try:
            # Serialize service creation with Stop for this session only.
            async with session._lock:
                if session.state != 'STARTING':
                    return session
                if self.closing:
                    session.state = 'STOPPED'
                    return session
                await self.runner.launch_tensorboard(session)
            deadline = asyncio.get_running_loop().time() + self.startup_timeout
            while asyncio.get_running_loop().time() < deadline:
                if session.state != 'STARTING':
                    return session
                if self.closing:
                    return await self._stop_session(session)
                status = await self.runner.inspect(session.unit)
                if not status.running:
                    raise RunnerError(f'TensorBoard exited: {status.result} (exit {status.exit_status})')
                ready = await self._ready(session.port)
                async with session._lock:
                    # Stop or shutdown can finish while a readiness probe is in flight.
                    if session.state != 'STARTING':
                        return session
                    if ready and not self.closing:
                        session.state = 'RUNNING'
                        return session
                await asyncio.sleep(0.2)
            raise RunnerError('TensorBoard startup timed out')
        except (RunnerError, OSError, asyncio.TimeoutError) as exc:
            await self._fail_startup(session, str(exc))
            return session
        except asyncio.CancelledError:
            await asyncio.shield(self._fail_startup(session, 'Launch cancelled'))
            raise

    async def _fail_startup(self, session: TensorBoardSession, reason: str) -> None:
        async with session._lock:
            if session.state != 'STARTING':
                return
            session.state = 'FAILED'
            session.failure_reason = reason
            try:
                await self._cleanup(session)
            except (RunnerError, OSError, asyncio.TimeoutError):
                logger.exception('TensorBoard failure cleanup failed')

    @staticmethod
    def _port_free(port: int) -> bool:
        try:
            with socket.socket() as sock:
                # Match TensorBoard's HTTP server when reusing a stopped port.
                sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                sock.bind(('0.0.0.0', port))
            return True
        except OSError:
            return False

    @staticmethod
    async def _ready(port: int) -> bool:
        writer = None
        try:
            reader, writer = await asyncio.wait_for(asyncio.open_connection('127.0.0.1', port), timeout=1)
            writer.write(b'GET / HTTP/1.0\r\nHost: localhost\r\n\r\n')
            await asyncio.wait_for(writer.drain(), timeout=1)
            line = await asyncio.wait_for(reader.readline(), timeout=1)
            return bool(re.match(rb'HTTP/1\.[01] (200|30[1278])\b', line))
        except (OSError, asyncio.TimeoutError, ValueError):
            return False
        finally:
            if writer:
                writer.close()
                try:
                    await asyncio.wait_for(writer.wait_closed(), timeout=1)
                except (OSError, asyncio.TimeoutError):
                    pass

    async def stop(self, session_id: str) -> TensorBoardSession:
        return await self._stop_session(self.get(session_id))

    async def _stop_session(self, session: TensorBoardSession) -> TensorBoardSession:
        async with session._lock:
            if session.state == 'STOPPED' and session.id not in self._unconfirmed_stops:
                return session
            await self._cleanup(session)
            session.state = 'STOPPED'
            return session

    async def _cleanup(self, session: TensorBoardSession) -> None:
        # Keep ownership and reserve the port until service termination is confirmed.
        self._unconfirmed_stops.add(session.id)
        await self.runner.cleanup(session.unit)
        self._unconfirmed_stops.discard(session.id)

    def _remove_history(self, session_ids: list[str]) -> int:
        # Called under the registry lock by both single and bulk deletion.
        if self._unconfirmed_stops.intersection(session_ids):
            raise SessionError('Stop failed TensorBoard services before clearing their history', 409)
        for session_id in session_ids:
            del self.sessions[session_id]
        return len(session_ids)

    async def delete_history(self, session_id: str) -> int:
        async with self.lock:
            session = self.get(session_id)
            if session.state not in HISTORY_STATES:
                raise SessionError('Stop the TensorBoard session before deleting its history', 409)
            return self._remove_history([session_id])

    async def clear_history(self, username: str | None = None) -> int:
        async with self.lock:
            session_ids = [session.id for session in self.sessions.values()
                           if session.state in HISTORY_STATES
                           and (username is None or session.username == username)]
            return self._remove_history(session_ids)

    async def _monitor(self) -> None:
        while True:
            await asyncio.sleep(2)
            # Inspection can await while other requests change the registry.
            for session in list(self.sessions.values()):
                if session.state != 'RUNNING':
                    continue
                try:
                    status = await self.runner.inspect(session.unit)
                    async with session._lock:
                        if session.state == 'RUNNING' and not status.running:
                            session.state = 'FAILED'
                            session.failure_reason = f'TensorBoard exited: {status.result} (exit {status.exit_status})'
                except (RunnerError, OSError, asyncio.TimeoutError):
                    logger.exception('Could not inspect TensorBoard %s', session.id)
