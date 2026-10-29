import asyncio
import socket
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest

from systemd_runner import SystemdRunner, UnitStatus
from tensorboards import SessionError, TensorBoardManager, TensorBoardRequest


@pytest.fixture
def setup_manager(tmp_path, monkeypatch):
    monkeypatch.setattr('tensorboards.get_user_home', lambda user: str(tmp_path) if user == 'alice' else None)
    runner = AsyncMock()
    runner.is_available.return_value = True
    runner.inspect.return_value = UnitStatus('unit', 'loaded', 'active', 'running', 123, 0, 'success')
    manager = TensorBoardManager(runner=runner, startup_timeout=0.02)
    manager.available = True
    manager._ready = AsyncMock(return_value=True)
    monkeypatch.setattr(manager, '_port_free', lambda port: True)
    body = TensorBoardRequest(username='alice', work_dir=str(tmp_path), log_dir='.')
    return manager, runner, body


@pytest.mark.asyncio
async def test_launch_duplicate_and_conflict(setup_manager):
    manager, runner, body = setup_manager
    first, second = await asyncio.gather(manager.launch(body), manager.launch(body))
    assert first is second
    assert first.state == 'RUNNING'
    runner.launch_tensorboard.assert_awaited_once()
    with pytest.raises(SessionError, match='another port'):
        await manager.launch(body.model_copy(update={'port': 6010}))
    await manager.stop(first.id)
    assert first.state == 'STOPPED'
    runner.cleanup.assert_awaited_once_with(first.unit)


@pytest.mark.asyncio
async def test_ports_and_user_boundaries(setup_manager, tmp_path):
    manager, runner, body = setup_manager
    with pytest.raises(SessionError):
        await manager.launch(body.model_copy(update={'username': 'missing'}))
    with pytest.raises(SessionError) as error:
        await manager.launch(body.model_copy(update={'log_dir': '..'}))
    assert error.value.status_code == 403
    (tmp_path / 'escape').symlink_to(tmp_path.parent, target_is_directory=True)
    with pytest.raises(SessionError):
        await manager.launch(body.model_copy(update={'log_dir': 'escape'}))
    manager._port_free = lambda port: False
    with pytest.raises(SessionError) as error:
        await manager.launch(body)
    assert error.value.status_code == 409
    runner.launch_tensorboard.assert_not_awaited()


@pytest.mark.asyncio
async def test_timeout_and_exit(setup_manager):
    manager, runner, body = setup_manager
    manager._ready.return_value = False
    session = await manager.launch(body)
    assert session.state == 'FAILED'
    assert 'timed out' in session.failure_reason
    runner.cleanup.assert_awaited_with(session.unit)
    runner.inspect.return_value = replace(runner.inspect.return_value, active_state='failed', sub_state='failed', exit_status=1)
    session = await manager.launch(body)
    assert session.state == 'FAILED'
    assert 'exit 1' in session.failure_reason


@pytest.mark.asyncio
async def test_shutdown_and_orphans(setup_manager):
    manager, runner, body = setup_manager
    await manager.start()
    runner.cleanup_tensorboards.assert_awaited_once()
    session = await manager.launch(body)
    await manager.close()
    assert session.state == 'STOPPED'
    runner.cleanup.assert_awaited_with(session.unit)


@pytest.mark.asyncio
async def test_venv_interpreter_preserves_symlink(setup_manager, tmp_path, monkeypatch):
    manager, runner, body = setup_manager
    path = str(tmp_path / 'venv' / 'bin' / 'python')
    monkeypatch.setattr('tensorboards.resolve_python', lambda *args: path)
    monkeypatch.setattr('tensorboards.os.path.isfile', lambda p: True)
    session = await manager.launch(body.model_copy(update={'env_type': 'venv', 'conda_env': str(tmp_path / 'venv')}))
    assert session.python == path


@pytest.mark.asyncio
async def test_runner_literal_arguments(setup_manager):
    manager, _, body = setup_manager
    session = await manager.launch(body)
    session.python = '/home/alice/envs/$HOME with spaces/bin/python'
    session.log_dir = '/home/alice/logs with spaces/$HOME; touch evil'
    runner = SystemdRunner()
    runner._run = AsyncMock(return_value=(0, '', ''))
    runner.inspect = AsyncMock()
    await runner.launch_tensorboard(session)
    args = runner._run.call_args.args
    assert '/bin/bash' not in args
    assert '--setenv=CUDA_VISIBLE_DEVICES=' in args
    assert args[args.index('--') + 1] == session.python
    assert '/home/alice/logs with spaces/$$HOME; touch evil' in args


def test_port_check_allows_reuse_after_server_closes_connection():
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.settimeout(1)
        listener.bind(('127.0.0.1', 0))
        listener.listen()
        port = listener.getsockname()[1]
        assert not TensorBoardManager._port_free(port)

        with socket.socket() as client:
            client.settimeout(1)
            client.connect(('127.0.0.1', port))
            connection, _ = listener.accept()
            with connection:
                # HTTP/1.0 closes from the server side, leaving TIME_WAIT there.
                connection.shutdown(socket.SHUT_WR)
                assert client.recv(1) == b''

    assert TensorBoardManager._port_free(port)


@pytest.mark.asyncio
async def test_real_readiness():
    async def handle(reader, writer):
        await reader.read(4096)
        writer.write(b'HTTP/1.0 200 OK\r\n\r\n')
        await writer.drain()
        writer.close()
    server = await asyncio.start_server(handle, '127.0.0.1', 0)
    try:
        assert await TensorBoardManager._ready(server.sockets[0].getsockname()[1])
    finally:
        server.close()
        await server.wait_closed()


@pytest.mark.asyncio
async def test_unavailable_supervisor_and_missing_environment(setup_manager, monkeypatch):
    from systemd_runner import SupervisorUnavailable
    manager, runner, body = setup_manager
    runner.is_available.return_value = False
    with pytest.raises(SupervisorUnavailable):
        await manager.launch(body)
    runner.is_available.return_value = True
    monkeypatch.setattr('tensorboards.resolve_python', lambda *args: None)
    with pytest.raises(SessionError, match='environment not found'):
        await manager.launch(body.model_copy(update={'env_type': 'conda', 'conda_env': 'missing'}))
    runner.launch_tensorboard.assert_not_awaited()


@pytest.mark.asyncio
async def test_startup_cleanup_failure_disables_launches(setup_manager):
    from systemd_runner import RunnerError, SupervisorUnavailable
    manager, runner, body = setup_manager
    runner.cleanup_tensorboards.side_effect = RunnerError('cleanup failed')
    await manager.start()
    try:
        with pytest.raises(SupervisorUnavailable):
            await manager.launch(body)
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_orphan_cleanup_only_matches_managed_units():
    runner = SystemdRunner()
    unit = 'rl-scheduler-tensorboard-' + 'a' * 32 + '.service'
    runner._run = AsyncMock(return_value=(0, f'{unit} loaded active running\nrl-scheduler-task-abc.service loaded active running\nrl-scheduler-tensorboard-other.service loaded active running\n', ''))
    runner.cleanup = AsyncMock()
    await runner.cleanup_tensorboards()
    runner.cleanup.assert_awaited_once_with(unit)


@pytest.mark.asyncio
async def test_unexpected_exit(setup_manager, monkeypatch):
    manager, runner, body = setup_manager
    session = await manager.launch(body)
    runner.inspect.return_value = replace(runner.inspect.return_value, sub_state='exited')
    original_sleep = asyncio.sleep
    calls = 0
    async def tick(_):
        nonlocal calls
        calls += 1
        if calls > 1:
            raise asyncio.CancelledError
        await original_sleep(0)
    monkeypatch.setattr('tensorboards.asyncio.sleep', tick)
    with pytest.raises(asyncio.CancelledError):
        await manager._monitor()
    assert session.state == 'FAILED'
    assert 'exited' in session.failure_reason


@pytest.mark.asyncio
async def test_clear_history_preserves_active_sessions_and_other_users(setup_manager):
    manager, runner, body = setup_manager
    session = await manager.launch(body)
    manager.sessions['stopped'] = replace(session, id='stopped', state='STOPPED')
    manager.sessions['failed'] = replace(session, id='failed', state='FAILED')
    manager.sessions['starting'] = replace(session, id='starting', state='STARTING')
    manager.sessions['bob'] = replace(session, id='bob', username='bob', state='STOPPED')
    assert await manager.clear_history('alice') == 2
    assert set(manager.sessions) == {session.id, 'starting', 'bob'}
    assert await manager.clear_history('alice') == 0
    assert await manager.clear_history() == 1
    assert set(manager.sessions) == {session.id, 'starting'}
    runner.cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_failed_cleanup_keeps_history_and_reserves_port(setup_manager):
    from systemd_runner import RunnerError
    manager, runner, body = setup_manager
    manager._ready.return_value = False
    runner.cleanup.side_effect = RunnerError('stop not confirmed')
    failed = await manager.launch(body)
    assert failed.state == 'FAILED'
    with pytest.raises(SessionError) as error:
        await manager.clear_history('alice')
    assert error.value.status_code == 409
    assert manager.get(failed.id) is failed
    manager._ready.return_value = True
    another = await manager.launch(body)
    assert another.port != failed.port
    runner.cleanup.side_effect = None
    await manager.stop(failed.id)
    assert await manager.clear_history('alice') == 1
    assert manager.get(another.id).state == 'RUNNING'


@pytest.mark.asyncio
async def test_cancelled_launch_preserves_cancellation_on_cleanup_failure(setup_manager):
    from systemd_runner import RunnerError
    manager, runner, body = setup_manager
    runner.launch_tensorboard.side_effect = asyncio.CancelledError
    runner.cleanup.side_effect = RunnerError('cannot stop')
    with pytest.raises(asyncio.CancelledError):
        await manager.launch(body)
    failed = next(iter(manager.sessions.values()))
    assert failed.state == 'FAILED'
    with pytest.raises(SessionError):
        await manager.clear_history()
    runner.cleanup.side_effect = None
    await manager.close()
    assert not manager._unconfirmed_stops


@pytest.mark.asyncio
async def test_diagnostics_limit_counts_bytes():
    runner = SystemdRunner()
    runner._run = AsyncMock(return_value=(0, '\N{SNOWMAN}' * 40000, ''))
    output = await runner.tensorboard_logs('test.service')
    assert len(output.encode('utf-8')) <= 65536
    assert output.endswith('\N{SNOWMAN}')


@pytest.mark.asyncio
async def test_shutdown_skips_confirmed_stopped_history(setup_manager):
    manager, runner, body = setup_manager
    session = await manager.launch(body)
    await manager.stop(session.id)
    runner.cleanup.reset_mock()
    await manager.close()
    runner.cleanup.assert_not_awaited()


@pytest.mark.asyncio
async def test_delete_individual_history(setup_manager):
    manager, runner, body = setup_manager
    session = await manager.launch(body)
    with pytest.raises(SessionError) as error:
        await manager.delete_history(session.id)
    assert error.value.status_code == 409
    await manager.stop(session.id)
    manager.sessions['other'] = replace(session, id='other', state='FAILED')
    assert await manager.delete_history(session.id) == 1
    assert set(manager.sessions) == {'other'}
    with pytest.raises(SessionError) as error:
        await manager.delete_history(session.id)
    assert error.value.status_code == 404
    manager._unconfirmed_stops.add('other')
    with pytest.raises(SessionError) as error:
        await manager.delete_history('other')
    assert error.value.status_code == 409
    assert set(manager.sessions) == {'other'}


@pytest.mark.asyncio
async def test_supervisor_recovers_without_restart(setup_manager):
    manager, runner, body = setup_manager
    runner.is_available.return_value = False
    await manager.start()
    try:
        assert not manager.available
        runner.is_available.return_value = True
        first, duplicate = await asyncio.gather(manager.launch(body), manager.launch(body))
        assert first is duplicate
        assert first.state == 'RUNNING'
        assert manager.available
        runner.cleanup_tensorboards.assert_awaited_once()
        runner.launch_tensorboard.assert_awaited_once()
        # A later outage must not repeat orphan cleanup and stop managed sessions.
        runner.is_available.return_value = False
        from systemd_runner import SupervisorUnavailable
        with pytest.raises(SupervisorUnavailable):
            await manager.launch(body)
        runner.is_available.return_value = True
        assert await manager.launch(body) is first
        runner.cleanup_tensorboards.assert_awaited_once()
    finally:
        runner.is_available.return_value = True
        await manager.close()


@pytest.mark.asyncio
async def test_failed_startup_cleanup_is_retried_before_recovery(setup_manager):
    from systemd_runner import RunnerError, SupervisorUnavailable
    manager, runner, body = setup_manager
    runner.cleanup_tensorboards.side_effect = RunnerError('temporary cleanup failure')
    await manager.start()
    try:
        with pytest.raises(SupervisorUnavailable):
            await manager.launch(body)
        runner.launch_tensorboard.assert_not_awaited()
        runner.cleanup_tensorboards.side_effect = None
        assert (await manager.launch(body)).state == 'RUNNING'
        assert runner.cleanup_tensorboards.await_count == 3
    finally:
        await manager.close()


@pytest.mark.asyncio
async def test_slow_launch_does_not_block_other_session_stop(setup_manager, tmp_path):
    manager, runner, body = setup_manager
    existing = await manager.launch(body)
    (tmp_path / 'other').mkdir()
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_launch(session):
        entered.set()
        await release.wait()

    runner.launch_tensorboard.side_effect = slow_launch
    launching = asyncio.create_task(manager.launch(body.model_copy(update={'log_dir': 'other'})))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        stopped = await asyncio.wait_for(manager.stop(existing.id), 1)
        assert stopped.state == 'STOPPED'
        assert not launching.done()
        runner.cleanup.assert_awaited_once_with(existing.unit)
    finally:
        release.set()
        await launching
        await manager.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('shutdown', [False, True])
async def test_stop_during_readiness_cannot_resurrect_session(setup_manager, shutdown):
    manager, runner, body = setup_manager
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_ready(port):
        entered.set()
        await release.wait()
        return True

    manager._ready.side_effect = slow_ready
    launching = asyncio.create_task(manager.launch(body))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        session = next(iter(manager.sessions.values()))
        assert session.state == 'STARTING'
        if shutdown:
            await asyncio.wait_for(manager.close(), 1)
        else:
            await asyncio.wait_for(manager.stop(session.id), 1)
        assert session.state == 'STOPPED'
        # Removing history must be safe even if the old readiness probe is pending.
        assert await manager.delete_history(session.id) == 1
        release.set()
        result = await asyncio.wait_for(launching, 1)
        assert result is session and result.state == 'STOPPED'
        assert not manager.sessions
        runner.launch_tensorboard.assert_awaited_once()
        runner.cleanup.assert_awaited_once_with(session.unit)
    finally:
        release.set()
        await launching
        await manager.close()


@pytest.mark.asyncio
async def test_pending_duplicate_reuses_reserved_session_and_port(setup_manager):
    manager, runner, body = setup_manager
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_ready(port):
        entered.set()
        await release.wait()
        return True

    manager._ready.side_effect = slow_ready
    launching = asyncio.create_task(manager.launch(body))
    try:
        await asyncio.wait_for(entered.wait(), 1)
        duplicate = await asyncio.wait_for(manager.launch(body), 1)
        assert duplicate.state == 'STARTING'
        assert len(manager.sessions) == 1
        runner.launch_tensorboard.assert_awaited_once()
        with pytest.raises(SessionError) as error:
            await manager.launch(body.model_copy(update={'port': duplicate.port + 1}))
        assert error.value.status_code == 409
        release.set()
        assert await launching is duplicate
        assert duplicate.state == 'RUNNING'
    finally:
        release.set()
        await launching
        await manager.close()


@pytest.mark.asyncio
async def test_shutdown_during_service_creation_stops_created_unit(setup_manager):
    manager, runner, body = setup_manager
    entered, release = asyncio.Event(), asyncio.Event()

    async def slow_launch(session):
        entered.set()
        await release.wait()

    runner.launch_tensorboard.side_effect = slow_launch
    launching = asyncio.create_task(manager.launch(body))
    closing = None
    try:
        await asyncio.wait_for(entered.wait(), 1)
        closing = asyncio.create_task(manager.close())
        await asyncio.sleep(0)
        assert manager.closing
        release.set()
        session = await asyncio.wait_for(launching, 1)
        await asyncio.wait_for(closing, 1)
        assert session.state == 'STOPPED'
        runner.cleanup.assert_awaited_once_with(session.unit)
        manager._ready.assert_not_awaited()
    finally:
        release.set()
        await launching
        if closing is not None:
            await closing
        else:
            await manager.close()
