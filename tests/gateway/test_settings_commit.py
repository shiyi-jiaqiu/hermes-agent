"""Gateway commit ownership across SQLite waits, cancellation and session boundaries."""
import asyncio
import sqlite3
import threading
from types import SimpleNamespace

import pytest

from gateway.config import GatewayConfig, Platform
from gateway.session import SessionStore, AsyncSessionStore, SessionSource
from gateway.control.settings import apply_gateway_settings, GatewaySettingsEndpoint
from hermes_cli.model_switch import ModelSwitchResult
from hermes_cli.runtime_settings import RuntimeSettings, SettingsRequest


@pytest.fixture
def runtime(tmp_path, monkeypatch):
    from gateway.run import GatewayRunner
    cfg = {'model': {'default': 'old', 'provider': 'p'}, 'agent': {'reasoning_effort': 'medium'}}
    monkeypatch.setattr('gateway.run._load_gateway_config', lambda **kw: cfg)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    store = SessionStore(tmp_path / 'sessions', GatewayConfig())
    source = SessionSource(Platform.FEISHU, 'chat', user_id='owner', chat_type='dm')
    entry = store.get_or_create_session(source)
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.session_store = store
    runner._normalize_source_for_session_key = lambda source: source
    runner._resolve_profile_home_for_source = lambda source: tmp_path
    runner._evict_cached_agent = lambda key: None
    runner._session_state(entry.session_key).conversation.model_override = RuntimeSettings('old', 'p').route()
    yield runner, source, cfg, entry
    for db in store._db_handles.values():
        db.close()


@pytest.mark.asyncio
async def test_sqlite_wait_keeps_loop_responsive_and_publish_stays_on_owner(runtime, monkeypatch):
    runner, source, cfg, entry = runtime
    db = runner.session_store._db_for_key(entry.session_key)
    db_path = db._conn.execute('PRAGMA database_list').fetchone()[2]
    release, locked = threading.Event(), threading.Event()
    entered = asyncio.Event()
    loop = asyncio.get_running_loop()
    original = db.update_runtime_settings
    def lock_writer():
        with sqlite3.connect(db_path) as other:
            other.execute('BEGIN IMMEDIATE')
            locked.set()
            release.wait(3)
    holder = threading.Thread(target=lock_writer)
    def write(*args):
        holder.start()
        assert locked.wait(3)
        loop.call_soon_threadsafe(entered.set)
        return original(*args)
    monkeypatch.setattr(db, 'update_runtime_settings', write)
    publisher_thread = []
    publish = GatewaySettingsEndpoint.publish
    def publish_here(self, settings):
        publisher_thread.append(threading.get_ident())
        publish(self, settings)
    monkeypatch.setattr(GatewaySettingsEndpoint, 'publish', publish_here)
    operation = asyncio.create_task(apply_gateway_settings(
        runner, source, SettingsRequest(model_target='new'), cfg,
        resolver=lambda **kw: ModelSwitchResult(True, 'new', 'p')))
    try:
        await asyncio.wait_for(entered.wait(), 5)
        assert not operation.done(), 'SQLite wait blocked event-loop callbacks until commit finished'
    finally:
        release.set()
        result = await operation
        await asyncio.to_thread(holder.join)
    assert result.applied
    assert publisher_thread == [threading.get_ident()]
    assert db.get_runtime_settings(entry.session_id)['model'] == 'new'


@pytest.mark.asyncio
@pytest.mark.parametrize('command', ['/new', '/resume saved'])
async def test_cancelled_waiter_keeps_commit_owner_and_fences_turn_and_session_change(runtime, monkeypatch, command):
    from unittest.mock import AsyncMock
    from gateway.platforms.base import MessageEvent
    runner, source, cfg, entry = runtime
    started, release = asyncio.Event(), threading.Event()
    loop = asyncio.get_running_loop()
    store = runner.session_store
    original = store.set_runtime_settings
    def write(*args, **kwargs):
        loop.call_soon_threadsafe(started.set)
        assert release.wait(5)
        return original(*args, **kwargs)
    monkeypatch.setattr(store, 'set_runtime_settings', write)
    operation = asyncio.create_task(apply_gateway_settings(
        runner, source, SettingsRequest(model_target='new'), cfg,
        resolver=lambda **kw: ModelSwitchResult(True, 'new', 'p')))
    await started.wait()
    owned = runner._session_state(entry.session_key).persistent.settings_commit
    class BoundaryReached(Exception):
        pass
    def reached(*args, **kwargs):
        raise BoundaryReached
    runner._invalidate_session_run_generation = reached
    runner._release_running_agent_state = reached
    runner._session_db = object()
    runner._resolve_resume_target = AsyncMock(return_value=('saved', 'saved'))
    runner._resume_access_denied_reply = AsyncMock(return_value=None)
    handler = runner._handle_reset_command if command == '/new' else runner._handle_resume_command
    boundary = asyncio.create_task(handler(MessageEvent(text=command, source=source)))
    try:
        operation.cancel()
        with pytest.raises(asyncio.CancelledError):
            await operation
        assert not owned.done()
        assert runner._claim_active_session_slot(entry.session_key, source)[1]
        await asyncio.sleep(0)
        assert not boundary.done()
    finally:
        release.set()
        assert (await owned).applied
        with pytest.raises(BoundaryReached):
            await boundary
    assert store.get_runtime_settings(entry.session_key)['model'] == 'new'


@pytest.mark.asyncio
async def test_session_changed_during_resolution_rejects_candidate(runtime):
    runner, source, cfg, entry = runtime
    old_id = entry.session_id
    started, release = threading.Event(), threading.Event()
    def resolve(**kwargs):
        started.set()
        assert release.wait(5)
        return ModelSwitchResult(True, 'obsolete', 'p')
    operation = asyncio.create_task(apply_gateway_settings(
        runner, source, SettingsRequest(model_target='obsolete'), cfg, resolver=resolve))
    assert await asyncio.to_thread(started.wait, 5)
    try:
        async with runner._session_state(entry.session_key).persistent.settings_lock:
            replacement = await runner.async_session_store.reset_session(entry.session_key)
            runner._clear_conversation_scope(entry.session_key, reason='test-reset')
    finally:
        release.set()
    result = await operation
    assert not result.applied and 'Session changed' in result.error
    assert runner.session_store._db_for_key(entry.session_key).get_runtime_settings(old_id) is None
    assert replacement.session_id != old_id
