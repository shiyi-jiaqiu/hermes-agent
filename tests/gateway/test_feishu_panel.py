"""Panel ownership invariants using the real controller and explicit narrow endpoints."""
from __future__ import annotations

import asyncio
from dataclasses import replace
import threading
from types import SimpleNamespace

import pytest

from gateway.config import Platform
from gateway.platforms.base import SendResult
from gateway.session import SessionSource
from gateway.control.panel_service import HermesPanelControlService, PanelControlResult
from plugins.platforms.feishu.panel.controller import FeishuPanelController


SOURCE = SessionSource(Platform.FEISHU, "oc_chat", user_id="ou_owner", chat_type="dm")


class Service:
    def __init__(self):
        self.closed = False
        self.active = 0
        self.load_started = asyncio.Event()
        self.control_started = asyncio.Event()
        self.block_load = self.block_control = False
        self.release = asyncio.Event()
        self.executions = []

    async def snapshot(self, **options):
        assert not self.closed
        if options["include_catalog"] and self.block_load:
            self.active += 1
            self.load_started.set()
            try:
                await self.release.wait()
            finally:
                self.active -= 1
                assert not self.closed
        return {"effective_model": "test-model", "effective_reasoning": "medium",
                "reasoning_options": [{"value": "low", "label": "low"}],
                "model_providers": [], "model_options": [], "sessions": [], "status_text": "Idle"}

    async def execute(self, **options):
        assert not self.closed
        self.executions.append(options["target"])
        self.active += 1
        self.control_started.set()
        try:
            if self.block_control:
                await self.release.wait()
            return PanelControlResult(True, "Applied")
        finally:
            self.active -= 1
            assert not self.closed

    async def close(self):
        assert self.active == 0
        self.closed = True


class Adapter:
    _app_id = "app"

    def __init__(self):
        self.cards, self.patches = [], []
        self.patch_started = asyncio.Event()
        self.patch_release = asyncio.Event()
        self.block_patch = False
        self.fail_send = False
        self.patch_inflight = self.max_inflight = 0

    def is_control_panel_operator_authorized(self, source, open_id):
        return open_id == "ou_owner"

    async def send_coding_progress_card(self, chat_id, card, **kwargs):
        if self.fail_send:
            return SendResult(False, error="send failed")
        self.cards.append(card)
        return SendResult(True, message_id=f"message-{len(self.cards)}")

    async def patch_interactive_message(self, *, message_id, card):
        self.patch_inflight += 1
        self.max_inflight = max(self.max_inflight, self.patch_inflight)
        self.patch_started.set()
        try:
            if self.block_patch:
                await self.patch_release.wait()
            self.patches.append((message_id, card))
            return SendResult(True, message_id=message_id)
        finally:
            self.patch_inflight -= 1


async def opened(service=None, adapter=None, **kwargs):
    service, adapter = service or Service(), adapter or Adapter()
    controller = FeishuPanelController(adapter, service)
    result = await controller.open(source=SOURCE, session_key="key", owner_open_id="ou_owner",
                                   status_text="", metadata=None, initial_view="home", **kwargs)
    assert result.success
    state = next(iter(controller._states.values()))
    return controller, adapter, service, state


def action(controller, state, op="nav", target="reasoning", nonce="click", **kwargs):
    return {"panel_action": True, "v": 1, "panel": state.panel_id,
            "rev": controller._states[state.panel_id].revision, "op": op, "target": target,
            "nonce": nonce, **kwargs}


def handle(controller, payload, *, owner="ou_owner", chat="oc_chat"):
    return controller.handle_sync(payload, open_id=owner, chat_id=chat, loop=controller._loop)


@pytest.mark.asyncio
async def test_message_binding_survives_navigation_and_sdk_thread_callback():
    controller, adapter, service, state = await opened()
    try:
        payload = action(controller, state)
        response = await asyncio.to_thread(handle, controller, payload)
        assert response.toast_type == "info"
        await asyncio.gather(*controller._tasks)
        assert controller._messages[state.panel_id] == "message-1"
        assert controller._states[state.panel_id].view == "reasoning"
        assert adapter.patches[-1][0] == "message-1"
        assert "推理设置" in adapter.patches[-1][1]["header"]["title"]["content"]
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_close_joins_navigation_and_control_before_service_close():
    service = Service()
    service.block_load = service.block_control = True
    controller, adapter, service, state = await opened(service)
    handle(controller, action(controller, state, target="model"))
    await service.load_started.wait()
    handle(controller, action(controller, state, op="exec", target="preset", index=0, nonce="apply"))
    await service.control_started.wait()
    tasks = list(controller._tasks)
    await controller.close()
    assert service.closed and all(task.done() for task in tasks)
    assert not controller._states and not controller._messages and not controller._tasks
    assert "失效" in handle(controller, action_payload(state)).toast


def action_payload(state):
    return {"panel_action": True, "v": 1, "panel": state.panel_id, "rev": 0,
            "op": "exec", "target": "new", "nonce": "old-card"}


@pytest.mark.asyncio
async def test_one_sender_serializes_updates_and_coalesces_to_latest_view():
    controller, adapter, service, state = await opened()
    adapter.block_patch = True
    try:
        handle(controller, action(controller, state, target="reasoning"))
        await adapter.patch_started.wait()
        handle(controller, action(controller, state, op="home", nonce="home"))
        handle(controller, action(controller, state, target="confirm_new", nonce="latest"))
        adapter.patch_release.set()
        await asyncio.gather(*controller._tasks)
        assert adapter.max_inflight == 1
        assert len(adapter.patches) == 2
        assert "确认新建会话" in adapter.patches[-1][1]["header"]["title"]["content"]
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_owner_chat_revision_dedup_and_failed_replacement():
    controller, adapter, service, state = await opened()
    try:
        payload = action(controller, state, op="exec", target="new")
        assert handle(controller, payload, owner="ou_other").toast_type == "error"
        assert handle(controller, payload, chat="oc_other").toast_type == "error"
        assert not service.executions
        adapter.fail_send = True
        result = await controller.open(source=SOURCE, session_key="key", owner_open_id="ou_owner",
                                       status_text="", metadata=None, initial_view="home")
        assert not result.success and controller._active[state.scope_key] == state.panel_id
        assert len(controller._states) == 1
        handle(controller, payload)
        assert "已经处理" in handle(controller, payload).toast
        await asyncio.gather(*controller._tasks)
        assert service.executions == ["new"]
        stale = {**payload, "nonce": "replay"}
        assert "页面已更新" in handle(controller, stale).toast
        assert service.executions == ["new"]
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_restart_invalidates_old_card_without_restoring_ui_state():
    old, adapter, service, state = await opened()
    payload = action_payload(state)
    await old.close()
    fresh = FeishuPanelController(adapter, Service())
    try:
        assert "失效" in handle(fresh, payload).toast
        assert not fresh._states
    finally:
        await fresh.close()


@pytest.mark.asyncio
async def test_catalog_singleflight_survives_waiter_cancel_and_closes_after_worker(tmp_path, monkeypatch):
    runner = SimpleNamespace(_resolve_profile_home_for_source=lambda source: tmp_path / (source.profile or "default"))
    service = HermesPanelControlService(runner)
    started, release = threading.Event(), threading.Event()
    calls = []
    def discover(cfg, *, refresh=False):
        calls.append(cfg)
        started.set()
        assert release.wait(5)
        return [{"slug": "provider", "models": ["model"], "is_current": False}]
    monkeypatch.setattr(service, "_discover_catalog", discover)
    first = asyncio.create_task(service._catalog(SOURCE, {"version": 1}))
    assert await asyncio.to_thread(started.wait, 5)
    second = asyncio.create_task(service._catalog(SOURCE, {"version": 1}))
    await asyncio.sleep(0)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    closing = asyncio.create_task(service.close())
    await asyncio.sleep(0)
    await closing
    assert service._discoveries and len(calls) == 1
    release.set()
    assert (await second)[0]["models"] == ["model"]
    await closing


def test_catalog_current_flags_are_local_and_alias_routes_remain_distinct():
    rows = [{"slug": "p", "name": "P", "models": ["m"], "is_current": False},
            {"slug": "other", "name": "Other", "models": ["n"], "is_current": False}]
    aliases = {"proxy-a": {"model": "m", "provider": "p", "base_url": "https://a.test/v1"},
               "proxy-b": {"model": "m", "provider": "p", "base_url": "https://b.test/v1"}}
    kwargs = dict(provider_rows=rows, aliases=aliases, effective_model="m", global_model="m", global_provider="p")
    providers, options = HermesPanelControlService._build_model_catalog(effective_provider="p", **kwargs)
    other, _ = HermesPanelControlService._build_model_catalog(effective_provider="other", **kwargs)
    assert {item["target"] for item in options} >= {"proxy-a", "proxy-b"}
    assert providers[0]["slug"] == "p" and other[0]["slug"] == "other"
    assert all(not row["is_current"] for row in rows)


@pytest.mark.asyncio
async def test_session_listing_uses_one_real_lane_scoped_query(tmp_path):
    from unittest.mock import AsyncMock, patch
    from hermes_state import SessionDB
    db = SessionDB(db_path=tmp_path / "state.db")
    key = "agent:main:feishu:dm:owner"
    db.create_session("current", "feishu", session_key=key)
    db.create_session("previous", "feishu", session_key=key)
    for index in range(210):
        db.create_session(f"foreign-{index}", "feishu", session_key=f"other-{index}")
    runner = SimpleNamespace(
        _session_db=SimpleNamespace(_db=db),
        async_session_store=SimpleNamespace(get_or_create_session=AsyncMock(return_value=SimpleNamespace(session_id="current"))))
    service = HermesPanelControlService(runner)
    with patch.object(db, "list_sessions_rich", wraps=db.list_sessions_rich) as query:
        rows = await service._session_rows(SOURCE, key)
    assert [row["id"] for row in rows] == ["previous"]
    query.assert_called_once()
    assert query.call_args.kwargs["session_key"] == key
    await service.close()
    db.close()


@pytest.mark.asyncio
async def test_panel_reads_channel_route_platform_display_and_route_capabilities(tmp_path, monkeypatch):
    from contextlib import asynccontextmanager
    from gateway.config import ChannelOverride
    from gateway.session_state import SessionState
    from gateway.control.settings import GatewaySettingsEndpoint
    from hermes_constants import resolve_reasoning_config
    cfg = {'model': 'global-model', 'agent': {'reasoning_effort': 'low'},
           'display': {'show_reasoning': False, 'platforms': {'feishu': {'show_reasoning': True}}}}
    state = SessionState()
    from gateway.run import GatewayRunner
    runner = object.__new__(GatewayRunner)
    runner.__dict__.update(config=object(),
        _resolve_profile_home_for_source=lambda source: tmp_path,
        _normalize_source_for_session_key=lambda source: source, _session_key_for_source=lambda source: 'key',
        _load_session_model_override=lambda key: None, _session_model_override=lambda key: state.conversation.model_override,
        _resolve_session_reasoning_config=lambda **kw: resolve_reasoning_config(cfg, kw['model']),
        _resolve_session_service_tier=lambda **kw: None, _session_state=lambda key: state,
        _peek_session_state=lambda key: state, _is_session_running=lambda key: False,
        _load_show_reasoning=lambda: False)
    monkeypatch.setattr('gateway.run._get_channel_override', lambda *a, **kw: ChannelOverride(model='channel-model', provider='proxy'))
    calls = []
    def fast(model, **route):
        calls.append((model, route))
        return {'service_tier': 'priority'} if route['provider'] == 'proxy' else None
    monkeypatch.setattr('hermes_cli.models.resolve_fast_mode_overrides', fast)
    service = HermesPanelControlService(runner)
    @asynccontextmanager
    async def scope(source):
        yield
    monkeypatch.setattr(service, '_scope', scope)
    monkeypatch.setattr(service, '_config', lambda source: cfg)
    actual = GatewaySettingsEndpoint(runner, SOURCE, 'key', cfg, 'session').read()
    snapshot = await service.snapshot(source=SOURCE, session_key='key', include_catalog=False,
                                      include_sessions=False, include_status=False)
    assert snapshot['effective_model'] == actual.model == 'channel-model'
    assert snapshot['effective_provider'] == actual.provider == 'proxy'
    assert snapshot['show_reasoning'] is True and snapshot['fast_supported'] is True
    assert snapshot['model_source'] == '频道配置'
    assert calls[-1] == ('channel-model', {'provider': 'proxy', 'base_url': ''})
    await service.close()


@pytest.mark.asyncio
async def test_catalog_refresh_and_expiry_preserve_singleflight(tmp_path, monkeypatch):
    runner = SimpleNamespace(_resolve_profile_home_for_source=lambda source: tmp_path)
    service = HermesPanelControlService(runner)
    inventory, calls = ['a'], []
    def discover(cfg, *, refresh=False):
        calls.append(refresh)
        return list(inventory)
    monkeypatch.setattr(service, '_discover_catalog', discover)
    assert await service._catalog(SOURCE, {}) == ['a']
    inventory.append('b')
    service.invalidate_catalog(SOURCE)
    results = await asyncio.gather(service._catalog(SOURCE, {}), service._catalog(SOURCE, {}))
    assert results == [['a', 'b'], ['a', 'b']] and calls == [False, True]
    inventory.append('c')
    monkeypatch.setattr(service, 'catalog_ttl', 0)
    assert await service._catalog(SOURCE, {}) == ['a', 'b', 'c']
    monkeypatch.setattr(service, 'catalog_ttl', 300)
    assert await service._catalog(SOURCE, {'display': {'show_reasoning': True}}) == ['a', 'b', 'c']
    assert calls == [False, True, True]
    (tmp_path / 'auth.json').write_text('{}')
    inventory.append('d')
    assert await service._catalog(SOURCE, {}) == ['a', 'b', 'c', 'd']
    assert calls[-1] is True
    await service.close()


@pytest.mark.asyncio
@pytest.mark.parametrize('target', ['new', 'resume'])
async def test_session_change_invalidates_inflight_view_results(target):
    class DelayedService(Service):
        async def snapshot(self, **options):
            result = await super().snapshot(**options)
            if options['include_sessions']:
                self.load_started.set()
                await self.release.wait()
                result['sessions'] = [{'id': 'obsolete', 'label': 'obsolete'}]
            return result
    service = DelayedService()
    controller, adapter, service, state = await opened(service)
    try:
        handle(controller, action(controller, state, target='sessions'))
        await service.load_started.wait()
        handle(controller, action(controller, state, op='exec', target=target, nonce='change'))
        await asyncio.gather(*controller._controls.values())
        service.release.set()
        await asyncio.gather(*controller._tasks)
        latest = controller._states[state.panel_id]
        assert 'sessions' not in latest.data['loaded_views']
        assert not latest.data['sessions']
    finally:
        await controller.close()


@pytest.mark.asyncio
async def test_refresh_starts_new_generation_while_old_model_query_is_still_running(tmp_path, monkeypatch):
    inventory_started, release = threading.Event(), threading.Event()
    catalog = HermesPanelControlService(SimpleNamespace(_resolve_profile_home_for_source=lambda source: tmp_path))
    calls = []
    def discover(cfg, *, refresh=False):
        calls.append(refresh)
        if not refresh:
            inventory_started.set()
            assert release.wait(5)
            return ['obsolete']
        return ['fresh']
    monkeypatch.setattr(catalog, '_discover_catalog', discover)
    class CatalogView(Service):
        invalidate_catalog = catalog.invalidate_catalog
        async def snapshot(self, **options):
            result = await super().snapshot(**options)
            if options['include_catalog']:
                models = await catalog._catalog(options['source'], {})
                result['model_options'] = [{'model': m, 'target': m, 'label': m, 'provider': 'p'} for m in models]
            return result
        async def close(self):
            await super().close()
            await catalog.close()
    controller, adapter, service, state = await opened(CatalogView())
    try:
        handle(controller, action(controller, state, target='model'))
        assert await asyncio.to_thread(inventory_started.wait, 5)
        old_load = controller._loads[(state.panel_id, 'model')]
        handle(controller, action(controller, state, op='refresh', nonce='refresh'))
        await controller._loads[(state.panel_id, 'model')]
        release.set()
        await old_load
        latest = controller._states[state.panel_id]
        assert latest.data['model_options'][0]['model'] == 'fresh'
        assert 'model' in latest.data['loaded_views'] and calls == [False, True]
    finally:
        release.set()
        await controller.close()
