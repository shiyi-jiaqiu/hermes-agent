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
    def discover(cfg):
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
    assert not closing.done() and len(calls) == 1
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
