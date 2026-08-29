"""Focused regressions for the stateful Feishu control panel."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.control.panel_service import HermesPanelControlService
from gateway.config import PlatformConfig
from plugins.platforms.feishu.adapter import FeishuAdapter
from plugins.platforms.feishu.panel.controller import FeishuPanelController
from plugins.platforms.feishu.panel.state import PanelState
from plugins.platforms.feishu.panel.store import PanelStateStore


def _state(*, view: str = "home") -> PanelState:
    return PanelState(
        panel_id="p_test",
        message_id="om_test",
        app_id="cli_test",
        owner_open_id="ou_owner",
        chat_id="oc_chat",
        thread_id="",
        session_key="agent:main:feishu:dm:ou_owner",
        view=view,
        data={
            "effective_model": "test-model",
            "effective_provider": "test-provider",
            "global_model": "test-model",
            "global_provider": "test-provider",
            "effective_reasoning": "medium",
            "global_reasoning": "medium",
            "loaded_views": ["home"],
            "loading_views": [],
            "load_errors": {},
            "preset_options": [],
            "reasoning_options": [],
        },
    )


@pytest.mark.asyncio
async def test_send_control_panel_schedules_direct_initial_view() -> None:
    state = _state(view="model")
    state.data["loaded_views"] = ["home"]
    state.data["loading_views"] = ["model"]

    controller = SimpleNamespace(
        create_panel_state=AsyncMock(return_value=(state, None)),
        attach_message_id=MagicMock(return_value=True),
        schedule_view_load=MagicMock(return_value=True),
        discard=MagicMock(),
        _load_view_name=lambda view: view,
    )
    adapter = SimpleNamespace(
        gateway_runner=object(),
        _panel_controller=controller,
        _panel_store=MagicMock(),
        _app_id="cli_test",
        _admins={"ou_owner"},
        _send_interactive_card=AsyncMock(
            return_value=SimpleNamespace(
                success=True,
                message_id="om_test",
                error="",
            )
        ),
        update_interactive_message=AsyncMock(),
    )
    source = SimpleNamespace(
        chat_id="oc_chat",
        thread_id=None,
        chat_type="dm",
        user_id="ou_owner",
        profile="default",
    )

    result = await FeishuAdapter.send_control_panel(
        adapter,
        chat_id="oc_chat",
        status_text="",
        session_key=state.session_key,
        source=source,
        owner_open_id="ou_owner",
        initial_view="model",
    )

    assert result.success is True
    controller.attach_message_id.assert_called_once_with(state, "om_test")
    controller.schedule_view_load.assert_called_once_with("p_test", "model")


def test_unscheduled_navigation_becomes_retryable_error(tmp_path) -> None:
    store = PanelStateStore(tmp_path / "panel.db")

    class Adapter:
        gateway_runner = None

        @staticmethod
        def _is_interactive_operator_authorized(_open_id: str) -> bool:
            return True

        @staticmethod
        def _submit_on_loop(_loop, coro) -> bool:
            coro.close()
            return False

    controller = FeishuPanelController(Adapter(), store)
    state = _state()
    store.create_active(state)
    data = SimpleNamespace(
        event=SimpleNamespace(
            token="c-test",
            operator=SimpleNamespace(open_id="ou_owner"),
            context=SimpleNamespace(open_chat_id="oc_chat"),
        )
    )
    action = {
        "panel_action": True,
        "v": 1,
        "panel": state.panel_id,
        "rev": state.revision,
        "op": "nav",
        "target": "model",
        "nonce": "a_test",
    }

    result = controller.handle_sync(data, action, loop=object())
    latest = store.get(state.panel_id)

    assert result.toast_type == "error"
    assert latest is not None
    assert "model" not in latest.data["loading_views"]
    assert latest.data["load_errors"]["model"] == "页面加载任务无法调度，请重试"
    store.close()


@pytest.mark.asyncio
async def test_callback_update_falls_back_to_message_update() -> None:
    adapter = SimpleNamespace(
        update_interactive_card_after_callback=AsyncMock(
            return_value=SimpleNamespace(success=False, error="transient")
        ),
        update_interactive_message=AsyncMock(
            return_value=SimpleNamespace(success=True, error="")
        ),
    )
    store = MagicMock()
    controller = FeishuPanelController(adapter, store)
    state = _state()

    with patch(
        "plugins.platforms.feishu.panel.controller._CALLBACK_SETTLE_SECONDS",
        0.0,
    ):
        await controller._update_loaded_card(
            state,
            callback_token="c-test",
            callback_started_at=time.monotonic(),
        )

    adapter.update_interactive_card_after_callback.assert_awaited_once()
    adapter.update_interactive_message.assert_awaited_once()
    assert adapter.update_interactive_message.await_args.kwargs["message_id"] == "om_test"


def _panel_runner() -> SimpleNamespace:
    return SimpleNamespace(
        config=SimpleNamespace(multiplex_profiles=False),
        _session_model_overrides={},
        _resolve_session_reasoning_config=MagicMock(return_value={"effort": "medium"}),
        _peek_session_state=MagicMock(return_value=None),
        _resolve_session_service_tier=MagicMock(return_value=None),
        _is_session_running=MagicMock(return_value=False),
        _load_show_reasoning=MagicMock(return_value=False),
        async_session_store=SimpleNamespace(
            get_or_create_session=AsyncMock(
                return_value=SimpleNamespace(session_id="current-session")
            )
        ),
        _session_db=SimpleNamespace(_db=object()),
        _resume_row_visible=AsyncMock(),
    )


@pytest.mark.asyncio
async def test_session_snapshot_filters_in_memory_without_n_plus_one() -> None:
    runner = _panel_runner()
    service = HermesPanelControlService(runner)
    source = SimpleNamespace(platform=SimpleNamespace(value="feishu"))
    session_key = "agent:main:feishu:dm:ou_owner"
    rows = [
        {
            "id": "allowed",
            "source": "feishu",
            "session_key": session_key,
            "title": "Allowed",
        },
        {
            "id": "other-chat",
            "source": "feishu",
            "session_key": "agent:main:feishu:dm:someone_else",
            "title": "Other",
        },
        {
            "id": "other-platform",
            "source": "telegram",
            "session_key": session_key,
            "title": "Other platform",
        },
    ]

    with (
        patch.object(service, "_config", return_value={"model": {"default": "m"}}),
        patch("hermes_cli.session_listing.query_session_listing", return_value=rows),
    ):
        snapshot = await service.snapshot(
            source=source,
            session_key=session_key,
            include_catalog=False,
            include_sessions=True,
            include_status=False,
        )

    assert [row["id"] for row in snapshot["sessions"]] == ["allowed"]
    runner._resume_row_visible.assert_not_awaited()


@pytest.mark.asyncio
async def test_preset_exception_restores_all_session_overrides() -> None:
    runner = _panel_runner()
    runner._session_model_overrides = {
        "session-key": {"model": "partially-applied", "provider": "broken"}
    }
    runner._snapshot_session_model_override = MagicMock(
        return_value={
            "had_override": True,
            "override": {"model": "before", "provider": "provider-before"},
        }
    )
    runner._peek_session_state = MagicMock(
        return_value=SimpleNamespace(
            conversation=SimpleNamespace(reasoning_override={"effort": "low"})
        )
    )
    runner._resolve_session_service_tier = MagicMock(return_value="priority")
    runner._restore_session_model_override = MagicMock()
    runner._set_session_reasoning_override = MagicMock()
    runner._set_session_service_tier_override = MagicMock()
    runner._evict_cached_agent = MagicMock()
    runner.async_session_store.set_model_override = AsyncMock()
    runner._handle_mode_command = AsyncMock(side_effect=RuntimeError("provider failed"))
    service = HermesPanelControlService(runner)

    with pytest.raises(RuntimeError, match="provider failed"):
        await service.execute(
            source=SimpleNamespace(),
            session_key="session-key",
            target="preset",
            index=0,
            state_data={
                "preset_options": [
                    {
                        "name": "deep",
                        "label": "Deep",
                        "model": "after",
                        "reasoning": "high",
                        "fast_mode": False,
                    }
                ]
            },
        )

    runner._restore_session_model_override.assert_called_once_with(
        "session-key",
        {
            "had_override": True,
            "override": {"model": "before", "provider": "provider-before"},
        },
    )
    runner._set_session_reasoning_override.assert_called_once_with(
        "session-key", {"effort": "low"}
    )
    runner._set_session_service_tier_override.assert_called_once_with(
        "session-key", "priority"
    )
    runner.async_session_store.set_model_override.assert_awaited_once_with(
        "session-key", {"model": "before", "provider": "provider-before"}
    )
    runner._evict_cached_agent.assert_called_once_with("session-key")


@pytest.mark.asyncio
async def test_cached_provider_inventory_rebuilds_session_current_route() -> None:
    runner = _panel_runner()
    service = HermesPanelControlService(runner)
    source = SimpleNamespace(platform=SimpleNamespace(value="feishu"))
    inventory = [
        {
            "slug": "provider-a",
            "name": "Provider A",
            "is_current": True,
            "models": ["shared-model"],
        },
        {
            "slug": "provider-b",
            "name": "Provider B",
            "models": ["other-model"],
        },
    ]
    runner._session_model_overrides["session-b"] = {
        "model": "other-model",
        "provider": "provider-b",
    }

    with patch.object(
        service,
        "_config",
        return_value={
            "model": {"default": "shared-model", "provider": "provider-a"},
            "model_aliases": {},
        },
    ):
        snapshot = await service.snapshot(
            source=source,
            session_key="session-b",
            include_catalog=True,
            include_sessions=False,
            include_status=False,
            catalog_provider_rows=inventory,
        )

    current = [
        row["slug"] for row in snapshot["model_providers"] if row["is_current"]
    ]
    assert current == ["provider-b"]
    assert snapshot["_model_provider_inventory"][0]["is_current"] is False


@pytest.mark.asyncio
async def test_panel_mode_stays_selected_when_fast_is_enabled() -> None:
    runner = _panel_runner()
    runner._session_model_overrides["session-quick"] = {
        "model": "gpt-5.6-luna",
        "provider": "openai-codex",
    }
    runner._resolve_session_reasoning_config = MagicMock(
        return_value={"enabled": True, "effort": "xhigh"}
    )
    runner._resolve_session_service_tier = MagicMock(return_value="priority")
    service = HermesPanelControlService(runner)
    source = SimpleNamespace(platform=SimpleNamespace(value="feishu"))

    with patch.object(
        service,
        "_config",
        return_value={
            "model": {
                "default": "gpt-5.6-luna",
                "provider": "openai-codex",
            },
            "model_aliases": {
                "luna": {
                    "model": "gpt-5.6-luna",
                    "provider": "openai-codex",
                }
            },
            "mode_presets": {
                "quick": {
                    "model": "luna",
                    "reasoning": "xhigh",
                    "fast_mode": False,
                }
            },
        },
    ):
        snapshot = await service.snapshot(
            source=source,
            session_key="session-quick",
            include_catalog=False,
            include_sessions=False,
            include_status=False,
        )

    assert snapshot["fast_mode"] is True
    assert snapshot["current_preset"] == "quick"
