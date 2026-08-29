"""Focused regressions for the stateful Feishu control panel."""

from __future__ import annotations

import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

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
