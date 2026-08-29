"""Gateway mode aliases: /quick, /daily, /deep, and /mode."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import gateway.run as gateway_run
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from hermes_cli.commands import is_gateway_known_command, resolve_command


PRESETS = {
    "quick": {"model": "luna", "reasoning": "xhigh", "fast_mode": False},
    "daily": {"model": "luna", "reasoning": "max", "fast_mode": False},
    "deep": {"model": "sol", "reasoning": "medium", "fast_mode": False},
}
ALIASES = {
    "luna": {"model": "gpt-5.6-luna", "provider": "openai-codex"},
    "sol": {"model": "gpt-5.6-sol", "provider": "openai-codex"},
}


def _event(text: str) -> MessageEvent:
    return MessageEvent(
        text=text,
        source=SessionSource(
            platform=Platform.FEISHU,
            user_id="ou_owner",
            chat_id="oc_chat",
            user_name="Owner",
        ),
    )


def _runner(monkeypatch: pytest.MonkeyPatch):
    runner = object.__new__(gateway_run.GatewayRunner)
    runner.config = SimpleNamespace(multiplex_profiles=False)
    runner._session_model_overrides = {
        "session-key": {"model": "before", "provider": "before-provider"}
    }
    runner._reasoning_value = {"enabled": True, "effort": "low"}
    runner._tier_value = "priority"
    runner.session_store = object()
    runner._async_session_store = SimpleNamespace(
        _store=runner.session_store,
        set_model_override=AsyncMock(),
    )
    runner._normalize_source_for_session_key = lambda source: source
    runner._session_key_for_source = lambda source: "session-key"
    runner._snapshot_session_model_override = MagicMock(
        side_effect=lambda key: {
            "had_override": True,
            "override": dict(runner._session_model_overrides[key]),
        }
    )
    runner._peek_session_state = lambda key: SimpleNamespace(
        conversation=SimpleNamespace(reasoning_override=dict(runner._reasoning_value))
    )
    runner._resolve_session_service_tier = lambda **kwargs: runner._tier_value
    runner._resolve_session_reasoning_config = lambda **kwargs: dict(
        runner._reasoning_value
    )
    runner._evict_cached_agent = MagicMock()

    def restore_model(key, snapshot):
        runner._session_model_overrides[key] = dict(snapshot["override"])

    runner._restore_session_model_override = MagicMock(side_effect=restore_model)
    runner._set_session_reasoning_override = MagicMock(
        side_effect=lambda key, value: setattr(
            runner, "_reasoning_value", dict(value) if value is not None else None
        )
    )
    runner._set_session_service_tier_override = MagicMock(
        side_effect=lambda key, value, **kwargs: setattr(runner, "_tier_value", value)
    )

    async def model_handler(event):
        target = event.get_command_args().split()[0]
        spec = ALIASES[target]
        runner._session_model_overrides["session-key"] = {
            "model": spec["model"],
            "provider": spec["provider"],
        }
        return f"switched to {spec['model']}"

    async def reasoning_handler(event):
        effort = event.get_command_args().split()[0]
        runner._reasoning_value = {"enabled": True, "effort": effort}
        return f"reasoning {effort}"

    async def fast_handler(event):
        value = event.get_command_args().split()[0]
        runner._tier_value = "priority" if value == "fast" else None
        return f"fast {value}"

    runner._handle_model_command = AsyncMock(side_effect=model_handler)
    runner._handle_reasoning_command = AsyncMock(side_effect=reasoning_handler)
    runner._handle_fast_command = AsyncMock(side_effect=fast_handler)
    monkeypatch.setattr(
        gateway_run,
        "_load_gateway_config",
        lambda **kwargs: {"mode_presets": PRESETS, "model_aliases": ALIASES},
    )
    return runner


def test_mode_and_direct_aliases_are_registered() -> None:
    for name in ("mode", "quick", "daily", "deep"):
        assert is_gateway_known_command(name)
        assert resolve_command(name).name == "mode"


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("name", "model", "reasoning"),
    (
        ("quick", "gpt-5.6-luna", "xhigh"),
        ("daily", "gpt-5.6-luna", "max"),
        ("deep", "gpt-5.6-sol", "medium"),
    ),
)
async def test_direct_mode_alias_defaults_fast_off(
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    model: str,
    reasoning: str,
) -> None:
    runner = _runner(monkeypatch)

    result = await runner._handle_mode_command(_event(f"/{name}"))

    assert "Fast `off`" in result
    assert runner._session_model_overrides["session-key"]["model"] == model
    assert runner._reasoning_value["effort"] == reasoning
    assert runner._tier_value is None


@pytest.mark.asyncio
@pytest.mark.parametrize("name", ("quick", "daily", "deep"))
async def test_fast_modifier_applies_to_every_mode(
    monkeypatch: pytest.MonkeyPatch, name: str
) -> None:
    runner = _runner(monkeypatch)

    result = await runner._handle_mode_command(_event(f"/{name} fast"))

    assert "Fast `on`" in result
    assert runner._tier_value == "priority"


@pytest.mark.asyncio
async def test_mode_command_form_is_equivalent(monkeypatch: pytest.MonkeyPatch) -> None:
    runner = _runner(monkeypatch)

    result = await runner._handle_mode_command(_event("/mode daily fast"))

    assert "Mode `daily` applied" in result
    assert runner._session_model_overrides["session-key"]["model"] == "gpt-5.6-luna"
    assert runner._reasoning_value["effort"] == "max"
    assert runner._tier_value == "priority"


@pytest.mark.asyncio
async def test_mode_failure_restores_previous_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _runner(monkeypatch)
    runner._handle_reasoning_command = AsyncMock(
        side_effect=RuntimeError("reasoning backend failed")
    )

    with pytest.raises(RuntimeError, match="reasoning backend failed"):
        await runner._handle_mode_command(_event("/deep fast"))

    assert runner._session_model_overrides["session-key"] == {
        "model": "before",
        "provider": "before-provider",
    }
    assert runner._reasoning_value == {"enabled": True, "effort": "low"}
    assert runner._tier_value == "priority"
    runner.async_session_store.set_model_override.assert_awaited_once_with(
        "session-key", {"model": "before", "provider": "before-provider"}
    )
