"""Gateway command mapping delegates one structured settings request."""
from types import SimpleNamespace
from unittest.mock import AsyncMock
import pytest
from gateway.config import Platform
from gateway.platforms.base import MessageEvent
from gateway.session import SessionSource
from gateway.slash_commands_model import GatewayModelCommandsMixin
from hermes_cli.commands import is_gateway_known_command, resolve_command
from hermes_cli.runtime_settings import RuntimeSettings, SettingsResult


@pytest.mark.asyncio
@pytest.mark.parametrize("command", ["/quick", "/mode quick", "/quick fast"])
async def test_gateway_alias_uses_shared_operation(monkeypatch, tmp_path, command):
    import gateway.run as facade
    import gateway.control.settings as settings
    cfg = {"mode_presets": {"quick": {"model": "endpoint-alias", "reasoning": "high"}}}
    monkeypatch.setattr(facade, "_load_gateway_config", lambda **kwargs: cfg)
    actual = RuntimeSettings("resolved", "custom:cpa", "https://cpa.test/v1", "codex_responses", "high")
    operation = AsyncMock(return_value=SettingsResult(actual, True))
    monkeypatch.setattr(settings, "apply_gateway_settings", operation)
    runner = SimpleNamespace(_resolve_profile_home_for_source=lambda source: tmp_path)
    event = MessageEvent(text=command, source=SessionSource(platform=Platform.FEISHU, user_id="owner", chat_id="chat"))
    result = await GatewayModelCommandsMixin._handle_mode_command(runner, event)
    selection = operation.call_args.args[2]
    assert selection.model_target == "endpoint-alias"
    assert selection.reasoning == "high"
    assert selection.service_tier == ("fast" if command.endswith(" fast") else "normal")
    assert result == SettingsResult(actual, True).text()
    for name in ("mode", "quick", "daily", "deep"):
        assert is_gateway_known_command(name) and resolve_command(name).name == "mode"
