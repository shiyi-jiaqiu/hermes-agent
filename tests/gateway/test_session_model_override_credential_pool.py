"""Session /model overrides must attach credential_pool for 402 rotation."""

from __future__ import annotations

from unittest.mock import MagicMock

from gateway.run import GatewayRunner, _credential_pool_for_provider


def test_fast_session_override_includes_credential_pool(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {
        "sess-1": {
            "model": "kimi-k2.7",
            "provider": "custom:hyper",
            "api_key": "sk-test",
            "base_url": "https://hyper.charm.land/v1",
            "api_mode": "chat_completions",
        },
    }
    fake_pool = object()

    monkeypatch.setattr(
        "gateway.run._resolve_gateway_model",
        lambda _uc=None: "default-model",
    )
    resolve_pool = MagicMock(return_value=fake_pool)
    monkeypatch.setattr("gateway.run._credential_pool_for_provider", resolve_pool)

    model, runtime = runner._resolve_session_agent_runtime(session_key="sess-1")

    assert model == "kimi-k2.7"
    assert runtime.get("credential_pool") is fake_pool
    resolve_pool.assert_called_once_with("custom:hyper", base_url="https://hyper.charm.land/v1", model="kimi-k2.7")




def test_restored_route_resolves_its_own_credentials(monkeypatch):
    runner = object.__new__(GatewayRunner)
    runner._session_model_overrides = {"session": {
        "model": "gemini", "provider": "custom:cpa", "base_url": "https://cpa.test/v1", "api_mode": "codex_responses"}}
    resolve = MagicMock(return_value={"provider": "custom:cpa", "api_key": "cpa-key", "base_url": "https://cpa.test/v1"})
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs_for_provider", resolve)
    monkeypatch.setattr("gateway.run._resolve_runtime_agent_kwargs", lambda: (_ for _ in ()).throw(AssertionError("ambient credentials used")))
    model, runtime = runner._resolve_session_agent_runtime(session_key="session")
    assert model == "gemini" and runtime["api_key"] == "cpa-key"
    assert runtime["api_mode"] == "codex_responses"
    resolve.assert_called_once_with("custom:cpa", base_url="https://cpa.test/v1", model="gemini")
