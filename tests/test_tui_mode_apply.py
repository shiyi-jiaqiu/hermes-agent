from types import SimpleNamespace

import pytest

from tui_gateway import server


@pytest.fixture
def mode_rpc(monkeypatch):
    sid = "mode-rpc-session"
    agent = SimpleNamespace(
        model="old-model",
        provider="old-provider",
        base_url="",
        api_mode="codex_responses",
        reasoning_config={"enabled": True, "effort": "low"},
        service_tier="priority",
        request_overrides={"service_tier": "priority"},
    )
    session = {
        "session_key": sid,
        "running": False,
        "agent": agent,
        "model_override": {"model": "old-model", "provider": "old-provider"},
        "create_reasoning_override": {"enabled": True, "effort": "low"},
        "create_service_tier_override": "priority",
        "slash_worker": None,
    }
    server._sessions[sid] = session

    config = {
        "mode_presets": {
            "fast": {"model": "flash-cpa", "reasoning": "high", "fast_mode": False}
        },
        "model_aliases": {
            "flash-cpa": {"model": "target-model", "provider": "target-provider"}
        },
    }
    monkeypatch.setattr(server, "_load_cfg", lambda: config)
    monkeypatch.setattr(server, "_resolve_model", lambda: "old-model")
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_persist_live_session_runtime", lambda _session: None)
    monkeypatch.setattr(server, "_emit", lambda *args, **kwargs: None)
    monkeypatch.setattr(server, "_session_info", lambda *args, **kwargs: {})
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *args, **kwargs: None)

    def config_set(rid, params):
        key = params["key"]
        if key == "model":
            session["model_override"] = {
                "model": "target-model",
                "provider": "target-provider",
            }
            agent.model = "target-model"
            agent.provider = "target-provider"
        elif key == "reasoning":
            session["create_reasoning_override"] = {
                "enabled": True,
                "effort": "high",
            }
            agent.reasoning_config = session["create_reasoning_override"]
        elif key == "fast":
            session["create_service_tier_override"] = ""
            agent.service_tier = None
            agent.request_overrides = {}
        return {"jsonrpc": "2.0", "id": rid, "result": {"key": key, "value": params["value"]}}

    monkeypatch.setitem(server._methods, "config.set", config_set)
    try:
        yield sid, session, agent, config
    finally:
        server._sessions.pop(sid, None)


@pytest.mark.parametrize("command", ["/quick", "/mode quick"])
def test_mode_apply_routes_quick_to_panel_preset(mode_rpc, command):
    sid, session, agent, _config = mode_rpc

    response = server._methods["mode.apply"](
        "r1", {"session_id": sid, "command": command}
    )

    assert response["result"]["mode"] == "quick"
    assert "Mode `quick` applied" in response["result"]["output"]
    assert session["model_override"] == {
        "model": "target-model",
        "provider": "target-provider",
    }
    assert agent.reasoning_config["effort"] == "high"
    assert agent.service_tier is None


def test_mode_apply_restores_state_after_verification_failure(mode_rpc, monkeypatch):
    sid, session, agent, _config = mode_rpc
    original_override = dict(session["model_override"])
    original_reasoning = dict(session["create_reasoning_override"])
    original_tier = session["create_service_tier_override"]

    def wrong_model_set(rid, params):
        key = params["key"]
        if key == "model":
            session["model_override"] = {
                "model": "wrong-model",
                "provider": "wrong-provider",
            }
            agent.model = "wrong-model"
            agent.provider = "wrong-provider"
        elif key == "reasoning":
            session["create_reasoning_override"] = {"enabled": True, "effort": "high"}
            agent.reasoning_config = session["create_reasoning_override"]
        elif key == "fast":
            session["create_service_tier_override"] = ""
            agent.service_tier = None
        return {"jsonrpc": "2.0", "id": rid, "result": {"key": key}}

    monkeypatch.setitem(server._methods, "config.set", wrong_model_set)

    response = server._methods["mode.apply"](
        "r2", {"session_id": sid, "command": "/quick"}
    )

    output = response["result"]["output"]
    assert "verification failed" in output
    assert "previous settings restored" in output
    assert session["model_override"] == original_override
    assert session["create_reasoning_override"] == original_reasoning
    assert session["create_service_tier_override"] == original_tier
    assert agent.model == "old-model"
    assert agent.provider == "old-provider"
    assert agent.reasoning_config == original_reasoning
    assert agent.service_tier == "priority"
