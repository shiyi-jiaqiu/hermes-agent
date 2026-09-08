"""Exercise the real RPC, session DB and next-turn publication contract."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock
import threading
import pytest
from tui_gateway import server
from hermes_state import SessionDB
from hermes_cli.model_switch import ModelSwitchResult


@pytest.fixture
def mode_rpc(monkeypatch, tmp_path):
    sid = "mode-rpc-session"
    agent = SimpleNamespace(model="old", provider="old-provider", base_url="", api_mode="chat_completions",
                            reasoning_config={"enabled": True, "effort": "low"}, service_tier=None,
                            release_clients=Mock())
    ready = threading.Event()
    ready.set()
    session = dict(session_key=sid, running=False, agent=agent, agent_ready=ready,
                   history_lock=threading.Lock(), create_reasoning_override=agent.reasoning_config,
                   create_service_tier_override="", slash_worker=None)
    cfg = {"mode_presets": {"quick": {"model": "endpoint-alias", "reasoning": "high"}}}
    db = SessionDB(db_path=tmp_path / "state.db")
    @contextmanager
    def database(session):
        yield db
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_session_db", database)
    monkeypatch.setattr(server, "_load_cfg", lambda: cfg)
    monkeypatch.setattr(server, "_load_service_tier", lambda: None)
    monkeypatch.setattr(server, "_emit", Mock())
    monkeypatch.setattr("hermes_cli.model_switch.switch_model", lambda **kwargs: ModelSwitchResult(
        True, "target", "custom:cpa", api_key="secret", base_url="https://cpa.test/v1", api_mode="codex_responses"))
    yield sid, session, agent, db
    db.close()


@pytest.mark.parametrize("command", ["/quick", "/mode quick"])
def test_mode_rpc_persists_route_and_defers_client_construction(mode_rpc, command):
    sid, session, agent, db = mode_rpc
    response = server._methods["mode.apply"]("r", {"session_id": sid, "command": command})
    result = response["result"]
    assert result["applied"]
    assert result["actual"] == db.get_runtime_settings(sid)
    assert session["model_override"]["api_mode"] == "codex_responses"
    assert session["model_override"]["base_url"] == "https://cpa.test/v1"
    assert session["agent"] is None and session["lazy"]
    assert not session["agent_ready"].is_set()
    agent.release_clients.assert_called_once()
    resumed = server._stored_session_runtime_overrides(db.get_session(sid))
    assert resumed["model_override"]["api_mode"] == "codex_responses"
    assert resumed["reasoning_config_override"]["effort"] == "high"


def test_busy_rpc_never_changes_runtime_or_database(mode_rpc):
    sid, session, agent, db = mode_rpc
    session["running"] = True
    response = server._methods["mode.apply"]("r", {"session_id": sid, "command": "/quick"})
    assert not response["result"]["applied"]
    assert session["agent"] is agent and db.get_session(sid) is None
    agent.release_clients.assert_not_called()


def test_live_tuning_slash_uses_current_session_route(mode_rpc):
    sid, session, agent, db = mode_rpc
    text = server._live_slash_command_output(sid, session, "reasoning", "high")
    assert "high" in text
    actual = db.get_runtime_settings(sid)
    assert actual["model"] == "old" and actual["provider"] == "old-provider"
    assert actual["reasoning"] == "high"
    assert session["agent"] is None
