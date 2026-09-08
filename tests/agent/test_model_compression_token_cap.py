"""Absolute per-model caps must agree on real Agent initialization and model switches."""
import contextlib
import io

import pytest

from agent.context_compressor import ContextCompressor
from hermes_state import SessionDB
from run_agent import AIAgent


def test_agent_initialization_switch_and_rebuild_use_the_same_caps(monkeypatch, tmp_path):
    from hermes_cli import config
    cfg = {"compression": {"threshold": 0.5, "threshold_tokens": 40000,
                           "model_threshold_tokens": {"test": 20000, "test-long": 12000}},
           "model": {"default": "test-long", "provider": "custom", "context_length": 100000},
           "prompt_caching": {"cache_ttl": "5m"}}
    monkeypatch.setattr(config, "load_config", lambda: cfg)
    monkeypatch.setattr(config, "load_config_readonly", lambda: cfg)
    db = SessionDB(db_path=tmp_path / "state.db")
    agents = []
    def build(model):
        with contextlib.redirect_stdout(io.StringIO()):
            agent = AIAgent(model=model, provider="custom", api_key="test-key",
                            base_url="http://127.0.0.1:12345/v1", api_mode="chat_completions",
                            max_tokens=1024, enabled_toolsets=[], disabled_toolsets=[], quiet_mode=True,
                            skip_memory=True, skip_context_files=True, session_db=db, session_id="cap-test")
        agents.append(agent)
        return agent
    try:
        agent = build("test-long")
        assert agent.context_compressor.threshold_tokens_cap == agent.context_compressor.threshold_tokens == 12000
        for model, expected in (("test-short", 20000), ("other", 40000), ("test-long", 12000)):
            agent.switch_model(model, "custom", api_key="test-key", base_url="http://127.0.0.1:12345/v1",
                               api_mode="chat_completions")
            rebuilt = build(model)
            assert agent.context_compressor.threshold_tokens == rebuilt.context_compressor.threshold_tokens == expected
            assert agent._primary_runtime["compressor_threshold_tokens"] == expected
    finally:
        for agent in agents:
            agent.release_clients()
        db.close()


@pytest.mark.parametrize("value", [True, False, float("inf"), float("nan"), -1, 0, "invalid", "9000"])
def test_invalid_model_caps_fall_back_consistently_after_switch(value):
    compressor = ContextCompressor(model="bad", threshold_tokens_cap=20000,
                                   model_threshold_tokens={"bad": value}, quiet_mode=True)
    compressor.context_length = 100000
    expected = 9000 if value == "9000" else 20000
    assert compressor.threshold_tokens == expected
    compressor.update_model("other", 100000)
    assert compressor.threshold_tokens == 20000
    compressor.update_model("bad", 100000)
    assert compressor.threshold_tokens == expected
