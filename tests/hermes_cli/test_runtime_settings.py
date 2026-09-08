"""Business invariants shared by all settings entry points."""
from dataclasses import replace
import json

import pytest

from hermes_cli.model_switch import ModelSwitchResult
from hermes_cli.runtime_settings import (
    RuntimeSettings, SettingsRequest, apply_settings, commit_settings, mode_request,
    parse_mode_command, prepare_settings,
)
from hermes_state import SessionDB


class Endpoint:
    def __init__(self, db, settings):
        self.db, self.settings = db, settings
        self.published = 0

    def read(self):
        return self.settings

    def validate(self):
        pass

    def persist(self, settings):
        self.db.update_runtime_settings("session", settings.persisted())

    def publish(self, settings):
        self.settings = settings
        self.published += 1


@pytest.fixture
def endpoint(tmp_path):
    db = SessionDB(db_path=tmp_path / "state.db")
    db.create_session("session", source="cli", model="old", model_config={"_branched_from": "parent"})
    ep = Endpoint(db, RuntimeSettings("old", "old-provider", "https://old.test/v1", "chat_completions"))
    yield ep
    db.close()


def test_complete_route_and_tuning_commit_together_and_preserve_metadata(endpoint):
    baseline = endpoint.read()
    target = RuntimeSettings("gemini", "cpa", "https://new.test/v1", "codex_responses", "high", "normal", "secret-key")
    result = commit_settings(endpoint, baseline, target)
    row = endpoint.db.get_session("session")
    stored = json.loads(row["model_config"])
    assert result.applied and result.actual == target
    assert row["model"] == "gemini"
    assert endpoint.db.get_runtime_settings("session") == target.persisted()
    assert "runtime_settings" not in stored
    assert stored["gateway_runtime"] == {k: getattr(target, k) for k in ("model", "provider", "base_url", "api_mode")}
    assert stored["reasoning_config"]["effort"] == "high"
    assert stored["service_tier"] == "normal"
    assert stored["_branched_from"] == "parent"
    assert "secret-key" not in row["model_config"]


def test_persistence_failure_never_publishes_partial_settings(endpoint):
    baseline = endpoint.read()
    endpoint.db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER deny_settings BEFORE UPDATE ON sessions BEGIN SELECT RAISE(ABORT, 'disk write rejected'); END"))
    result = commit_settings(endpoint, baseline, replace(baseline, model="new", reasoning="high"))
    assert not result.applied
    assert "disk write rejected" in result.error
    assert result.actual == baseline == endpoint.read()
    assert endpoint.published == 0
    assert endpoint.db.get_session("session")["model"] == "old"


def test_slow_resolution_cannot_overwrite_a_newer_selection(endpoint):
    before = endpoint.read()
    latest = replace(before, reasoning="low")
    assert commit_settings(endpoint, before, latest).applied
    result = commit_settings(endpoint, before, replace(before, model="obsolete"))
    assert not result.applied
    assert result.actual == latest
    assert endpoint.published == 1


def test_mode_passes_original_alias_to_resolver_including_endpoint_and_protocol():
    cfg = {"mode_presets": {"quick": {"model": "proxy-alias", "reasoning": "high"}}}
    request = mode_request(cfg, "quick")
    calls = []
    def resolve(**kwargs):
        calls.append(kwargs)
        return ModelSwitchResult(True, "gemini", "cpa", api_key="test-key", base_url="https://cpa.test/v1",
                                 api_mode="codex_responses", runtime_capabilities={"reasoning": True})
    actual = prepare_settings(RuntimeSettings("old"), request, cfg, resolver=resolve)
    assert calls[0]["raw_input"] == "proxy-alias"
    assert actual.persisted() == {"model": "gemini", "provider": "cpa", "base_url": "https://cpa.test/v1",
                                  "api_mode": "codex_responses", "reasoning": "high", "service_tier": "normal", "reasoning_inherited": False}
    assert actual.route()["capabilities"] == {"reasoning": True}


@pytest.mark.parametrize("command", ["/quick", "/mode quick", "quick"])
def test_command_aliases_parse_the_same_request(command):
    assert parse_mode_command(command) == ("quick", "")


@pytest.mark.parametrize("selection", [SettingsRequest(reasoning="bad"), SettingsRequest(service_tier="bad")])
def test_invalid_request_does_not_write(endpoint, selection):
    before = endpoint.read()
    result = apply_settings(endpoint, selection, {})
    assert not result.applied and result.actual == before and endpoint.published == 0


def test_cli_resume_restores_same_model_with_different_endpoint_and_tuning(endpoint, monkeypatch):
    from types import SimpleNamespace
    from hermes_cli.cli_model_switch_mixin import CLIModelSwitchMixin
    saved = RuntimeSettings("old", "old-provider", "https://new.test/v1", "codex_responses", "high", "priority")
    assert commit_settings(endpoint, endpoint.read(), saved).applied
    calls = []
    def credentials(**kwargs):
        calls.append(kwargs)
        return {"api_key": "new-key"}
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", credentials)
    cli = SimpleNamespace(model="old", provider="old-provider", base_url="https://old.test/v1",
                          api_mode="chat_completions", api_key="ambient-key", agent=object(), session_id="session")
    CLIModelSwitchMixin._restore_session_model(cli, endpoint.db.get_session("session"))
    assert (cli.model, cli.base_url, cli.api_mode, cli.reasoning_config["effort"], cli.service_tier) == (
        "old", "https://new.test/v1", "codex_responses", "high", "priority")
    assert cli.agent is None and cli.api_key == "new-key"
    assert calls == [{"requested": "old-provider", "explicit_base_url": "https://new.test/v1", "target_model": "old"}]
    endpoint.db.update_session_model("session", "newer", provider="newer-provider")
    assert endpoint.db.get_runtime_settings("session")["model"] == "newer"
    assert endpoint.db.get_runtime_settings("session")["provider"] == "newer-provider"


def test_initial_tuning_resolves_full_route_before_commit(endpoint, monkeypatch):
    endpoint.settings = RuntimeSettings("old", "custom:proxy", "https://proxy.test/v1", runtime_resolved=False)
    calls = []
    def resolve(**kwargs):
        calls.append(kwargs)
        return dict(provider="custom:proxy", base_url="https://proxy.test/v1", api_key="new-key", api_mode="codex_responses")
    monkeypatch.setattr("hermes_cli.runtime_provider.resolve_runtime_provider", resolve)
    result = apply_settings(endpoint, SettingsRequest(reasoning="high"), {})
    assert result.applied and result.actual.api_mode == "codex_responses"
    assert endpoint.db.get_runtime_settings("session")["base_url"] == "https://proxy.test/v1"
    assert calls[0]["explicit_base_url"] == "https://proxy.test/v1"
    assert calls[0]["target_model"] == "old"


@pytest.mark.parametrize('reasoning, expected', [(None, 'high'), ('low', 'low'), ('invalid', 'high')])
def test_cli_initial_reasoning_preserves_source_when_switching_model(monkeypatch, reasoning, expected):
    from types import SimpleNamespace
    import cli as cli_module
    from hermes_cli.settings_endpoint import CLISettingsEndpoint
    cfg = {'agent': {'reasoning_overrides': {'model-a': 'low', 'model-b': 'high'}}}
    monkeypatch.setattr(cli_module, 'CLI_CONFIG', cfg)
    cli = SimpleNamespace(model='model-a', provider='p', base_url='', api_mode='', api_key='', session_id='s')
    cli_module.HermesCLI._init_prompt_and_reasoning(cli, reasoning)
    target = prepare_settings(CLISettingsEndpoint(cli).read(), SettingsRequest(model_target='model-b'), cfg,
                              resolver=lambda **kw: ModelSwitchResult(True, 'model-b', 'p'))
    assert target.reasoning == expected
    assert target.reasoning_inherited is (reasoning != 'low')


def test_unchanged_settings_skip_storage_but_runtime_and_source_changes_publish(endpoint):
    initial = endpoint.read()
    endpoint.db._execute_write(lambda conn: conn.execute(
        "CREATE TRIGGER deny_noop BEFORE UPDATE ON sessions BEGIN SELECT RAISE(ABORT, 'unnecessary write'); END"))
    result = commit_settings(endpoint, initial, replace(initial))
    assert result.applied and endpoint.published == 0
    for changes in ({'api_key': 'rotated'}, {'capabilities': {'reasoning': True}},
                    {'request_overrides': {'x': 1}}, {'reasoning_inherited': True}):
        if 'reasoning_inherited' in changes:
            endpoint.db._execute_write(lambda conn: conn.execute('DROP TRIGGER deny_noop'))
        before = endpoint.read()
        result = commit_settings(endpoint, before, replace(before, **changes))
        assert result.applied
    assert endpoint.published == 4


def test_cli_noop_keeps_agent_and_real_replacement_only_releases_clients(endpoint, monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import Mock
    import cli as cli_module
    from hermes_cli.settings_endpoint import apply_cli_settings
    cfg = {'agent': {}}
    monkeypatch.setattr(cli_module, 'CLI_CONFIG', cfg)
    retired = SimpleNamespace(release_clients=Mock(), cleanup=Mock())
    current = endpoint.read()
    cli = SimpleNamespace(**current.route(), session_id='session', agent=retired, _session_db=endpoint.db,
                          _agent_running=False)
    cli_module.HermesCLI._init_prompt_and_reasoning(cli, None)
    result = apply_cli_settings(cli, SettingsRequest(), cfg)
    assert result.applied and not result.changed and cli.agent is retired
    retired.release_clients.assert_not_called()
    result = apply_cli_settings(cli, SettingsRequest(model_target='new'), cfg,
                               resolver=lambda **kw: ModelSwitchResult(True, 'new', 'p'))
    assert result.applied and result.changed and cli.agent is None
    retired.release_clients.assert_called_once_with()
    retired.cleanup.assert_not_called()
    cli.agent = retired
    cli._pending_one_turn_model_restore = {'model': 'previous'}
    result = apply_cli_settings(cli, SettingsRequest(), cfg)
    assert result.applied and result.changed and cli._pending_one_turn_model_restore is None
    assert retired.release_clients.call_count == 2
