"""CLI storage/publication binding for session settings."""
from __future__ import annotations

from hermes_cli.runtime_settings import RuntimeSettings, reasoning_name


class CLISettingsEndpoint:
    def __init__(self, cli):
        self.cli = cli
        self.session_id = cli.session_id
        self.retired_agent = None

    def read(self):
        cli = self.cli
        return RuntimeSettings(cli.model, cli.provider or "", cli.base_url or "", cli.api_mode or "",
                               reasoning_name(cli.reasoning_config), cli.service_tier or "normal",
                               cli.api_key or "", cli._settings_request_overrides,
                               cli._settings_capabilities, cli._settings_reasoning_inherited,
                               temporary=cli._pending_one_turn_model_restore is not None)

    def validate(self):
        cli = self.cli
        if getattr(cli, "_agent_running", False):
            raise ValueError("Stop the running turn before changing settings")
        if cli.session_id != self.session_id:
            raise ValueError("Session changed during settings resolution")

    def persist(self, settings):
        cli = self.cli
        if cli._session_db is None:
            from hermes_state import SessionDB
            cli._session_db = SessionDB()
        cli._session_db.ensure_session(self.session_id, source="cli", model=cli.model)
        cli._session_db.update_runtime_settings(self.session_id, settings.persisted())

    def publish(self, settings):
        cli = self.cli
        self.retired_agent = cli.agent
        cli.model, cli.provider = settings.model, settings.provider
        cli.requested_provider = settings.provider
        cli.base_url = cli._explicit_base_url = settings.base_url
        cli.api_key = cli._explicit_api_key = settings.api_key
        cli.api_mode = cli._explicit_api_mode = settings.api_mode
        cli.reasoning_config = settings.reasoning_config()
        cli.service_tier = None if settings.service_tier == "normal" else settings.service_tier
        cli._settings_request_overrides = settings.request_overrides
        cli._settings_capabilities = settings.capabilities
        cli._settings_reasoning_inherited = settings.reasoning_inherited
        cli.agent = None
        cli._pending_one_turn_model_restore = None

    def release_retired(self):
        if self.retired_agent is not None:
            import logging
            try:
                self.retired_agent.release_clients()
            except Exception:
                logging.getLogger(__name__).warning("Failed to release retired agent clients", exc_info=True)


def runtime_settings_lock(cli):
    import threading
    return cli.__dict__.setdefault("_runtime_settings_lock", threading.RLock())


def apply_cli_settings(cli, request, config, *, resolver=None):
    from hermes_cli.runtime_settings import SettingsResult, commit_settings, prepare_settings, settings_error
    endpoint = CLISettingsEndpoint(cli)
    baseline = endpoint.read()
    try:
        candidate = prepare_settings(baseline, request, config, resolver=resolver)
    except Exception as exc:
        return SettingsResult(endpoint.read(), False, settings_error(exc))
    with runtime_settings_lock(cli):
        result = commit_settings(endpoint, baseline, candidate)
    endpoint.release_retired()
    return result
