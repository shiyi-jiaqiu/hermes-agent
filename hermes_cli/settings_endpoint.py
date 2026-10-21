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
                               cli.api_key or "", getattr(cli, "_settings_request_overrides", None),
                               getattr(cli, "_settings_capabilities", None),
                               getattr(cli, "_settings_reasoning_inherited", True),
                               temporary=getattr(cli, "_pending_one_turn_model_restore", None) is not None)

    def validate(self):
        cli = self.cli
        if getattr(cli, "_agent_running", False):
            raise ValueError("Stop the running turn before changing settings")
        if cli.session_id != self.session_id:
            raise ValueError("Session changed during settings resolution")

    def persist(self, settings):
        cli = self.cli
        from hermes_cli.gateway_tools_notify import gateway_store
        if gateway_store(cli) is not None:
            raise ValueError("Gateway settings must be committed by the session owner")
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


def sync_gateway_settings(cli, connection, snapshot=None):
    """Mirror the owner's selection without rebuilding the stand-in or sending settings back."""
    snapshot = snapshot if snapshot is not None else connection.call("session.resume", session_id=cli.session_id)
    state = connection.state(cli.session_id)
    state.adopt_snapshot(snapshot)
    info = state.info
    cli.model = info.get("model") or cli.model
    cli.provider = cli.requested_provider = info.get("provider") or cli.provider
    cli.api_mode = cli._explicit_api_mode = info.get("api_mode") or ""
    cli.api_key = cli._explicit_api_key = None
    cli.base_url = cli._explicit_base_url = ""
    from hermes_constants import parse_reasoning_effort
    level = info.get("reasoning_effort") or reasoning_name(cli.reasoning_config)
    cli.reasoning_config = parse_reasoning_effort(level)
    cli.service_tier = info.get("service_tier") or None
    cli._settings_request_overrides = cli._settings_capabilities = None
    cli._reasoning_user_set = False
    cli._pending_one_turn_model_restore = None
    state.service_tier_sent = cli.service_tier or "normal"
    from hermes_cli.gateway_agent import GatewayAgent
    if isinstance(cli.agent, GatewayAgent):
        cli.agent._apply_info(info)
        cli.agent._reasoning_config = cli.reasoning_config
        cli.agent._service_tier = cli.service_tier
        cli.agent._turn_route = cli.agent._turn_reasoning = None
    return CLISettingsEndpoint(cli).read()


def _owner_selection(cli, connection, session_id, request, snapshot):
    import uuid
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.gateway_mutations import model_refusal_text
    if request.model_target is not None:
        # Preserve the alias's endpoint and credential identity at the owner.
        operation, payload = "model", {"model": request.model_target}
        payload.update({key: value for key, value in (
            ("provider", request.provider), ("reasoning", request.reasoning),
            ("service_tier", request.service_tier)) if value is not None})
    elif request.reasoning is not None or request.reset_reasoning:
        operation, payload = "reasoning", {"effort": None if request.reset_reasoning else request.reasoning}
    else:
        connection.call("config.set", session_id=session_id, key="service_tier", value=request.service_tier)
        return {}
    fence = dict(session_id=session_id, operation=operation, expected_revision=snapshot["revision"],
                 expected_generation=snapshot["execution_generation"])
    result = connection.call("session.mutate", request_id=uuid.uuid4().hex, payload=payload, **fence)
    if result.get("status") != "confirmation_required":
        return result
    confirm = getattr(cli, "_confirm_owner_model_switch", None)
    if confirm is None:
        raise GatewayClientError(model_refusal_text(result))
    if not confirm(result):
        raise GatewayClientError("Model switch cancelled")
    result = connection.call("session.mutate", request_id=uuid.uuid4().hex,
                             payload={**payload, "confirm": result["confirm"]}, **fence)
    if result.get("status") == "confirmation_required":
        raise GatewayClientError(model_refusal_text(result, confirmed=True))
    return result


def _apply_owner_settings(cli, connection, request):
    import logging
    from hermes_cli.gateway_client import GatewayClientError
    from hermes_cli.runtime_settings import SettingsResult, settings_error
    endpoint = CLISettingsEndpoint(cli)
    with runtime_settings_lock(cli):
        try:
            endpoint.validate()
            snapshot = connection.call("session.resume", session_id=endpoint.session_id)
            result = _owner_selection(cli, connection, endpoint.session_id, request, snapshot)
            if result.get("status") == "deferred":
                return SettingsResult(endpoint.read(), False, "Selection queued for the next turn boundary")
        except (GatewayClientError, ValueError) as exc:
            return SettingsResult(endpoint.read(), False, settings_error(exc))
        # Committed: a failed refresh must not tell the caller the selection was rolled back.
        info = {**(snapshot.get("info") or {}), **{k: result[k] for k in ("model", "provider") if k in result}}
        if "effort" in result:
            info["reasoning_effort"] = result["effort"]
        if request.service_tier is not None:
            from agent.fast_mode import parse_service_tier
            info["service_tier"] = parse_service_tier(request.service_tier) or ""
        try:
            snapshot = connection.call("session.resume", session_id=endpoint.session_id)
        except GatewayClientError:
            logging.getLogger(__name__).warning("Settings committed; gateway snapshot refresh failed", exc_info=True)
            snapshot = {**snapshot, "info": info}
        if cli.session_id != endpoint.session_id:
            return SettingsResult(endpoint.read(), False, "Selection saved to the previous session")
        actual = sync_gateway_settings(cli, connection, snapshot)
        if request.reasoning is not None or request.reset_reasoning:
            from dataclasses import replace
            cli._settings_reasoning_inherited = request.reset_reasoning
            actual = replace(actual, reasoning_inherited=request.reset_reasoning)
        return SettingsResult(actual, True)


def runtime_settings_lock(cli):
    import threading
    return cli.__dict__.setdefault("_runtime_settings_lock", threading.RLock())


def apply_cli_settings(cli, request, config, *, resolver=None):
    from hermes_cli.runtime_settings import SettingsResult, commit_settings, prepare_settings, settings_error
    from hermes_cli.gateway_tools_notify import gateway_store
    store = gateway_store(cli)
    if store is not None:
        return _apply_owner_settings(cli, store.connection, request)
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
