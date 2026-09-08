"""Gateway binding of the shared session settings operation."""
from __future__ import annotations

import asyncio
from hermes_cli.runtime_settings import (
    RuntimeSettings, SettingsRequest, SettingsResult, commit_settings, prepare_settings_async, reasoning_name,
)


class GatewaySettingsEndpoint:
    def __init__(self, runner, source, session_key, config, session_id):
        self.runner, self.source, self.key = runner, source, session_key
        self.config, self.session_id = config, session_id

    def read(self):
        self.runner._rehydrate_session_model_override(self.key)
        route = self.config.get("model") or {}
        if isinstance(route, str):
            route = {"default": route}
        override = self.runner._session_model_override(self.key) or {}
        if not override:
            from gateway.run import _get_channel_override
            channel = _get_channel_override(self.runner.config, self.source.platform, self.source.chat_id,
                thread_id=self.source.thread_id, parent_id=self.source.parent_chat_id)
            if channel:
                route = dict(route)
                if channel.model:
                    route["default"] = channel.model
                if channel.provider:
                    route.update(provider=channel.provider, base_url="", api_mode="")
        model = override.get("model") or route.get("default", "")
        rc = self.runner._resolve_session_reasoning_config(source=self.source, session_key=self.key, model=model)
        return RuntimeSettings(model, override.get("provider", route.get("provider", "")) or "",
                               override.get("base_url", route.get("base_url", "")) or "",
                               override.get("api_mode", route.get("api_mode", "")) or "",
                               reasoning_name(rc),
                               self.runner._resolve_session_service_tier(session_key=self.key) or "normal",
                               override.get("api_key") or "", override.get("request_overrides"), override.get("capabilities"),
                               self.runner._session_state(self.key).conversation.reasoning_override is None,
                               runtime_resolved=bool(override))

    def persist(self, settings):
        if self.runner._is_session_running(self.key):
            raise ValueError("Stop the running turn before changing settings")
        self.runner.session_store.set_runtime_settings(self.key, settings.persisted(), session_id=self.session_id)

    def publish(self, settings):
        state = self.runner._session_state(self.key)
        state.conversation.model_override = settings.route()
        state.conversation.one_turn_restore = None
        state.conversation.reasoning_override = None if settings.reasoning_inherited else settings.reasoning_config()
        self.runner._set_session_service_tier_override(
            self.key, None if settings.service_tier == "normal" else settings.service_tier)
        self.runner._evict_cached_agent(self.key)


async def apply_gateway_settings(runner, source, request: SettingsRequest, config: dict, *, resolver=None) -> SettingsResult:
    from gateway.run import _async_profile_runtime_scope
    async with _async_profile_runtime_scope(runner._resolve_profile_home_for_source(source)):
        normalized = await asyncio.to_thread(runner._normalize_source_for_session_key, source)
        key = runner._session_key_for_source(normalized)
        entry = await runner.async_session_store.get_or_create_session(normalized)
        endpoint = GatewaySettingsEndpoint(runner, normalized, key, config, entry.session_id)
        baseline = endpoint.read()
        try:
            target = await prepare_settings_async(baseline, request, config, resolver=resolver)
        except Exception as exc:
            from agent.redact import redact_sensitive_text
            return SettingsResult(endpoint.read(), False, redact_sensitive_text(str(exc), force=True, redact_url_credentials=True))
        return commit_settings(endpoint, baseline, target)


async def apply_gateway_global_tuning(runner, source, *, reasoning=None, service_tier=None):
    """Save a profile default, then reset the current session through the same commit path.

    Config and session DB are distinct stores: report a partial outcome explicitly.
    """
    from gateway.run import _async_profile_runtime_scope
    async with _async_profile_runtime_scope(runner._resolve_profile_home_for_source(source)):
        from gateway.run import _load_gateway_config
        from hermes_constants import parse_reasoning_effort
        if reasoning is not None and parse_reasoning_effort(reasoning) is None:
            return False, "Invalid reasoning level; settings unchanged"
        if service_tier is not None and service_tier not in {"fast", "priority", "normal", "auto", "cold"}:
            return False, "Invalid service tier; settings unchanged"
        field, value = ("reasoning_effort", reasoning) if reasoning is not None else ("service_tier", service_tier)
        if not runner._save_gateway_config_key("agent." + field, value):
            return False, "Configuration write failed; settings unchanged"
        cfg = _load_gateway_config(config_path=runner._resolve_profile_home_for_source(source) / "config.yaml")
        cfg = {**cfg, "agent": {**cfg.get("agent", {}), field: value}}
        selection = SettingsRequest(reset_reasoning=True) if reasoning is not None else SettingsRequest(service_tier=service_tier)
        result = await apply_gateway_settings(runner, source, selection, cfg)
        prefix = "Global default saved. " if result.applied else "Global default saved, but current session was not updated. "
        return result.applied, prefix + result.text()
