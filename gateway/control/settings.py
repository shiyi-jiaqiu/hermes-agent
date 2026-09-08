"""Gateway binding of the shared session settings operation."""
from __future__ import annotations

import asyncio
from hermes_cli.runtime_settings import (
    RuntimeSettings, SettingsRequest, SettingsResult, unchanged_result, needs_persistence,
    prepare_settings_async, reasoning_name, settings_error,
)


class GatewaySettingsEndpoint:
    def __init__(self, runner, source, session_key, config, session_id):
        self.runner, self.source, self.key = runner, source, session_key
        self.config, self.session_id = config, session_id

    def read(self):
        route = self.config.get("model") or {}
        if isinstance(route, str):
            route = {"default": route}
        override = self.runner._session_model_override(self.key) or {}
        self.model_source = "本会话覆盖" if override else "Profile 全局默认"
        if not override:
            from gateway.run import _get_channel_override
            channel = _get_channel_override(self.runner.config, self.source.platform, self.source.chat_id,
                thread_id=self.source.thread_id, parent_id=self.source.parent_chat_id)
            if channel:
                if channel.model or channel.provider:
                    self.model_source = "频道配置"
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
                               runtime_resolved=bool(override),
                               temporary=self.runner._session_state(self.key).conversation.one_turn_restore is not None)

    async def load(self):
        if self.runner._session_model_override(self.key) is None:
            task = self.runner._retain_background_task(asyncio.create_task(
                asyncio.to_thread(self.runner._load_session_model_override, self.key)))
            task.add_done_callback(lambda done: None if done.cancelled() else done.exception())
            loaded = await asyncio.shield(task)
            if loaded is not None:
                self.runner._publish_session_model_override(self.key, *loaded)
        return self.read()

    async def commit(self, baseline, target):
        """The task owns the session through DB completion, even if its UI waiter leaves."""
        async with self.runner._session_state(self.key).persistent.settings_lock:
            current = self.read()
            if not await self.runner.async_session_store.matches_session(self.key, self.session_id):
                return SettingsResult(current, False, "Session changed during settings resolution")
            if self.runner._is_session_running(self.key):
                return SettingsResult(current, False, "Stop the running turn before changing settings")
            unchanged = unchanged_result(current, baseline, target)
            if unchanged is not None:
                return unchanged
            try:
                if needs_persistence(current, target):
                    await self.runner.async_session_store.set_runtime_settings(
                        self.key, target.persisted(), session_id=self.session_id)
            except Exception as exc:
                return SettingsResult(self.read(), False, settings_error(exc))
            self.publish(target)
            return SettingsResult(target, True)

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
        async with runner._session_state(key).persistent.settings_lock:
            entry = await runner.async_session_store.get_or_create_session(normalized)
            endpoint = GatewaySettingsEndpoint(runner, normalized, key, config, entry.session_id)
            baseline = await endpoint.load()
        try:
            target = await prepare_settings_async(baseline, request, config, resolver=resolver,
                                                  retain=runner._retain_background_task)
        except Exception as exc:
            return SettingsResult(endpoint.read(), False, settings_error(exc))
        owner = runner._session_state(key).persistent
        if runner._draining:
            return SettingsResult(endpoint.read(), False, "Gateway is shutting down; settings unchanged")
        if owner.settings_commit is not None:
            return SettingsResult(endpoint.read(), False, "A settings commit is already in progress")
        task = asyncio.create_task(endpoint.commit(baseline, target))
        owner.settings_commit = task
        def completed(done):
            owner.settings_commit = None
            if not done.cancelled():
                done.exception()
        task.add_done_callback(completed)
        return await asyncio.shield(task)


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
