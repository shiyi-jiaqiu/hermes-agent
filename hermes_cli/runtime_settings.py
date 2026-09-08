"""One operation for session route, reasoning and service-tier selections.

Resolution does I/O before commit. Commit compares the baseline, persists one
candidate, then publishes it for the next turn. UI adapters own neither rollback
nor live-client mutation; client construction remains at the turn boundary.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Callable, Protocol


@dataclass(frozen=True)
class RuntimeSettings:
    model: str
    provider: str = ""
    base_url: str = ""
    api_mode: str = ""
    reasoning: str = "medium"
    service_tier: str = "normal"
    api_key: str = field(default="", repr=False, compare=False)

    request_overrides: dict | None = field(default=None, repr=False, compare=False)
    capabilities: dict | None = field(default=None, repr=False, compare=False)
    reasoning_inherited: bool = False
    runtime_resolved: bool = field(default=True, compare=False, repr=False)

    def persisted(self) -> dict:
        return {k: getattr(self, k) for k in
                ("model", "provider", "base_url", "api_mode", "reasoning", "service_tier", "reasoning_inherited")}


    def route(self) -> dict:
        return {k: getattr(self, k) for k in
                ("model", "provider", "base_url", "api_mode", "api_key", "request_overrides", "capabilities")}

    def reasoning_config(self) -> dict:
        from hermes_constants import parse_reasoning_effort
        return parse_reasoning_effort(self.reasoning)


@dataclass(frozen=True)
class SettingsRequest:
    model_target: str | None = None
    provider: str | None = None
    reasoning: str | None = None
    service_tier: str | None = None
    reset_reasoning: bool = False


@dataclass(frozen=True)
class SettingsResult:
    actual: RuntimeSettings
    applied: bool
    error: str = ""

    def text(self) -> str:
        if not self.applied:
            return f"Settings unchanged: {self.error}"
        return (f"Settings saved for the next turn: {self.actual.model} ({self.actual.provider}) · "
                f"Reasoning {self.actual.reasoning} · {self.actual.service_tier}")


class SettingsEndpoint(Protocol):
    def read(self) -> RuntimeSettings: ...
    def persist(self, settings: RuntimeSettings) -> None: ...
    def publish(self, settings: RuntimeSettings) -> None: ...


def prepare_settings(current: RuntimeSettings, request: SettingsRequest, config: dict,
                     *, resolver: Callable | None = None) -> RuntimeSettings:
    """Resolve the original alias (including its endpoint), then validate all fields."""
    target = current
    if request.model_target is not None:
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.model_switch import switch_model
        resolved = (resolver or switch_model)(
            raw_input=request.model_target, explicit_provider=request.provider or "",
            current_model=current.model, current_provider=current.provider,
            current_base_url=current.base_url, current_api_key=current.api_key,
            is_global=False, user_providers=config.get("providers"),
            custom_providers=get_compatible_custom_providers(config))
        if not resolved.success:
            raise ValueError(resolved.error_message)
        target = replace(target, model=resolved.new_model, provider=resolved.target_provider or "",
                         base_url=resolved.base_url or "", api_mode=resolved.api_mode or "",
                         api_key=resolved.api_key or "", request_overrides=resolved.request_overrides,
                         capabilities=resolved.runtime_capabilities, runtime_resolved=True)
    elif not current.runtime_resolved:
        from hermes_cli.runtime_provider import resolve_runtime_provider
        runtime = resolve_runtime_provider(requested=current.provider or None,
                                           explicit_base_url=current.base_url or None,
                                           target_model=current.model or None)
        target = replace(current, model=runtime.get("model") or current.model,
                         provider=current.provider or runtime.get("provider") or "",
                         base_url=runtime.get("base_url") or "", api_mode=current.api_mode or runtime.get("api_mode") or "",
                         api_key=runtime.get("api_key") or "", request_overrides=runtime.get("request_overrides"),
                         capabilities=runtime.get("capabilities"), runtime_resolved=True)
    if request.reset_reasoning or (request.model_target is not None and current.reasoning_inherited):
        from hermes_constants import resolve_reasoning_config
        target = replace(target, reasoning=reasoning_name(resolve_reasoning_config(config, target.model)),
                         reasoning_inherited=True)
    if request.reasoning is not None:
        from hermes_constants import parse_reasoning_effort
        level = request.reasoning.lower()
        if level in {"provider", "provider-managed", "provider_managed", "auto"}:
            level = "none"
        if parse_reasoning_effort(level) is None:
            raise ValueError(f"Unknown reasoning effort: {request.reasoning}")
        target = replace(target, reasoning=level, reasoning_inherited=False)
    if request.service_tier is not None:
        tiers = {"fast": "priority", "on": "priority", "priority": "priority",
                 "normal": "normal", "off": "normal", "auto": "auto", "cold": "cold"}
        if request.service_tier not in tiers:
            raise ValueError(f"Unknown service tier: {request.service_tier}")
        target = replace(target, service_tier=tiers[request.service_tier])
    if target.service_tier == "priority":
        from hermes_cli.models import resolve_fast_mode_overrides
        if resolve_fast_mode_overrides(target.model, provider=target.provider, base_url=target.base_url) is None:
            raise ValueError("Fast mode is not available for the selected model")
    return target


def commit_settings(endpoint: SettingsEndpoint, baseline: RuntimeSettings,
                    target: RuntimeSettings) -> SettingsResult:
    """Called by the endpoint's owner, with no await between comparison and publish."""
    current = endpoint.read()
    if current != baseline:
        return SettingsResult(current, False, "Settings changed during resolution; retry the selection")
    try:
        endpoint.persist(target)
    except Exception as exc:
        from agent.redact import redact_sensitive_text
        return SettingsResult(endpoint.read(), False, redact_sensitive_text(str(exc), force=True, redact_url_credentials=True))
    endpoint.publish(target)  # local assignments only; all fallible work precedes this point
    return SettingsResult(target, True)


def apply_settings(endpoint: SettingsEndpoint, request: SettingsRequest, config: dict, *, resolver=None) -> SettingsResult:
    baseline = endpoint.read()
    try:
        target = prepare_settings(baseline, request, config, resolver=resolver)
    except Exception as exc:
        from agent.redact import redact_sensitive_text
        return SettingsResult(endpoint.read(), False, redact_sensitive_text(str(exc), force=True, redact_url_credentials=True))
    return commit_settings(endpoint, baseline, target)


def mode_request(config: dict, name: str, modifier: str = "") -> SettingsRequest:
    from hermes_cli.mode_presets import resolve_mode_preset
    preset = resolve_mode_preset(config, name)
    if preset is None:
        raise ValueError(f"Mode '{name}' is not configured")
    if modifier not in {"", "fast", "normal", "off"}:
        raise ValueError("Usage: /mode <name> [fast|normal]")
    fast = modifier == "fast" if modifier else preset.fast_mode
    return SettingsRequest(preset.model_target, reasoning=preset.reasoning,
                           service_tier="fast" if fast else "normal")


def reasoning_name(config: dict | None) -> str:
    return "none" if config and config.get("enabled") is False else (config or {}).get("effort", "medium")


def parse_mode_command(command: str) -> tuple[str, str]:
    import shlex
    words = shlex.split(command)
    if not words:
        raise ValueError("Usage: /mode <name> [fast|normal]")
    verb = words.pop(0).lstrip("/").lower()
    if verb != "mode":
        words.insert(0, verb)
    if not 1 <= len(words) <= 2:
        raise ValueError("Usage: /mode <name> [fast|normal]")
    return words[0].lower(), words[1].lower() if len(words) == 2 else ""


async def prepare_settings_async(current, request, config, *, resolver=None):
    """Cancellation stops publication, but joins the owned resolver thread first."""
    import asyncio
    task = asyncio.create_task(asyncio.to_thread(prepare_settings, current, request, config, resolver=resolver))
    try:
        return await asyncio.shield(task)
    except asyncio.CancelledError:
        await asyncio.gather(task, return_exceptions=True)
        raise


def stored_settings(row: dict | None) -> dict | None:
    """Decode settings without keeping a second, potentially stale route snapshot."""
    import json
    raw = (row or {}).get("model_config") or {}
    if isinstance(raw, str):
        raw = json.loads(raw)
    if not isinstance(raw, dict) or not raw.get("settings_override"):
        return None
    return {"model": (row or {}).get("model") or raw.get("model") or "",
            **{k: raw.get(k) or "" for k in ("provider", "base_url", "api_mode")},
            "reasoning": reasoning_name(raw.get("reasoning_config")),
            "service_tier": raw.get("service_tier") or "normal",
            "reasoning_inherited": bool(raw.get("reasoning_inherited"))}
