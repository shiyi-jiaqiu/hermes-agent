"""Panel presentation and trusted controls; settings use the shared business operation."""

from __future__ import annotations

import asyncio
import shlex
from contextlib import asynccontextmanager
import hashlib
import json
from dataclasses import dataclass
from typing import Any

from gateway.platforms.base import MessageEvent, MessageType



@dataclass(frozen=True)
class PanelControlResult:
    success: bool
    text: str


class HermesPanelControlService:
    def __init__(self, runner: Any):
        self.runner = runner
        self._catalogs = {}
        self._closed = False

    @asynccontextmanager
    async def _scope(self, source):
        from gateway.run import _async_profile_runtime_scope
        async with _async_profile_runtime_scope(self.runner._resolve_profile_home_for_source(source)):
            yield

    def _config(self, source):
        from gateway.run import _load_gateway_config
        path = self.runner._resolve_profile_home_for_source(source) / "config.yaml"
        return _load_gateway_config(config_path=path) or {}

    def _canonical_session_key(self, source, unused):
        normalized = self.runner._normalize_source_for_session_key(source)
        return self.runner._session_key_for_source(normalized)

    async def close(self):
        self._closed = True
        # Discovery tasks own their worker until completion, even if every view cancels.
        await asyncio.gather(*self._catalogs.values(), return_exceptions=True)
        self._catalogs.clear()

    async def _catalog(self, source, cfg):
        if self._closed:
            raise RuntimeError("Panel service is closed")
        home = str(self.runner._resolve_profile_home_for_source(source))
        version = hashlib.sha256(json.dumps(cfg, sort_keys=True, default=str).encode()).hexdigest()
        key = (home, version)
        task = self._catalogs.get(key)
        if task is None:
            for old, done in list(self._catalogs.items()):
                if old[0] == home and done.done():
                    del self._catalogs[old]
            task = asyncio.create_task(asyncio.to_thread(self._discover_catalog, cfg))
            self._catalogs[key] = task
        try:
            return await asyncio.shield(task)
        except Exception:
            if self._catalogs.get(key) is task:
                del self._catalogs[key]
            raise

    @staticmethod
    def _discover_catalog(cfg):
        from hermes_cli.config import get_compatible_custom_providers
        from hermes_cli.model_switch import list_authenticated_providers
        model = cfg.get("model") or {}
        if isinstance(model, str):
            model = {"default": model}
        policy = cfg.get("feishu_panel") or {}
        excluded = set((cfg.get("model_catalog") or {}).get("excluded_providers") or [])
        excluded.update(policy.get("hidden_providers") or [])
        rows = list_authenticated_providers(
            current_provider=model.get("provider", ""), current_base_url=model.get("base_url", ""),
            current_model=model.get("default", ""), user_providers=cfg.get("providers"),
            custom_providers=get_compatible_custom_providers(cfg), max_models=2000,
            probe_custom_providers=False, for_picker=True, excluded_providers=sorted(excluded))
        hidden = policy.get("hidden_model_prefixes") or {}
        return [{**row, "is_current": False,
                 "models": [m for m in row["models"] if not any(
                     m.startswith(prefix) for prefix in hidden.get(row["slug"], []))]}
                for row in rows if row["slug"] not in excluded]

    @staticmethod
    def _reasoning_value(config: Any) -> str:
        if not isinstance(config, dict):
            return "medium"
        if config.get("enabled") is False:
            return "none"
        return str(config.get("effort") or "medium")

    @staticmethod
    def _model_alias_target(spec: Any, fallback: str) -> tuple[str, str, str]:
        if isinstance(spec, dict):
            return (
                str(spec.get("model") or fallback),
                str(spec.get("provider") or ""),
                fallback,
            )
        value = str(spec or fallback)
        return value, "", fallback

    @staticmethod
    def _build_model_catalog(*, provider_rows, aliases, effective_model, effective_provider,
                             global_model, global_provider, policy=None):
        """Inventory is shared; selection flags and alias targets belong to this snapshot."""
        policy = policy or {}
        hidden_providers = set(policy.get("hidden_providers") or [])
        hidden_prefixes = policy.get("hidden_model_prefixes") or {}
        providers, options, seen = {}, [], set()

        def add(provider, model, target, label, name=""):
            if provider in hidden_providers or any(model.startswith(prefix) for prefix in hidden_prefixes.get(provider, [])):
                return
            identity = (provider, target)
            if not model or identity in seen:
                return
            seen.add(identity)
            row = providers.setdefault(provider, {
                "slug": provider, "name": name or provider or "Default", "model_indices": [],
                "is_current": provider == effective_provider,
            })
            row["model_indices"].append(len(options))
            options.append({"model": model, "provider": provider, "target": target, "label": label})

        for row in provider_rows:
            for model in row["models"]:
                add(row["slug"], model, model, model, row.get("name", ""))
        for alias, spec in aliases.items():
            if isinstance(spec, dict):
                model = spec.get("model") or ""
                provider = spec.get("provider") or global_provider
            else:
                model, provider = str(spec), global_provider
            # Keep each alias selectable: aliases for one model may have distinct endpoints/api modes.
            add(provider, model, alias, f"{alias} · {model}")
        for model, provider in ((effective_model, effective_provider), (global_model, global_provider)):
            add(provider, model, model, model)
        result = sorted(providers.values(), key=lambda row: (not row["is_current"], row["name"].lower()))
        for row in result:
            row["available_models"] = row["total_models"] = len(row["model_indices"])
        return result, options

    async def snapshot(
        self,
        *,
        source: Any,
        session_key: str,
        status_text: str = "",
        include_catalog: bool = True,
        include_sessions: bool = True,
        include_status: bool = True,
    ) -> dict[str, Any]:
        """Return only JSON-serializable, server-trusted panel data.

        The three ``include_*`` switches keep the first card local-only and let
        each view request just its own optional data. The default remains a
        complete snapshot for direct non-Panel callers.
        """
        async with self._scope(source):
            session_key = self._canonical_session_key(source, session_key)
            cfg = self._config(source)
            # Complete discovery before sampling the session's selected values.
            provider_rows = await self._catalog(source, cfg) if include_catalog else []
            raw_model_cfg = cfg.get("model")
            model_cfg: dict[str, Any] = (
                dict(raw_model_cfg) if isinstance(raw_model_cfg, dict) else {}
            )
            global_model = str(model_cfg.get("default") or "unknown")
            global_provider = str(model_cfg.get("provider") or "")
            self.runner._rehydrate_session_model_override(session_key)
            model_override = self.runner._session_model_override(session_key) or {}
            effective_model = str(model_override.get("model") or global_model)
            effective_provider = str(model_override.get("provider") or global_provider)
            effective_base_url = str(model_override.get("base_url", model_cfg.get("base_url")) or "")
            effective_api_mode = str(model_override.get("api_mode", model_cfg.get("api_mode")) or "")

            reasoning_cfg = self.runner._resolve_session_reasoning_config(
                source=source,
                session_key=session_key,
                model=effective_model,
            )
            effective_reasoning = self._reasoning_value(reasoning_cfg)
            raw_agent_cfg = cfg.get("agent")
            agent_cfg: dict[str, Any] = (
                dict(raw_agent_cfg) if isinstance(raw_agent_cfg, dict) else {}
            )
            raw_reasoning_overrides = agent_cfg.get("reasoning_overrides")
            reasoning_overrides: dict[str, Any] = (
                dict(raw_reasoning_overrides)
                if isinstance(raw_reasoning_overrides, dict)
                else {}
            )
            global_reasoning = str(
                reasoning_overrides.get(effective_model)
                or agent_cfg.get("reasoning_effort")
                or "medium"
            )
            reasoning_state = self.runner._peek_session_state(session_key)
            has_reasoning_override = bool(
                reasoning_state is not None
                and reasoning_state.conversation.reasoning_override is not None
            )
            fast_mode = self.runner._resolve_session_service_tier(
                session_key=session_key
            ) == "priority"
            try:
                from hermes_cli.models import model_supports_fast_mode

                fast_supported = bool(model_supports_fast_mode(effective_model))
            except Exception:
                fast_supported = False
            running = bool(self.runner._is_session_running(session_key))

            raw_aliases = cfg.get("model_aliases")
            aliases: dict[str, Any] = (
                dict(raw_aliases) if isinstance(raw_aliases, dict) else {}
            )
            raw_presets = cfg.get("mode_presets")
            presets: dict[str, Any] = (
                dict(raw_presets) if isinstance(raw_presets, dict) else {}
            )
            model_providers: list[dict[str, Any]] = []
            model_options: list[dict[str, str]] = []
            if include_catalog:
                model_providers, model_options = self._build_model_catalog(
                    provider_rows=list(provider_rows or []),
                    aliases=aliases,
                    effective_model=effective_model,
                    effective_provider=effective_provider,
                    global_model=global_model,
                    global_provider=global_provider,
                    policy=cfg.get("feishu_panel"),
                )

            preset_options: list[dict[str, Any]] = []
            label_map = {"quick": "⚡ Quick", "daily": "⚖ Daily", "deep": "🧠 Deep"}
            current_preset = ""
            alias_models = {
                str(alias): self._model_alias_target(spec, str(alias))[0]
                for alias, spec in aliases.items()
            }
            for name, spec in presets.items():
                if not isinstance(spec, dict):
                    continue
                preset_model_target = str(spec.get("model") or "")
                preset_model = alias_models.get(preset_model_target, preset_model_target)
                alias_spec = aliases.get(preset_model_target) or {}
                alias_spec = alias_spec if isinstance(alias_spec, dict) else {}
                preset_provider = str(alias_spec.get("provider") or "")
                preset_base_url = str(alias_spec.get("base_url") or "")
                preset_api_mode = str(alias_spec.get("api_mode") or "")
                preset_reasoning = str(spec.get("reasoning") or "")
                preset_fast_mode = bool(spec.get("fast_mode", False))
                preset_options.append(
                    {
                        "name": str(name),
                        "label": label_map.get(str(name).lower(), str(name).title()),
                        "model": preset_model,
                        "reasoning": preset_reasoning,
                        "fast_mode": preset_fast_mode,
                    }
                )
                # Fast is an orthogonal modifier shared by every mode. A mode
                # remains selected when Fast is toggled independently.
                if (
                    preset_model == effective_model
                    and preset_reasoning == effective_reasoning
                    and (not preset_provider or preset_provider == effective_provider)
                    and (not preset_base_url or preset_base_url == effective_base_url)
                    and (not preset_api_mode or preset_api_mode == effective_api_mode)
                ):
                    current_preset = str(name)

            session_rows: list[dict[str, Any]] = []
            if include_sessions:
                session_rows = await self._session_rows(source, session_key)

            if include_status and not status_text:
                status_event = MessageEvent(
                    text="/status",
                    message_type=MessageType.COMMAND,
                    source=source,
                    message_id="",
                )
                status_text = str(await self.runner._handle_status_command(status_event) or "")

            result: dict[str, Any] = {
                "effective_model": effective_model,
                "effective_provider": effective_provider,
                "global_model": global_model,
                "global_provider": global_provider,
                "model_source": "本会话覆盖" if model_override else "Profile 全局默认",
                "effective_reasoning": effective_reasoning,
                "global_reasoning": global_reasoning,
                "reasoning_source": "本会话覆盖" if has_reasoning_override else "Profile 全局默认",
                "value_source": (
                    "本会话覆盖"
                    if model_override or has_reasoning_override
                    else "Profile 全局默认"
                ),
                "show_reasoning": bool(self.runner._load_show_reasoning()),
                "fast_mode": fast_mode,
                "fast_supported": fast_supported,
                "fast_options": [
                    {"value": "fast", "label": "⚡ Fast", "is_current": fast_mode},
                    {"value": "normal", "label": "正常", "is_current": not fast_mode},
                ],
                "running": running,
                "current_preset": current_preset,
                "preset_options": preset_options[:8],
                "reasoning_options": [
                    {"value": value, "label": value}
                    for value in ("none", "minimal", "low", "medium", "high", "max")
                ],
            }
            if include_catalog:
                result["model_providers"] = model_providers
                result["model_options"] = model_options
            if include_sessions:
                result["sessions"] = session_rows
            if include_status:
                result["status_text"] = str(status_text)[:3000]
            return result

    async def _session_rows(self, source, session_key):
        from hermes_cli.session_listing import query_session_listing
        current = await self.runner.async_session_store.get_or_create_session(source)
        return await asyncio.to_thread(
            query_session_listing, self.runner._session_db._db,
            source=source.platform.value, session_key=session_key,
            current_session_id=current.session_id, include_all_sources=False,
            include_unnamed=True, search_query=None, limit=50, exclude_sources=["tool"])

    def _event(self, source, command):
        return MessageEvent(text=command, message_type=MessageType.COMMAND, source=source,
                            raw_message={"_hermes_panel_control": True}, message_id="")

    async def execute(self, *, source, session_key, target, index, state_data):
        from hermes_cli.runtime_settings import SettingsRequest, mode_request
        from .settings import apply_gateway_settings
        async with self._scope(source):
            session_key = self._canonical_session_key(source, session_key)
            command_for_target = {
                "preset": "mode", "model": "model", "fast": "fast", "reasoning": "reasoning",
                "global_reasoning": "reasoning", "reasoning_reset": "reasoning", "reasoning_display": "reasoning",
                "resume": "resume", "new": "new", "stop": "stop", "snapshot": "status",
            }
            command = command_for_target.get(target)
            if command is not None:
                denial = self.runner._check_slash_access(source, command)
                if denial:
                    return PanelControlResult(False, denial)
            cfg = self._config(source)
            selection_keys = {"preset": "preset_options", "model": "model_options",
                              "reasoning": "reasoning_options", "global_reasoning": "reasoning_options",
                              "fast": "fast_options", "resume": "sessions"}
            selected = None
            if target in selection_keys:
                options = state_data.get(selection_keys[target], [])
                if index is None or not 0 <= index < len(options):
                    return PanelControlResult(False, "Invalid selection; refresh the panel")
                selected = options[index]
            builders = {
                "preset": lambda: mode_request(cfg, selected["name"]),
                "model": lambda: SettingsRequest(selected["target"], selected["provider"] or None),
                "reasoning": lambda: SettingsRequest(reasoning=selected["value"]),
                "fast": lambda: SettingsRequest(service_tier=selected["value"]),
            }
            if target in builders:
                result = await apply_gateway_settings(self.runner, source, builders[target](), cfg)
                return PanelControlResult(result.applied, result.text())
            controls = {
                "snapshot": self._refresh, "stop": self._stop, "new": self._new,
                "resume": self._resume, "global_reasoning": self._global_reasoning,
                "reasoning_reset": self._reasoning_reset, "reasoning_display": self._reasoning_display,
            }
            handler = controls.get(target)
            if handler is None:
                return PanelControlResult(False, "Unsupported control")
            return await handler(source, session_key, selected, index, cfg)

    async def _refresh(self, *args):
        return PanelControlResult(True, "状态已刷新")

    async def _stop(self, source, key, *args):
        text = await self.runner._handle_stop_command(self._event(source, "/stop"))
        return PanelControlResult(True, text or "停止请求已发送")

    async def _new(self, source, key, *args):
        before = await self.runner.async_session_store.get_or_create_session(source)
        previous_id = before.session_id
        text = await self.runner._handle_reset_command(self._event(source, "/new"))
        after = await self.runner.async_session_store.get_or_create_session(source)
        return PanelControlResult(after.session_id != previous_id, text or "新会话已创建")

    async def _resume(self, source, key, selected, *args):
        text = await self.runner._handle_resume_command(self._event(source, f"/resume {shlex.quote(selected['id'])}"))
        after = await self.runner.async_session_store.get_or_create_session(source)
        return PanelControlResult(after.session_id == selected["id"], text or "会话已恢复")

    async def _global_reasoning(self, source, key, selected, index, cfg):
        from .settings import apply_gateway_global_tuning
        success, text = await apply_gateway_global_tuning(self.runner, source, reasoning=selected["value"])
        return PanelControlResult(success, text)

    async def _reasoning_reset(self, source, key, selected, index, cfg):
        from hermes_cli.runtime_settings import SettingsRequest
        from .settings import apply_gateway_settings
        result = await apply_gateway_settings(self.runner, source, SettingsRequest(reset_reasoning=True), cfg)
        return PanelControlResult(result.applied, result.text())

    async def _reasoning_display(self, source, key, selected, index, cfg):
        from gateway.run import _platform_config_key
        if index not in (0, 1):
            return PanelControlResult(False, "Invalid display setting")
        show = index == 0
        success = self.runner._save_gateway_config_key(
            f"display.platforms.{_platform_config_key(source.platform)}.show_reasoning", show)
        if success:
            self.runner._show_reasoning = show
        return PanelControlResult(success, "显示设置已更新" if success else "Configuration write failed")
