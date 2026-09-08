"""Panel presentation and trusted controls; settings use the shared business operation."""

from __future__ import annotations

import asyncio
import shlex
from contextlib import asynccontextmanager
import hashlib
import json
import time
from dataclasses import dataclass
from typing import Any

from gateway.platforms.base import MessageEvent, MessageType



@dataclass(frozen=True)
class PanelControlResult:
    success: bool
    text: str


class HermesPanelControlService:
    catalog_ttl = 300.0

    def __init__(self, runner: Any):
        self.runner = runner
        self._catalogs = {}
        self._discoveries = set()
        self._refresh_catalogs = set()
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

    def _canonical_session_key(self, source):
        normalized = self.runner._normalize_source_for_session_key(source)
        return self.runner._session_key_for_source(normalized)

    async def close(self):
        self._closed = True
        # Workers only own their input config and network discovery. Keep references until
        # completion, but closing a UI does not wait for optional provider inventory.
        self._catalogs.clear()
        self._refresh_catalogs.clear()

    def invalidate_catalog(self, source, *, refresh=True):
        home = str(self.runner._resolve_profile_home_for_source(source))
        if refresh:
            self._refresh_catalogs.add(home)
        for key in list(self._catalogs):
            if key[0] == home:
                del self._catalogs[key]

    async def _catalog(self, source, cfg):
        if self._closed:
            raise RuntimeError("Panel service is closed")
        path = self.runner._resolve_profile_home_for_source(source)
        home = str(path)
        relevant = {k: cfg.get(k) for k in ("model", "providers", "custom_providers", "model_catalog", "feishu_panel")}
        version = hashlib.sha256(json.dumps(relevant, sort_keys=True, default=str).encode()).hexdigest()
        auth_version = []
        for name in (".env", "auth.json"):
            try:
                stat = (path / name).stat()
                auth_version.append((stat.st_mtime_ns, stat.st_size, stat.st_ino))
            except FileNotFoundError:
                auth_version.append(None)
        key = (home, version, tuple(auth_version))
        cached = self._catalogs.get(key)
        if cached is None or (cached[0].done() and time.monotonic() - cached[1] >= self.catalog_ttl):
            force_refresh = home in self._refresh_catalogs or any(k[0] == home for k in self._catalogs)
            self.invalidate_catalog(source, refresh=False)
            self._refresh_catalogs.discard(home)
            task = asyncio.create_task(asyncio.to_thread(self._discover_catalog, cfg, refresh=force_refresh))
            self._discoveries.add(task)
            cached = [task, float("inf")]
            self._catalogs[key] = cached
            def completed(done):
                self._discoveries.discard(done)
                cached[1] = time.monotonic()
                if not done.cancelled():
                    done.exception()
            task.add_done_callback(completed)
        task = cached[0]
        try:
            return await asyncio.shield(task)
        except Exception:
            if self._catalogs.get(key) is cached:
                del self._catalogs[key]
            raise

    @staticmethod
    def _discover_catalog(cfg, *, refresh=False):
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
            probe_custom_providers=False, for_picker=True, excluded_providers=sorted(excluded), refresh=refresh)
        hidden = policy.get("hidden_model_prefixes") or {}
        return [{**row, "is_current": False,
                 "models": [m for m in row["models"] if not any(
                     m.startswith(prefix) for prefix in hidden.get(row["slug"], []))]}
                for row in rows if row["slug"] not in excluded]

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
            session_key = self._canonical_session_key(source)
            cfg = self._config(source)
            # Complete discovery before sampling the session's selected values.
            provider_rows = await self._catalog(source, cfg) if include_catalog else []
            from .settings import GatewaySettingsEndpoint
            from gateway.display_config import resolve_display_setting
            from hermes_cli.runtime_settings import reasoning_name
            from hermes_cli.models import resolve_fast_mode_overrides
            from hermes_constants import resolve_reasoning_config

            endpoint = GatewaySettingsEndpoint(self.runner, source, session_key, cfg, None)
            async with self.runner._session_state(session_key).persistent.settings_lock:
                actual = await endpoint.load()
            model_cfg = cfg.get("model") or {}
            if isinstance(model_cfg, str):
                model_cfg = {"default": model_cfg}
            global_model = model_cfg.get("default") or "unknown"
            global_provider = model_cfg.get("provider") or ""
            effective_model, effective_provider = actual.model, actual.provider
            effective_base_url, effective_api_mode = actual.base_url, actual.api_mode
            effective_reasoning = actual.reasoning
            global_reasoning = reasoning_name(resolve_reasoning_config(cfg, actual.model))
            has_reasoning_override = not actual.reasoning_inherited
            fast_mode = actual.service_tier == "priority"
            fast_supported = resolve_fast_mode_overrides(
                actual.model, provider=actual.provider, base_url=actual.base_url) is not None
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
                "model_source": endpoint.model_source,
                "effective_reasoning": effective_reasoning,
                "global_reasoning": global_reasoning,
                "reasoning_source": "本会话覆盖" if has_reasoning_override else "Profile 全局默认",
                "value_source": (
                    "本会话覆盖"
                    if endpoint.model_source == "本会话覆盖" or has_reasoning_override
                    else endpoint.model_source
                ),
                "show_reasoning": resolve_display_setting(cfg, source.platform.value, "show_reasoning"),
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
            session_key = self._canonical_session_key(source)
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
