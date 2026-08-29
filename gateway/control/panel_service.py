"""Trusted Hermes controls for stateful interactive panels.

This service deliberately calls the existing slash-command application methods
rather than entering ``GatewayRunner._handle_message``. It therefore reuses the
same model/reasoning/session semantics without the per-session Agent queue or a
second implementation of those mutations.
"""

from __future__ import annotations

import asyncio
import shlex
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

from gateway.platforms.base import MessageEvent, MessageType
from .panel_catalog import (
    HIDDEN_PANEL_PROVIDER_SLUGS,
    is_hidden_openrouter_model,
    is_hidden_panel_provider,
)


@dataclass(frozen=True)
class PanelControlResult:
    success: bool
    text: str


class HermesPanelControlService:
    def __init__(self, runner: Any):
        self.runner = runner

    def _scope(self, source: Any):
        if not getattr(getattr(self.runner, "config", None), "multiplex_profiles", False):
            return nullcontext()
        from gateway.run import _profile_runtime_scope

        return _profile_runtime_scope(self.runner._resolve_profile_home_for_source(source))

    def _config(self, source: Any) -> dict[str, Any]:
        from gateway.run import _load_gateway_config

        path = self.runner._resolve_profile_home_for_source(source) / "config.yaml"
        return _load_gateway_config(config_path=path) or {}

    @staticmethod
    def _reasoning_value(config: Any) -> str:
        if not isinstance(config, dict):
            return "medium"
        if config.get("enabled") is False:
            return "none"
        return str(config.get("effort") or "medium")

    @staticmethod
    def _failed(text: str) -> bool:
        lowered = str(text or "").strip().lower()
        return lowered.startswith("❌") or "partial" in lowered or "but reasoning failed" in lowered

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

    @classmethod
    def _build_model_catalog(
        cls,
        *,
        provider_rows: list[dict[str, Any]],
        aliases: dict[str, Any],
        effective_model: str,
        effective_provider: str,
        global_model: str,
        global_provider: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
        """Normalize picker inventory into provider -> trusted model indices.

        Feishu action payloads still carry only an opaque panel reference and a
        server-owned index. Provider slugs, model IDs and aliases remain in the
        persisted panel state and can never be supplied by a modified card.
        """
        providers: list[dict[str, Any]] = []
        provider_by_slug: dict[str, dict[str, Any]] = {}
        model_options: list[dict[str, str]] = []
        option_by_route: dict[tuple[str, str], int] = {}
        aliases_by_route: dict[tuple[str, str], list[str]] = {}
        aliases_by_model: dict[str, list[str]] = {}

        for alias, spec in aliases.items():
            model, provider, _target = cls._model_alias_target(spec, str(alias))
            normalized_model = str(model or "").strip()
            normalized_provider = str(provider or "").strip()
            if not normalized_model:
                continue
            key = (normalized_provider, normalized_model)
            if normalized_provider:
                aliases_by_route.setdefault(key, []).append(str(alias))
            else:
                aliases_by_model.setdefault(normalized_model, []).append(str(alias))

        def ensure_provider(
            slug: str,
            *,
            name: str = "",
            is_current: bool = False,
            reported_total: int = 0,
        ) -> dict[str, Any]:
            normalized = str(slug or "").strip() or "unknown"
            existing = provider_by_slug.get(normalized)
            if existing is not None:
                existing["is_current"] = bool(existing.get("is_current") or is_current)
                existing["total_models"] = max(
                    int(existing.get("total_models") or 0), int(reported_total or 0)
                )
                if name and existing.get("name") in {"", normalized}:
                    existing["name"] = str(name)
                return existing
            provider = {
                "slug": normalized,
                "name": str(name or normalized),
                "is_current": bool(is_current),
                "total_models": max(0, int(reported_total or 0)),
                "model_indices": [],
            }
            providers.append(provider)
            provider_by_slug[normalized] = provider
            return provider

        def add_model(provider: dict[str, Any], model: str) -> None:
            normalized_model = str(model or "").strip()
            if not normalized_model:
                return
            slug = str(provider.get("slug") or "unknown")
            if is_hidden_panel_provider(slug) or is_hidden_openrouter_model(
                slug, normalized_model
            ):
                return
            route = (slug, normalized_model)
            if route in option_by_route:
                return
            aliases_for_model = (
                aliases_by_route.get(route) or aliases_by_model.get(normalized_model) or []
            )
            label = (
                f"{' / '.join(aliases_for_model)} · {normalized_model}"
                if aliases_for_model
                else normalized_model
            )
            index = len(model_options)
            option_by_route[route] = index
            model_options.append(
                {
                    "model": normalized_model,
                    # A selection made below a provider must preserve that
                    # provider explicitly instead of relying on alias/global
                    # resolution that could route the same ID elsewhere.
                    "target": normalized_model,
                    "provider": slug if slug != "unknown" else "",
                    "label": label,
                }
            )
            provider["model_indices"].append(index)

        for row in provider_rows:
            if not isinstance(row, dict):
                continue
            slug = str(row.get("slug") or "").strip()
            if not slug:
                continue
            models = [str(model).strip() for model in (row.get("models") or []) if str(model).strip()]
            if slug.lower() == "openrouter":
                models = [
                    model
                    for model in models
                    if not is_hidden_openrouter_model(slug, model)
                ]
            provider_entry = ensure_provider(
                slug,
                name=str(row.get("name") or slug),
                is_current=bool(row.get("is_current") or slug == effective_provider),
                # Do not expose the pre-filter count in the Panel.  The count
                # is part of the user-visible provider button.
                reported_total=len(models),
            )
            for model in models:
                add_model(provider_entry, model)

        # Configured aliases may point at a custom provider/model absent from a
        # cached or temporarily unavailable remote inventory. Keep those routes
        # selectable, but group them under their actual configured provider.
        for (provider_slug, model), _alias_names in aliases_by_route.items():
            provider_entry = ensure_provider(
                provider_slug,
                is_current=provider_slug == effective_provider,
            )
            add_model(provider_entry, model)

        for model, provider_slug in (
            (effective_model, effective_provider),
            (global_model, global_provider),
        ):
            if not str(model or "").strip():
                continue
            provider_entry = ensure_provider(
                provider_slug,
                is_current=bool(provider_slug and provider_slug == effective_provider),
            )
            add_model(provider_entry, model)

        # Providers with no callable models are not useful menu entries. The
        # current provider remains first, matching the existing /model picker.
        providers = [item for item in providers if item.get("model_indices")]
        providers.sort(
            key=lambda item: (
                not bool(item.get("is_current")),
                str(item.get("name") or item.get("slug") or "").lower(),
            )
        )
        for provider_entry in providers:
            provider_entry["available_models"] = len(
                provider_entry.get("model_indices") or []
            )
            provider_entry["total_models"] = max(
                int(provider_entry.get("total_models") or 0),
                provider_entry["available_models"],
            )
        return providers, model_options

    async def snapshot(
        self,
        *,
        source: Any,
        session_key: str,
        status_text: str = "",
        include_catalog: bool = True,
        include_sessions: bool = True,
        include_status: bool = True,
        catalog_provider_rows: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Return only JSON-serializable, server-trusted panel data.

        The three ``include_*`` switches keep the first card local-only and let
        each view request just its own optional data. The default remains a
        complete snapshot for direct non-Panel callers.
        """
        with self._scope(source):
            cfg = self._config(source)
            raw_model_cfg = cfg.get("model")
            model_cfg: dict[str, Any] = (
                dict(raw_model_cfg) if isinstance(raw_model_cfg, dict) else {}
            )
            global_model = str(model_cfg.get("default") or "unknown")
            global_provider = str(model_cfg.get("provider") or "")
            model_override = dict(
                ((getattr(self.runner, "_session_model_overrides", {}) or {}).get(session_key) or {})
            )
            effective_model = str(model_override.get("model") or global_model)
            effective_provider = str(model_override.get("provider") or global_provider)
            effective_base_url = str(model_override.get("base_url") or model_cfg.get("base_url") or "")

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
            provider_rows: list[dict[str, Any]] = []
            if include_catalog:
                if catalog_provider_rows is not None:
                    # The cache stores provider inventory only. Session-relative
                    # flags are deliberately stripped before rebuilding this
                    # request's effective/global routes below.
                    provider_rows = [
                        {**dict(row), "is_current": False}
                        for row in catalog_provider_rows
                        if isinstance(row, dict)
                    ]
                else:
                    try:
                        from hermes_cli.config import get_compatible_custom_providers
                        from hermes_cli.model_switch import list_authenticated_providers

                        try:
                            custom_providers = get_compatible_custom_providers(cfg)
                        except Exception:
                            custom_providers = cfg.get("custom_providers")
                        model_catalog = cfg.get("model_catalog")
                        configured_exclusions = {
                            str(item).strip()
                            for item in (
                                (model_catalog.get("excluded_providers") or [])
                                if isinstance(model_catalog, dict)
                                else []
                            )
                            if str(item).strip()
                        }
                        excluded_providers = sorted(
                            configured_exclusions | HIDDEN_PANEL_PROVIDER_SLUGS
                        )
                        raw_user_providers = cfg.get("providers")
                        user_providers = (
                            dict(raw_user_providers)
                            if isinstance(raw_user_providers, dict)
                            else {}
                        )
                        discovered = await asyncio.to_thread(
                            list_authenticated_providers,
                            current_provider=effective_provider,
                            current_base_url=effective_base_url,
                            current_model=effective_model,
                            user_providers=user_providers,
                            custom_providers=custom_providers,
                            max_models=50,
                            probe_custom_providers=False,
                            for_picker=True,
                            excluded_providers=excluded_providers,
                        )
                        provider_rows = [
                            {**dict(row), "is_current": False}
                            for row in (discovered or [])
                            if isinstance(row, dict)
                        ]
                    except Exception:
                        provider_rows = []
                model_providers, model_options = self._build_model_catalog(
                    provider_rows=list(provider_rows or []),
                    aliases=aliases,
                    effective_model=effective_model,
                    effective_provider=effective_provider,
                    global_model=global_model,
                    global_provider=global_provider,
                )

            preset_options: list[dict[str, Any]] = []
            label_map = {"fast": "⚡ Quick", "quick": "⚡ Quick", "daily": "⚖ Daily", "deep": "🧠 Deep"}
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
                ):
                    current_preset = str(name)

            session_rows: list[dict[str, Any]] = []
            if include_sessions:
                try:
                    from hermes_cli.session_listing import query_session_listing

                    current_entry = await self.runner.async_session_store.get_or_create_session(source)
                    rows = await asyncio.to_thread(
                        query_session_listing,
                        getattr(self.runner._session_db, "_db", self.runner._session_db),
                        source=source.platform.value,
                        session_key=session_key,
                        current_session_id=current_entry.session_id,
                        include_all_sources=False,
                        include_unnamed=True,
                        search_query=None,
                        limit=50,
                        exclude_sources=["tool"],
                    )
                    caller_source = source.platform.value if source.platform else ""
                    # query_session_listing already applies both predicates at
                    # SQL level. Recheck them in memory so a malformed/test DB
                    # row still fails closed, without issuing one get_session
                    # query per row through _resume_row_visible().
                    session_rows = [
                        dict(row)
                        for row in rows
                        if str(row.get("session_key") or "") == session_key
                        and str(row.get("source") or "") == caller_source
                    ]
                except Exception:
                    session_rows = []

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
                # Controller-only cache material; _view_payload never persists
                # this key into PanelState or sends it to Feishu.
                result["_model_provider_inventory"] = provider_rows
            if include_sessions:
                result["sessions"] = session_rows
            if include_status:
                result["status_text"] = str(status_text)[:3000]
            return result

    def _event(self, source: Any, command: str, *, trusted_model_selection: bool = False) -> MessageEvent:
        raw: dict[str, Any] = {"_hermes_panel_control": True}
        if trusted_model_selection:
            # Existing /mode uses this marker to identify an already-confirmed,
            # server-owned model choice. The panel has the same trust boundary.
            raw["_hermes_mode_preset"] = "panel"
        return MessageEvent(
            text=command,
            message_type=MessageType.COMMAND,
            source=source,
            raw_message=raw,
            message_id="",
        )

    async def execute(
        self,
        *,
        source: Any,
        session_key: str,
        target: str,
        index: int | None,
        state_data: dict[str, Any],
    ) -> PanelControlResult:
        """Execute one allowlisted control using values resolved from state_data."""
        with self._scope(source):
            if target == "snapshot":
                return PanelControlResult(True, "状态已刷新")
            if target == "fast":
                options = list(
                    state_data.get("fast_options")
                    or [
                        {"value": "fast", "label": "⚡ Fast"},
                        {"value": "normal", "label": "正常"},
                    ]
                )
                if index is None or index >= len(options):
                    return PanelControlResult(False, "无效的 Fast 设置")
                value = str(options[index].get("value") or "").strip().lower()
                if value not in {"fast", "normal"}:
                    return PanelControlResult(False, "无效的 Fast 设置")
                result = str(
                    await self.runner._handle_fast_command(
                        self._event(source, f"/fast {shlex.quote(value)}")
                    )
                    or ""
                )
                return PanelControlResult(not self._failed(result), result or "Fast 设置已更新")
            if target == "preset":
                options = list(state_data.get("preset_options") or [])
                if index is None or index >= len(options):
                    return PanelControlResult(False, "无效的预设索引")
                selected = options[index]
                name = str(selected.get("name") or "")
                model_snapshot = self.runner._snapshot_session_model_override(session_key)
                state = self.runner._peek_session_state(session_key)
                reasoning_snapshot = (
                    None
                    if state is None or state.conversation.reasoning_override is None
                    else dict(state.conversation.reasoning_override)
                )
                service_tier_snapshot = self.runner._resolve_session_service_tier(
                    session_key=session_key
                )

                async def restore_preset_snapshot() -> None:
                    self.runner._restore_session_model_override(session_key, model_snapshot)
                    self.runner._set_session_reasoning_override(session_key, reasoning_snapshot)
                    self.runner._set_session_service_tier_override(
                        session_key, service_tier_snapshot
                    )
                    restored = (
                        model_snapshot.get("override")
                        if model_snapshot.get("had_override")
                        else None
                    )
                    try:
                        await self.runner.async_session_store.set_model_override(
                            session_key, restored
                        )
                    except Exception:
                        pass
                    self.runner._evict_cached_agent(session_key)

                try:
                    result = str(
                        await self.runner._handle_mode_command(
                            self._event(
                                source,
                                f"/mode {shlex.quote(name)}",
                                trusted_model_selection=True,
                            )
                        )
                        or ""
                    )
                except Exception:
                    # /mode can mutate one or more in-memory overrides before a
                    # provider/persistence failure escapes. Preserve the atomic
                    # Panel preset contract before the controller reports it.
                    await restore_preset_snapshot()
                    raise
                if self._failed(result):
                    await restore_preset_snapshot()
                    return PanelControlResult(False, f"预设应用失败，已回滚：{result}")

                desired_fast = bool(selected.get("fast_mode", False))
                self.runner._set_session_service_tier_override(
                    session_key,
                    "priority" if desired_fast else None,
                )
                self.runner._evict_cached_agent(session_key)

                # A slash handler returning success is not sufficient: confirm
                # the effective session state before claiming an atomic preset
                # application. This catches deferred/partial model switches and
                # guarantees that the next rendered card cannot say "success"
                # while still showing the previous model.
                desired_model = str(selected.get("model") or "")
                desired_reasoning = str(selected.get("reasoning") or "")
                model_override = dict(
                    ((getattr(self.runner, "_session_model_overrides", {}) or {}).get(session_key) or {})
                )
                actual_model = str(model_override.get("model") or "")
                actual_reasoning = self._reasoning_value(
                    self.runner._resolve_session_reasoning_config(
                        source=source,
                        session_key=session_key,
                        model=actual_model,
                    )
                )
                actual_fast = self.runner._resolve_session_service_tier(
                    session_key=session_key
                ) == "priority"
                expected_reasoning = (
                    "none"
                    if desired_reasoning.lower()
                    in {"provider", "provider-managed", "provider_managed", "auto"}
                    else desired_reasoning
                )
                if (
                    actual_model != desired_model
                    or actual_reasoning != expected_reasoning
                    or actual_fast != desired_fast
                ):
                    await restore_preset_snapshot()
                    return PanelControlResult(
                        False,
                        "预设结果校验失败，已回滚"
                        f"（期望 {desired_model}/{expected_reasoning}/Fast "
                        f"{'on' if desired_fast else 'off'}；实际 "
                        f"{actual_model or 'unknown'}/{actual_reasoning}/Fast "
                        f"{'on' if actual_fast else 'off'}）",
                    )

                label = str(selected.get("label") or name)
                return PanelControlResult(
                    True,
                    f"已应用 {label}：{desired_model} · Reasoning "
                    f"{expected_reasoning} · Fast {'on' if desired_fast else 'off'}",
                )
            if target == "model":
                options = list(state_data.get("model_options") or [])
                if index is None or index >= len(options):
                    return PanelControlResult(False, "无效的模型索引")
                option = options[index]
                model_target = str(option.get("target") or option.get("model") or "")
                provider = str(option.get("provider") or "")
                command = f"/model {shlex.quote(model_target)} --session"
                if provider and model_target == str(option.get("model") or ""):
                    command += f" --provider {shlex.quote(provider)}"
                result = str(
                    await self.runner._handle_model_command(
                        self._event(source, command, trusted_model_selection=True)
                    )
                    or ""
                )
                return PanelControlResult(not self._failed(result), result or "模型已切换")
            if target in {"reasoning", "global_reasoning"}:
                options = list(state_data.get("reasoning_options") or [])
                if index is None or index >= len(options):
                    return PanelControlResult(False, "无效的推理等级索引")
                value = str(options[index].get("value") or "")
                command = f"/reasoning {shlex.quote(value)}"
                if target == "global_reasoning":
                    command += " --global"
                result = str(await self.runner._handle_reasoning_command(self._event(source, command)) or "")
                return PanelControlResult(not self._failed(result), result or "推理设置已更新")
            if target == "reasoning_reset":
                result = str(await self.runner._handle_reasoning_command(self._event(source, "/reasoning reset")) or "")
                return PanelControlResult(not self._failed(result), result or "会话覆盖已重置")
            if target == "reasoning_display":
                value = "show" if index == 0 else "hide" if index == 1 else ""
                if not value:
                    return PanelControlResult(False, "无效的显示设置")
                result = str(await self.runner._handle_reasoning_command(self._event(source, f"/reasoning {value}")) or "")
                return PanelControlResult(not self._failed(result), result or "显示设置已更新")
            if target == "resume":
                sessions = list(state_data.get("sessions") or [])
                if index is None or index >= len(sessions):
                    return PanelControlResult(False, "无效的会话索引")
                session_id = str(sessions[index].get("id") or "")
                result = str(
                    await self.runner._handle_resume_command(
                        self._event(source, f"/resume {shlex.quote(session_id)}")
                    )
                    or ""
                )
                return PanelControlResult(not self._failed(result), result or "会话已恢复")
            if target == "new":
                result = str(await self.runner._handle_reset_command(self._event(source, "/new")) or "")
                return PanelControlResult(not self._failed(result), result or "新会话已创建")
            if target == "stop":
                result = str(await self.runner._handle_stop_command(self._event(source, "/stop")) or "")
                return PanelControlResult(True, result or "停止请求已发送")
            return PanelControlResult(False, "不支持的控制操作")
