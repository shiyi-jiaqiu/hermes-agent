"""Synchronous callback router and asynchronous control executor."""

from __future__ import annotations

import asyncio
import copy
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

from gateway.control import HermesPanelControlService

from .actions import PanelAction, PanelActionError, parse_panel_action
from .reducer import reduce_panel_state
from .renderer import MODEL_PAGE_SIZE, PROVIDER_PAGE_SIZE, SESSION_PAGE_SIZE, render_panel
from .state import PanelState
from .store import PanelStateStore

logger = logging.getLogger(__name__)

# A control task is process-local. If its persisted lease survives this long,
# the task either died with a gateway reload or failed before clearing state.
_BUSY_LEASE_SECONDS = 120.0

# Feishu requires delayed-card updates to run after the callback response has
# been delivered. The SDK callback has no response-sent hook, so keep a small
# lower bound before using the callback token.
_CALLBACK_SETTLE_SECONDS = 0.20

# Slow data is loaded only when its page is opened. A bounded timeout keeps the
# callback lane responsive even when provider discovery or a local endpoint is
# offline. Model catalogs are profile-scoped and reused briefly across panels.
_VIEW_LOAD_TIMEOUT_SECONDS = 6.0
_MODEL_CATALOG_CACHE_TTL_SECONDS = 300.0
_VIEW_ALIASES = {
    "model": "model",
    "model_provider": "model",
    "sessions": "sessions",
    "status": "status",
    "home": "home",
}

# These operations contain no view-relative index and can be safely applied to
# the latest state. This is concurrency handling, not a legacy-card fallback.
_REBASABLE_STALE_OPS = frozenset({"nav", "home", "close", "refresh"})


@dataclass(frozen=True)
class PanelCallbackResult:
    card: Optional[dict[str, Any]] = None
    toast: str = ""
    toast_type: str = "info"


class FeishuPanelController:
    """Own one state machine per user/chat/thread panel scope."""

    def __init__(self, adapter: Any, store: PanelStateStore):
        self.adapter = adapter
        self.store = store
        self._view_tasks: set[asyncio.Task[Any]] = set()
        self._model_catalog_cache: dict[
            str, tuple[float, dict[str, Any]]
        ] = {}

    def _service(self) -> HermesPanelControlService:
        runner = getattr(self.adapter, "gateway_runner", None)
        if runner is None:
            raise RuntimeError("Feishu panel is not attached to a gateway runner")
        return HermesPanelControlService(runner)

    def _profile_for_source(self, source: Any) -> str:
        profile = str(getattr(source, "profile", "") or "").strip()
        runner = getattr(self.adapter, "gateway_runner", None)
        if not profile and runner is not None:
            try:
                profile = str(runner._profile_name_for_source(source) or "").strip()
            except Exception:
                profile = ""
        return profile or "default"

    @staticmethod
    def _load_view_name(view: str) -> str:
        return _VIEW_ALIASES.get(str(view or ""), "")

    @staticmethod
    def _mark_view_loading(state: PanelState, view: str) -> None:
        name = FeishuPanelController._load_view_name(view)
        if not name:
            return
        loaded = set(state.data.get("loaded_views") or [])
        loading = set(state.data.get("loading_views") or [])
        loaded.discard(name)
        loading.add(name)
        state.data["loaded_views"] = sorted(loaded)
        state.data["loading_views"] = sorted(loading)
        errors = dict(state.data.get("load_errors") or {})
        errors.pop(name, None)
        state.data["load_errors"] = errors

    async def create_panel_state(
        self,
        *,
        source: Any,
        session_key: str,
        status_text: str,
        owner_open_id: str = "",
        initial_view: str = "home",
    ) -> tuple[PanelState, Optional[PanelState]]:
        owner = str(
            owner_open_id
            or getattr(source, "user_id", "")
            or getattr(source, "user_id_alt", "")
            or ""
        ).strip()
        if not owner:
            raise ValueError("Panel owner identity is required")
        data = await self._service().snapshot(
            source=source,
            session_key=session_key,
            status_text=status_text,
            # Opening a panel must not wait for provider discovery, session
            # enumeration, or status formatting. Those are hydrated after the
            # initial card has been acknowledged by Feishu.
            include_catalog=False,
            include_sessions=False,
            include_status=False,
        )
        data["loaded_views"] = ["home"]
        data["loading_views"] = []
        data["load_errors"] = {}
        state = PanelState(
            panel_id=f"p_{uuid.uuid4().hex}",
            message_id="",
            app_id=str(getattr(self.adapter, "_app_id", "") or "feishu"),
            owner_open_id=owner,
            chat_id=str(getattr(source, "chat_id", "") or ""),
            thread_id=str(getattr(source, "thread_id", "") or ""),
            session_key=str(session_key or ""),
            profile=self._profile_for_source(source),
            chat_type=str(getattr(source, "chat_type", "") or "group"),
            user_id=str(getattr(source, "user_id", "") or ""),
            user_id_alt=str(getattr(source, "user_id_alt", "") or ""),
            user_name=str(getattr(source, "user_name", "") or ""),
            data=data,
        )
        if initial_view in {"model", "reasoning", "sessions", "status"}:
            state.view = initial_view
            state.view_stack = ["home"]
            if self._load_view_name(initial_view):
                self._mark_view_loading(state, initial_view)
        replaced = self.store.create_active(state)
        return state, replaced

    def schedule_view_load(self, panel_id: str, view: str) -> bool:
        """Load one page after its initial card has been sent."""
        try:
            task = asyncio.create_task(self._load_view(panel_id=panel_id, view=view))
        except RuntimeError:
            return False
        self._view_tasks.add(task)

        def _complete(done: asyncio.Task[Any]) -> None:
            self._view_tasks.discard(done)
            if done.cancelled():
                return
            try:
                done.result()
            except Exception:
                logger.warning("[Feishu Panel] view load failed", exc_info=True)

        task.add_done_callback(_complete)
        return True

    @staticmethod
    def _view_snapshot_flags(view: str) -> tuple[bool, bool, bool]:
        return view == "model", view == "sessions", view == "status"

    @staticmethod
    def _view_payload(view: str, snapshot: dict[str, Any]) -> dict[str, Any]:
        keys: tuple[str, ...]
        if view == "model":
            keys = ("model_providers", "model_options")
        elif view == "sessions":
            keys = ("sessions",)
        elif view == "status":
            keys = ("status_text", "running")
        else:
            # Home refreshes only cheap local/session state. Optional catalogs
            # are absent because all include flags are false.
            return dict(snapshot)
        return {key: snapshot[key] for key in keys if key in snapshot}

    async def _update_loaded_card(
        self,
        state: PanelState,
        *,
        callback_token: str = "",
        callback_started_at: float = 0.0,
    ) -> None:
        if not state.message_id:
            return
        card = render_panel(state)
        if callback_token:
            elapsed = time.monotonic() - callback_started_at
            if elapsed < _CALLBACK_SETTLE_SECONDS:
                await asyncio.sleep(_CALLBACK_SETTLE_SECONDS - elapsed)
            update = await self.adapter.update_interactive_card_after_callback(
                callback_token=callback_token,
                card=card,
            )
        else:
            # This path is only used for an initial /panel <view> card, before
            # any interaction callback exists.
            update = await self.adapter.update_interactive_message(
                message_id=state.message_id,
                card=card,
            )
        if not update.success:
            logger.warning(
                "[Feishu Panel] failed to update loaded view panel=%s view=%s: %s",
                state.panel_id,
                state.view,
                update.error,
            )

    async def _load_view(
        self,
        *,
        panel_id: str,
        view: str,
        callback_token: str = "",
        callback_started_at: float = 0.0,
        force_reload: bool = False,
    ) -> None:
        """Load only the data required by one view, under a hard deadline."""
        view = self._load_view_name(view)
        if not view:
            return
        state = self.store.get(panel_id)
        if state is None or not self.store.is_active(state):
            return
        source = self._source(state)
        include_catalog, include_sessions, include_status = self._view_snapshot_flags(view)
        error = ""
        payload: dict[str, Any] | None = None
        cache_key = state.profile or "default"
        if view == "model" and not force_reload:
            cached = self._model_catalog_cache.get(cache_key)
            if cached and time.monotonic() - cached[0] < _MODEL_CATALOG_CACHE_TTL_SECONDS:
                payload = copy.deepcopy(cached[1])
        try:
            if payload is None:
                snapshot = await asyncio.wait_for(
                    self._service().snapshot(
                        source=source,
                        session_key=state.session_key,
                        status_text="",
                        include_catalog=include_catalog,
                        include_sessions=include_sessions,
                        include_status=include_status,
                    ),
                    timeout=_VIEW_LOAD_TIMEOUT_SECONDS,
                )
                payload = self._view_payload(view, snapshot)
                if view == "model":
                    self._model_catalog_cache[cache_key] = (
                        time.monotonic(),
                        copy.deepcopy(payload),
                    )
            if view == "model":
                effective_provider = str(state.data.get("effective_provider") or "")
                for provider in payload.get("model_providers") or []:
                    if isinstance(provider, dict):
                        provider["is_current"] = (
                            str(provider.get("slug") or "") == effective_provider
                        )
        except asyncio.TimeoutError:
            payload = {}
            error = f"加载超时（>{_VIEW_LOAD_TIMEOUT_SECONDS:g}s），请重试"
        except Exception as exc:
            logger.warning(
                "[Feishu Panel] view load failed panel=%s view=%s: %s",
                panel_id,
                view,
                exc,
                exc_info=True,
            )
            payload = {}
            error = f"加载失败：{str(exc)[:120]}"

        updated: PanelState | None = None
        for _attempt in range(3):
            latest = self.store.get(panel_id)
            if latest is None or not self.store.is_active(latest):
                return
            expected = latest.revision
            latest.data.update(payload)
            loaded = set(latest.data.get("loaded_views") or [])
            loading = set(latest.data.get("loading_views") or [])
            errors = dict(latest.data.get("load_errors") or {})
            loading.discard(view)
            if error:
                loaded.discard(view)
                errors[view] = error
            else:
                loaded.add(view)
                errors.pop(view, None)
            latest.data["loaded_views"] = sorted(loaded)
            latest.data["loading_views"] = sorted(loading)
            latest.data["load_errors"] = errors
            latest.revision += 1
            if self.store.compare_and_set(expected, latest):
                updated = latest
                break
        if updated is not None:
            await self._update_loaded_card(
                updated,
                callback_token=callback_token,
                callback_started_at=callback_started_at,
            )

    def attach_message_id(self, state: PanelState, message_id: str) -> bool:
        current = self.store.get(state.panel_id)
        if current is None:
            return False
        expected = current.revision
        current.message_id = str(message_id or "")
        return self.store.compare_and_set(expected, current)

    def discard(self, state: PanelState) -> None:
        self.store.delete(state.panel_id)

    def _source(self, state: PanelState) -> Any:
        source = self.adapter.build_source(
            chat_id=state.chat_id,
            chat_name=state.chat_id or "Feishu Chat",
            chat_type=state.chat_type,
            user_id=state.user_id or state.owner_open_id,
            user_name=state.user_name or state.owner_open_id,
            thread_id=state.thread_id or None,
            user_id_alt=state.user_id_alt or None,
        )
        source.profile = state.profile or "default"
        return source

    @staticmethod
    def _event_identity(data: Any) -> tuple[str, str]:
        event = getattr(data, "event", None)
        operator = getattr(event, "operator", None)
        operator_id = getattr(operator, "operator_id", None)
        open_id = str(
            getattr(operator, "open_id", "")
            or getattr(operator_id, "open_id", "")
            or ""
        )
        context = getattr(event, "context", None)
        chat_id = str(getattr(context, "open_chat_id", "") or "")
        return open_id, chat_id

    @staticmethod
    def _clamp_page(state: PanelState) -> None:
        if state.view == "model":
            total = len(state.data.get("model_providers") or [])
            pages = max(1, math.ceil(total / PROVIDER_PAGE_SIZE))
            state.page = min(max(0, state.page), pages - 1)
        elif state.view == "model_provider":
            providers = list(state.data.get("model_providers") or [])
            selected_slug = str(state.filters.get("model_provider") or "")
            provider = next(
                (
                    item
                    for item in providers
                    if isinstance(item, dict)
                    and str(item.get("slug") or "") == selected_slug
                ),
                {},
            )
            total = len(provider.get("model_indices") or [])
            pages = max(1, math.ceil(total / MODEL_PAGE_SIZE))
            state.page = min(max(0, state.page), pages - 1)
        elif state.view == "sessions":
            total = len(state.data.get("sessions") or [])
            pages = max(1, math.ceil(total / SESSION_PAGE_SIZE))
            state.page = min(max(0, state.page), pages - 1)
        else:
            state.page = 0

    def handle_sync(self, data: Any, raw_value: Any, loop: Any) -> PanelCallbackResult:
        """Validate, CAS and render before Feishu's three-second deadline."""
        callback_started_at = time.monotonic()
        callback_token = str(
            getattr(getattr(data, "event", None), "token", "") or ""
        )
        try:
            action = parse_panel_action(raw_value)
        except PanelActionError:
            return PanelCallbackResult(toast="无效的面板操作", toast_type="error")
        state = self.store.get(action.panel_id)
        if state is None or state.expires_at <= time.time():
            return PanelCallbackResult(toast="此面板已过期，请重新打开 /panel", toast_type="warning")
        open_id, chat_id = self._event_identity(data)
        if open_id != state.owner_open_id or (chat_id and chat_id != state.chat_id):
            logger.warning(
                "[Feishu Panel] rejected owner/chat mismatch panel=%s operator=%s chat=%s",
                state.panel_id,
                open_id or "<unknown>",
                chat_id or "<unknown>",
            )
            return PanelCallbackResult(toast="你无权操作此用户的面板", toast_type="error")
        if not self.adapter._is_interactive_operator_authorized(open_id):
            return PanelCallbackResult(toast="未授权的面板操作", toast_type="error")
        if not self.store.is_active(state):
            state.active = False
            state.lifecycle = "replaced" if state.lifecycle == "replaced" else state.lifecycle
            return PanelCallbackResult(card=render_panel(state), toast="此面板已被替代，请使用最新面板", toast_type="warning")

        # Recover an orphaned process-local task lease. This is deliberately
        # checked before revision/nonce validation so a card left busy by a
        # crash or reload can heal on the first callback instead of remaining
        # permanently disabled.
        if state.busy_action_id and (
            not state.busy_started_at
            or time.time() - state.busy_started_at >= _BUSY_LEASE_SECONDS
        ):
            recovered = state.clone()
            recovered.busy_action_id = ""
            recovered.busy_started_at = 0.0
            recovered.data["flash"] = "⚠️ 上一次控制任务已中断，面板已自动恢复"
            recovered.revision += 1
            if self.store.compare_and_set(state.revision, recovered):
                return PanelCallbackResult(
                    card=render_panel(recovered),
                    toast="已恢复中断的控制任务，请重试刚才的操作",
                    toast_type="warning",
                )
            state = self.store.get(state.panel_id) or state
        if action.nonce in state.handled_nonces:
            return PanelCallbackResult(card=render_panel(state), toast="该操作已经处理", toast_type="warning")
        if action.revision != state.revision and not (
            action.revision < state.revision
            and (
                action.op in _REBASABLE_STALE_OPS
                or (action.op == "exec" and action.target == "stop")
            )
        ):
            return PanelCallbackResult(card=render_panel(state), toast="页面已更新，已恢复到最新状态", toast_type="warning")

        if action.op in {"nav", "back", "home", "page", "close"}:
            try:
                new_state = reduce_panel_state(state, action)
            except PanelActionError:
                return PanelCallbackResult(card=render_panel(state), toast="当前页面不支持该操作", toast_type="error")
            self._clamp_page(new_state)
            load_view = self._load_view_name(new_state.view)
            loaded_views = set(new_state.data.get("loaded_views") or [])
            loading_views = set(new_state.data.get("loading_views") or [])
            needs_load = bool(
                load_view
                and load_view not in loaded_views
                and load_view not in loading_views
            )
            if needs_load:
                if not callback_token.startswith("c-"):
                    return PanelCallbackResult(
                        card=render_panel(state),
                        toast="飞书回调凭证无效，请重新打开面板",
                        toast_type="error",
                    )
                self._mark_view_loading(new_state, load_view)
            new_state.remember_nonce(action.nonce)
            new_state.revision += 1
            if not self.store.compare_and_set(state.revision, new_state):
                latest = self.store.get(state.panel_id) or state
                return PanelCallbackResult(card=render_panel(latest), toast="页面已被其他操作更新", toast_type="warning")
            if needs_load:
                scheduled = self.adapter._submit_on_loop(
                    loop,
                    self._load_view(
                        panel_id=state.panel_id,
                        view=load_view,
                        callback_token=callback_token,
                        callback_started_at=callback_started_at,
                    ),
                )
                if not scheduled:
                    return PanelCallbackResult(
                        card=render_panel(new_state),
                        toast="页面加载任务无法调度，请重试",
                        toast_type="error",
                    )
            return PanelCallbackResult(
                card=render_panel(new_state),
                toast="面板已关闭" if action.op == "close" else "",
            )

        if action.op == "select":
            if action.index is None:
                return PanelCallbackResult(card=render_panel(state), toast="无效的选择", toast_type="error")
            new_state = state.clone()
            if action.target == "model_provider":
                providers = list(state.data.get("model_providers") or [])
                if action.index >= len(providers):
                    return PanelCallbackResult(card=render_panel(state), toast="供应商选择已失效", toast_type="error")
                provider = providers[action.index]
                slug = str(provider.get("slug") or "") if isinstance(provider, dict) else ""
                if not slug or not list(provider.get("model_indices") or []):
                    return PanelCallbackResult(card=render_panel(state), toast="该供应商没有可用模型", toast_type="warning")
                new_state.filters["model_provider"] = slug
                if new_state.view != "model_provider":
                    new_state.view_stack.append(new_state.view)
                new_state.view = "model_provider"
            elif action.target == "global_reasoning":
                options = list(state.data.get("reasoning_options") or [])
                if action.index >= len(options):
                    return PanelCallbackResult(card=render_panel(state), toast="选择已失效", toast_type="error")
                new_state.data["pending_global_reasoning_index"] = action.index
                new_state.view_stack.append(new_state.view)
                new_state.view = "confirm_global_reasoning"
            else:
                return PanelCallbackResult(card=render_panel(state), toast="无效的选择", toast_type="error")
            new_state.page = 0
            new_state.remember_nonce(action.nonce)
            new_state.revision += 1
            if not self.store.compare_and_set(state.revision, new_state):
                latest = self.store.get(state.panel_id) or state
                return PanelCallbackResult(card=render_panel(latest), toast="页面已被其他操作更新", toast_type="warning")
            return PanelCallbackResult(card=render_panel(new_state))

        if action.op == "refresh":
            if not callback_token.startswith("c-"):
                return PanelCallbackResult(
                    card=render_panel(state),
                    toast="飞书回调凭证无效，请重新打开面板",
                    toast_type="error",
                )
            refresh_view = self._load_view_name(state.view) or "home"
            if refresh_view in set(state.data.get("loading_views") or []):
                return PanelCallbackResult(
                    card=render_panel(state),
                    toast="当前页面正在刷新",
                    toast_type="warning",
                )
            refreshing = state.clone()
            self._mark_view_loading(refreshing, refresh_view)
            refreshing.data.pop("flash", None)
            refreshing.remember_nonce(action.nonce)
            refreshing.revision += 1
            if not self.store.compare_and_set(state.revision, refreshing):
                latest = self.store.get(state.panel_id) or state
                return PanelCallbackResult(card=render_panel(latest), toast="刷新发生冲突，请重试", toast_type="warning")
            scheduled = self.adapter._submit_on_loop(
                loop,
                self._load_view(
                    panel_id=state.panel_id,
                    view=refresh_view,
                    callback_token=callback_token,
                    callback_started_at=callback_started_at,
                    force_reload=True,
                ),
            )
            if not scheduled:
                return PanelCallbackResult(card=render_panel(refreshing), toast="刷新任务无法调度", toast_type="error")
            return PanelCallbackResult(card=render_panel(refreshing), toast="正在刷新…")

        if action.op != "exec":
            return PanelCallbackResult(card=render_panel(state), toast="不支持的面板操作", toast_type="error")
        target = action.target
        if not callback_token.startswith("c-"):
            return PanelCallbackResult(
                card=render_panel(state),
                toast="飞书回调凭证无效，请重新打开面板",
                toast_type="error",
            )
        if state.busy_action_id and target != "stop":
            return PanelCallbackResult(card=render_panel(state), toast="已有控制操作正在处理", toast_type="warning")

        busy = state.clone()
        busy.busy_action_id = action.nonce
        busy.busy_started_at = time.time()
        busy.data.pop("flash", None)
        busy.remember_nonce(action.nonce)
        busy.revision += 1
        if not self.store.compare_and_set(state.revision, busy):
            latest = self.store.get(state.panel_id) or state
            return PanelCallbackResult(card=render_panel(latest), toast="操作发生冲突，请重试", toast_type="warning")
        scheduled = self.adapter._submit_on_loop(
            loop,
            self._execute_control(
                panel_id=state.panel_id,
                action_id=action.nonce,
                target=target,
                index=action.index,
                state_data=state.data,
                callback_token=callback_token,
                callback_started_at=callback_started_at,
            ),
        )
        if not scheduled:
            failed = self.store.get(state.panel_id) or busy
            expected = failed.revision
            failed.busy_action_id = ""
            failed.busy_started_at = 0.0
            failed.data["flash"] = "❌ 控制任务无法调度，请重试"
            failed.revision += 1
            self.store.compare_and_set(expected, failed)
            return PanelCallbackResult(card=render_panel(failed), toast="控制任务无法调度", toast_type="error")
        return PanelCallbackResult(card=render_panel(busy), toast="正在处理…")

    async def _execute_control(
        self,
        *,
        panel_id: str,
        action_id: str,
        target: str,
        index: int | None,
        state_data: dict[str, Any],
        callback_token: str = "",
        callback_started_at: float = 0.0,
    ) -> None:
        state = self.store.get(panel_id)
        if state is None or state.busy_action_id != action_id or not self.store.is_active(state):
            return
        source = self._source(state)
        try:
            result = await self._service().execute(
                source=source,
                session_key=state.session_key,
                target=target,
                index=index,
                state_data=state_data,
            )
            snapshot = await self._service().snapshot(
                source=source,
                session_key=state.session_key,
                status_text="",
                include_catalog=False,
                include_sessions=False,
                include_status=False,
            )
        except Exception as exc:
            logger.error("[Feishu Panel] control execution failed: %s", exc, exc_info=True)
            result = None
            snapshot = dict(state.data)
            snapshot["flash"] = f"❌ 操作失败：{str(exc)[:180]}"

        latest = self.store.get(panel_id)
        if latest is None or latest.busy_action_id != action_id or not self.store.is_active(latest):
            return
        expected = latest.revision
        latest.data.update(snapshot)
        if result is not None:
            prefix = "✅" if result.success else "❌"
            text = str(result.text or "").strip()
            # Slash-command handlers often include their own status glyph.
            # Do not produce the screenshot-visible "✅ ✅ ..." duplication.
            latest.data["flash"] = (
                text
                if text.startswith(("✅", "❌"))
                else f"{prefix} {text}"
            )[:300]
        latest.busy_action_id = ""
        latest.busy_started_at = 0.0
        latest.data.pop("pending_global_reasoning_index", None)
        loaded_views = set(latest.data.get("loaded_views") or [])
        loaded_views.discard("status")
        if target in {"new", "resume"}:
            loaded_views.discard("sessions")
        latest.data["loaded_views"] = sorted(loaded_views)
        if target in {"new", "resume", "preset"}:
            latest.view = "home"
            latest.view_stack.clear()
            latest.page = 0
        elif target == "global_reasoning":
            latest.view = "reasoning"
            latest.view_stack = [view for view in latest.view_stack if view not in {"reasoning_global", "confirm_global_reasoning"}]
            latest.page = 0
        latest.revision += 1
        if not self.store.compare_and_set(expected, latest):
            return
        if latest.message_id:
            card = render_panel(latest)
            elapsed = time.monotonic() - callback_started_at
            if elapsed < _CALLBACK_SETTLE_SECONDS:
                await asyncio.sleep(_CALLBACK_SETTLE_SECONDS - elapsed)
            update = await self.adapter.update_interactive_card_after_callback(
                callback_token=callback_token,
                card=card,
            )
            if not update.success:
                logger.warning(
                    "[Feishu Panel] delayed update failed panel=%s message=%s: %s",
                    latest.panel_id,
                    latest.message_id,
                    update.error,
                )
