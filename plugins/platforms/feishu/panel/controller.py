"""Loop-owned Panel state, tasks and one ordered sender per card."""
from __future__ import annotations

import asyncio
import concurrent.futures
import logging
import math
import time
import uuid
from dataclasses import dataclass
from typing import Any

from .actions import PanelAction, PanelActionError, parse_panel_action
from .reducer import reduce_panel_state
from .renderer import MODEL_PAGE_SIZE, PROVIDER_PAGE_SIZE, SESSION_PAGE_SIZE, render_panel
from .state import PanelState

logger = logging.getLogger(__name__)
_VIEW_ALIASES = {"model": "model", "model_provider": "model", "sessions": "sessions",
                 "status": "status", "home": "home"}
_REBASABLE_STALE_OPS = frozenset({"nav", "home", "close", "refresh"})


@dataclass(frozen=True)
class PanelCallbackResult:
    toast: str = ""
    toast_type: str = "info"


class FeishuPanelController:
    def __init__(self, adapter: Any, service: Any):
        self.adapter = adapter
        self.service = service
        self._states: dict[str, PanelState] = {}
        self._active: dict[str, str] = {}
        self._messages: dict[str, str] = {}
        self._sources: dict[str, Any] = {}
        self._tasks: set[asyncio.Task] = set()
        self._senders: dict[str, asyncio.Task] = {}
        self._pending_cards: dict[str, dict] = {}
        self._loads: dict[tuple[str, str], asyncio.Task] = {}
        self._controls: dict[str, asyncio.Task] = {}
        self._executing: dict[str, int] = {}
        self._open_locks: dict[tuple, asyncio.Lock] = {}
        self._opening: dict[tuple, int] = {}
        self._closed = False
        self._loop = asyncio.get_running_loop()

    def _spawn(self, coro) -> asyncio.Task | None:
        if self._closed:
            coro.close()
            return None
        task = self._loop.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._completed)
        return task

    def _completed(self, task: asyncio.Task) -> None:
        self._tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            logger.error("Panel task failed", exc_info=task.exception())

    async def close(self) -> None:
        self._closed = True
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await self.service.close()
        self._states.clear()
        self._active.clear()
        self._messages.clear()
        self._sources.clear()
        self._pending_cards.clear()
        self._loads.clear()
        self._controls.clear()
        self._executing.clear()
        self._senders.clear()

    @staticmethod
    def _load_view_name(view: str) -> str:
        return _VIEW_ALIASES.get(view, "")

    def _is_active(self, state: PanelState) -> bool:
        return state.active and self._active.get(state.scope_key) == state.panel_id

    async def open(self, **kwargs):
        from gateway.platforms.base import SendResult
        task = self._spawn(self._open_panel(**kwargs))
        if task is None:
            return SendResult(success=False, error="Panel is closed")
        return await task

    async def _open_panel(self, *, source, session_key, owner_open_id, status_text, metadata, initial_view):
        key = (source.chat_id, source.thread_id, owner_open_id)
        lock = self._open_locks.setdefault(key, asyncio.Lock())
        self._opening[key] = self._opening.get(key, 0) + 1
        state = None
        attached = False
        try:
            async with lock:
                self._prune()
                state = await self.create_panel_state(source=source, session_key=session_key,
                    owner_open_id=owner_open_id, status_text=status_text, initial_view=initial_view)
                result = await self.adapter.send_coding_progress_card(source.chat_id, render_panel(state),
                                                                       metadata=metadata)
                if not result.success:
                    return result
                attached = self.attach_message_id(state, result.message_id)
                if not attached:
                    from gateway.platforms.base import SendResult
                    return SendResult(success=False, error="Card sent without a usable message ID")
                view = self._load_view_name(initial_view)
                if view and view != "home":
                    self.schedule_view_load(state.panel_id, view)
                return result
        finally:
            if state is not None and not attached:
                self.discard(state)
            self._opening[key] -= 1
            if not self._opening[key]:
                del self._opening[key]
                del self._open_locks[key]

    def _prune(self):
        """Retain active cards plus a small retired tail; never remove an owned task's state."""
        now = time.time()
        retired = []
        for state in self._states.values():
            if state.panel_id not in self._messages:
                continue  # a first send is still in progress
            if state.expires_at <= now:
                state.active = False
            if not state.active:
                retired.append(state)
        excess = max(0, len(retired) - 64)
        for state in retired:
            panel_id = state.panel_id
            if panel_id in self._senders or any(key[0] == panel_id for key in self._loads):
                continue
            if state.scope_key in self._controls or state.busy_action_id or state.panel_id in self._executing:
                continue
            if state.expires_at <= now or excess > 0:
                self.discard(state)
                excess -= 1

    async def create_panel_state(self, *, source, session_key: str, status_text: str = "",
                                 owner_open_id: str, initial_view: str = "home") -> PanelState:
        if self._closed:
            raise RuntimeError("Panel is closed")
        if not owner_open_id:
            raise ValueError("Panel owner open_id is required")
        data = await self.service.snapshot(source=source, session_key=session_key,
                                          status_text=status_text, include_catalog=False,
                                          include_sessions=False, include_status=False)
        if self._closed:
            raise RuntimeError("Panel is closed")
        data.update(loaded_views=["home"], loading_views=[], load_errors={})
        state = PanelState(panel_id=f"p_{uuid.uuid4().hex}", app_id=self.adapter._app_id,
                           owner_open_id=owner_open_id, chat_id=source.chat_id,
                           thread_id=source.thread_id or "", session_key=session_key,
                           profile=source.profile or "default", chat_type=source.chat_type, data=data)
        if initial_view in {"model", "reasoning", "sessions", "status"}:
            state.view, state.view_stack = initial_view, ["home"]
            if self._load_view_name(initial_view):
                self._mark_view_loading(state, initial_view)
        self._states[state.panel_id] = state
        self._sources[state.panel_id] = source
        return state

    def attach_message_id(self, state: PanelState, message_id: str) -> bool:
        """Publish only a successfully sent card; binding never changes its UI revision."""
        if self._closed or state.panel_id not in self._states or not message_id:
            return False
        self._messages[state.panel_id] = message_id
        previous_id = self._active.get(state.scope_key)
        self._active[state.scope_key] = state.panel_id
        if previous_id and previous_id != state.panel_id:
            previous = self._states[previous_id]
            previous.active, previous.lifecycle = False, "replaced"
            previous.revision += 1
            self._queue_card(previous)
        return True

    def discard(self, state: PanelState) -> None:
        self._states.pop(state.panel_id, None)
        self._sources.pop(state.panel_id, None)
        self._messages.pop(state.panel_id, None)
        self._pending_cards.pop(state.panel_id, None)
        if self._active.get(state.scope_key) == state.panel_id:
            del self._active[state.scope_key]

    def _queue_card(self, state: PanelState) -> None:
        if self._closed or state.panel_id not in self._messages:
            return
        self._pending_cards[state.panel_id] = render_panel(state)
        if state.panel_id not in self._senders:
            self._senders[state.panel_id] = self._spawn(self._send_cards(state.panel_id))

    async def _send_cards(self, panel_id: str) -> None:
        try:
            while panel_id in self._pending_cards:
                card = self._pending_cards.pop(panel_id)
                result = await self.adapter.patch_interactive_message(
                    message_id=self._messages[panel_id], card=card)
                if not result.success:
                    logger.warning("Panel card update failed: %s", result.error)
        finally:
            self._senders.pop(panel_id, None)
            self._prune()

    @staticmethod
    def _mark_view_loading(state: PanelState, view: str) -> None:
        view = _VIEW_ALIASES.get(view, view)
        state.data["loaded_views"] = [v for v in state.data["loaded_views"] if v != view]
        state.data["loading_views"] = [*state.data["loading_views"], view]
        state.data["load_errors"] = {k: v for k, v in state.data["load_errors"].items() if k != view}

    def schedule_view_load(self, panel_id: str, view: str) -> bool:
        view = self._load_view_name(view)
        if self._closed or not view:
            return False
        key = panel_id, view
        if key not in self._loads:
            self._loads[key] = self._spawn(self._load_view(panel_id, view))
        return True

    async def _load_view(self, panel_id: str, view: str) -> None:
        try:
            state = self._states[panel_id]
            snapshot = await self.service.snapshot(
                source=self._sources[panel_id], session_key=state.session_key, status_text="",
                include_catalog=view == "model", include_sessions=view == "sessions",
                include_status=view == "status")
            payload_keys = {"model": ("model_providers", "model_options"),
                            "sessions": ("sessions",), "status": ("status_text", "running")}
            payload = ({key: snapshot[key] for key in payload_keys[view]} if view in payload_keys else snapshot)
            error = ""
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            logger.warning("Panel view failed: %s", exc)
            payload, error = {}, "加载失败，请刷新重试"
        finally:
            self._loads.pop((panel_id, view), None)
        latest = self._states[panel_id]
        if not self._is_active(latest):
            return
        latest.data.update(payload)
        latest.data["loading_views"] = [v for v in latest.data["loading_views"] if v != view]
        if error:
            latest.data["load_errors"][view] = error
        else:
            latest.data["loaded_views"] = [*latest.data["loaded_views"], view]
            latest.data["load_errors"].pop(view, None)
        latest.revision += 1
        self._queue_card(latest)

    def handle_sync(self, raw_value: dict, *, open_id: str, chat_id: str, loop) -> PanelCallbackResult:
        """The SDK thread only posts work; all state access runs on the owning loop."""
        try:
            action = parse_panel_action(raw_value)
        except PanelActionError:
            return PanelCallbackResult("无效的面板操作", "error")
        if loop is not self._loop or self._closed:
            return PanelCallbackResult("面板已失效，请重新打开 /panel", "warning")
        try:
            running = asyncio.get_running_loop()
        except RuntimeError:
            running = None
        if running is self._loop:
            return self._handle(action, open_id, chat_id)
        result = concurrent.futures.Future()
        def dispatch():
            if not result.set_running_or_notify_cancel():
                return
            try:
                result.set_result(self._handle(action, open_id, chat_id))
            except Exception as exc:
                result.set_exception(exc)
        try:
            self._loop.call_soon_threadsafe(dispatch)
        except RuntimeError:
            return PanelCallbackResult("面板已失效，请重新打开 /panel", "warning")
        try:
            return result.result(timeout=2)
        except concurrent.futures.TimeoutError:
            result.cancel()
            return PanelCallbackResult("面板繁忙，请重试", "warning")

    def _handle(self, action: PanelAction, open_id: str, chat_id: str) -> PanelCallbackResult:
        state = self._states.get(action.panel_id)
        if self._closed or state is None or state.expires_at <= time.time():
            return PanelCallbackResult("面板已失效，请重新打开 /panel", "warning")
        if open_id != state.owner_open_id or chat_id != state.chat_id:
            return PanelCallbackResult("无权操作此面板", "error")
        if not self.adapter.is_control_panel_operator_authorized(self._sources[state.panel_id], open_id):
            return PanelCallbackResult("无权操作此面板", "error")
        if not self._is_active(state):
            return PanelCallbackResult("此面板已关闭或被替代，请使用最新 /panel", "warning")
        if action.nonce in state.handled_nonces:
            return PanelCallbackResult("该操作已经处理", "warning")
        if action.revision != state.revision and not (
            action.revision < state.revision and action.op in _REBASABLE_STALE_OPS):
            self._queue_card(state)
            return PanelCallbackResult("页面已更新，请重试", "warning")
        if action.op == "exec":
            if state.scope_key in self._controls and action.target != "stop":
                return PanelCallbackResult("已有控制操作正在处理", "warning")
            self._executing[state.panel_id] = self._executing.get(state.panel_id, 0) + 1
            if action.target == "stop":
                self._spawn(self._execute_control(state.panel_id, action, stop=True))
            else:
                state.busy_action_id = action.nonce
                self._controls[state.scope_key] = self._spawn(self._execute_control(state.panel_id, action))
            state.remember_nonce(action.nonce)
            state.revision += 1
            self._queue_card(state)
            return PanelCallbackResult("正在处理…")
        try:
            updated = self._navigate(state, action)
        except PanelActionError as exc:
            return PanelCallbackResult(str(exc), "error")
        updated.remember_nonce(action.nonce)
        updated.revision += 1
        self._states[state.panel_id] = updated
        self._clamp_page(updated)
        view = self._load_view_name(updated.view)
        if updated.active and view and view not in updated.data["loaded_views"]:
            if (state.panel_id, view) not in self._loads:
                self._mark_view_loading(updated, view)
                self.schedule_view_load(updated.panel_id, view)
        self._queue_card(updated)
        return PanelCallbackResult("面板已关闭" if action.op == "close" else "")

    def _navigate(self, state: PanelState, action: PanelAction) -> PanelState:
        if action.op == "refresh":
            updated = state.clone()
            view = self._load_view_name(updated.view) or "home"
            updated.data["loaded_views"] = [v for v in updated.data["loaded_views"] if v != view]
            return updated
        if action.op != "select":
            return reduce_panel_state(state, action)
        if action.index is None:
            raise PanelActionError("无效的选择")
        updated = state.clone()
        if action.target == "model_provider":
            providers = state.data.get("model_providers", [])
            if action.index >= len(providers):
                raise PanelActionError("供应商选择已失效")
            provider = providers[action.index]
            if not provider["model_indices"]:
                raise PanelActionError("该供应商没有可用模型")
            updated.filters["model_provider"] = provider["slug"]
            updated.view_stack.append(state.view)
            updated.view = "model_provider"
        elif action.target == "global_reasoning":
            if action.index >= len(state.data["reasoning_options"]):
                raise PanelActionError("选择已失效")
            updated.data["pending_global_reasoning_index"] = action.index
            updated.view_stack.append(state.view)
            updated.view = "confirm_global_reasoning"
        else:
            raise PanelActionError("无效的选择")
        updated.page = 0
        return updated

    async def _execute_control(self, panel_id: str, action: PanelAction, *, stop=False) -> None:
        state = self._states[panel_id]
        try:
            result = await self.service.execute(source=self._sources[panel_id],
                session_key=state.session_key, target=action.target, index=action.index,
                state_data=state.data)
            flash = f"{'✅' if result.success else '❌'} {result.text}"[:300]
            snapshot = await self.service.snapshot(source=self._sources[panel_id],
                session_key=state.session_key, status_text="", include_catalog=False,
                include_sessions=False, include_status=False)
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.exception("Panel control failed")
            snapshot, flash = {}, "❌ 操作失败，请刷新查看实际设置"
        finally:
            self._executing[panel_id] -= 1
            if not self._executing[panel_id]:
                del self._executing[panel_id]
            if not stop:
                self._controls.pop(state.scope_key, None)
                self._states[panel_id].busy_action_id = ""
        active_id = self._active.get(state.scope_key)
        latest = self._states.get(active_id)
        if latest is None or not self._is_active(latest):
            return
        latest.data.update(snapshot)
        latest.data["flash"] = flash
        latest.data.pop("pending_global_reasoning_index", None)
        invalidated = {"status", "sessions"} if action.target in {"new", "resume"} else {"status"}
        latest.data["loaded_views"] = [v for v in latest.data["loaded_views"] if v not in invalidated]
        if action.target in {"new", "resume", "preset"}:
            latest.view, latest.view_stack, latest.page = "home", [], 0
        elif action.target == "global_reasoning":
            latest.view, latest.view_stack, latest.page = "reasoning", ["home"], 0
        latest.revision += 1
        self._queue_card(latest)

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
