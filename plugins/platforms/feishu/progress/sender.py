"""One native card per turn, bounded display history and explicit finalization."""
from __future__ import annotations

import asyncio
from collections import deque
import json
import logging
import queue
import time

from agent.display import build_tool_preview, capture_local_edit_snapshot, _detect_tool_failure
from agent.redact import redact_sensitive_text
from gateway.display_config import resolve_display_setting
from gateway.tool_progress_diff import build_edit_diff_summary
from .renderer import TOOL_ARGUMENT_KEYS, render_progress_card

logger = logging.getLogger(__name__)


def display_value(value):
    """Redact with key context intact, then bound values before queueing them."""
    serialized = json.dumps(value, ensure_ascii=False)
    safe = json.loads(redact_sensitive_text(serialized, force=True, redact_url_credentials=True))
    budget = [6000]
    def bounded(item):
        if isinstance(item, dict):
            return {k: bounded(v) for k, v in list(item.items())[:16] if budget[0] > 0}
        if isinstance(item, list):
            return [bounded(v) for v in item[:8] if budget[0] > 0]
        if isinstance(item, str):
            limit = min(1200, budget[0])
            budget[0] -= min(limit, len(item))
            return item if len(item) <= limit else item[:limit] + "…"
        return item
    return bounded(safe)


def display_arguments(tool_name, args):
    keys = TOOL_ARGUMENT_KEYS.get(tool_name, ())
    return display_value({key: args[key] for key in keys if key in args})


class ProgressState:
    def __init__(self, limit: int):
        self.active = {}
        self.recent = deque(maxlen=limit)
        self.total = self.succeeded = self.failed = 0
        self.status = "working"

    def apply(self, event):
        kind = event["type"]
        if kind == "turn.finished":
            self.status = event["status"]
            if self.active:
                self.status = "interrupted" if self.status == "finished" else self.status
                for item in self.active.values():
                    self.recent.append({**item, "status": "interrupted"})
                self.active.clear()
            return
        call_id = event["tool_call_id"]
        if kind == "tool.started":
            if call_id not in self.active:
                self.total += 1
            self.active[call_id] = {**event, "call_id": call_id, "status": "running"}
        elif kind == "tool.completed":
            item = self.active.pop(call_id, None)
            if item is None:
                self.total += 1
                item = {"call_id": call_id}
            failed = event["is_error"]
            self.failed += int(failed)
            self.succeeded += int(not failed)
            self.recent.append({**item, **event, "status": "error" if failed else "success"})

    def visible(self, limit):
        return [*self.recent, *self.active.values()][-limit:]


class FeishuProgress:
    def __init__(self, adapter, source, config):
        self.adapter, self.source = adapter, source
        setting = lambda name: resolve_display_setting(config, source.platform.value, name)
        self.edit_display = setting("tool_edit_display")
        self.diff_visibility = setting("tool_diff_visibility")
        self.max_files = setting("tool_diff_max_files")
        self.max_lines = setting("tool_diff_max_lines")
        self.max_diff_chars = setting("tool_diff_max_chars")
        self.max_items = max(1, setting("tool_progress_max_items"))
        self.max_chars = setting("tool_progress_card_max_chars")
        self.state = ProgressState(self.max_items)
        self._started = {}
        self._snapshots = {}
        self._closed = False
        self._message_id = None
        self._disabled = False
        self._finish_sent = False

    def _disable(self, events):
        self._disabled = True
        self._started.clear()
        self._snapshots.clear()
        events.put({"type": "display.failed"})
        logger.exception("Native progress presentation disabled for this turn")

    def start(self, events, call_id, tool_name, args):
        if self._closed or self._disabled or self._finish_sent:
            return
        try:
            self._started[call_id] = time.monotonic()
            if self.edit_display != "off":
                snapshot = capture_local_edit_snapshot(tool_name, args)
                if snapshot is not None:
                    self._snapshots[call_id] = snapshot
            safe_args = display_arguments(tool_name, args)
            events.put({"type": "tool.started", "tool_call_id": call_id, "tool_name": tool_name,
                        "args": safe_args, "preview": build_tool_preview(tool_name, safe_args, max_len=1000) or ""})
        except Exception:
            self._disable(events)

    def complete(self, events, call_id, tool_name, args, result):
        if self._closed or self._disabled or self._finish_sent:
            return
        try:
            self._complete(events, call_id, tool_name, args, result)
        except Exception:
            self._disable(events)

    def _complete(self, events, call_id, tool_name, args, result):
        started = self._started.pop(call_id, None)
        snapshot = self._snapshots.pop(call_id, None)
        is_error, error = _detect_tool_failure(tool_name, result)
        exit_code = None
        if tool_name == "terminal":
            try:
                parsed = json.loads(result)
                if isinstance(parsed, dict):
                    exit_code = parsed.get("exit_code")
            except (ValueError, TypeError):
                pass
        diff = None
        if self.edit_display != "off" and not is_error:
            include_body = self.diff_visibility == "all" or self.source.chat_type in {"dm", "p2p", "private", "direct"}
            try:
                diff = build_edit_diff_summary(tool_name, result, function_args=args, snapshot=snapshot,
                    max_files=self.max_files, max_lines=self.max_lines if include_body else 0,
                    max_chars=self.max_diff_chars if include_body else 0)
            except Exception:
                logger.exception("Tool diff presentation failed")
                error = "Diff unavailable"
        events.put({"type": "tool.completed", "tool_call_id": call_id, "tool_name": tool_name,
                    "args": display_arguments(tool_name, args), "is_error": is_error, "error": display_value(error),
                    "duration": max(0, time.monotonic() - started) if started is not None else None,
                    "exit_code": exit_code, "diff": diff})

    def finish(self, events, status):
        # The outer turn owns this boundary, including watchdog-abandoned workers.
        if self._finish_sent:
            return
        self._finish_sent = True
        events.put({"type": "turn.finished", "status": status})

    async def send_events(self, events, *, reply_to, metadata, on_delivery):
        last_attempt = 0.0
        dirty = False
        failed = False
        try:
            while self.state.status == "working":
                for _ in range(64):
                    try:
                        event = events.get_nowait()
                    except queue.Empty:
                        break
                    if isinstance(event, str):
                        await self.adapter.send(chat_id=self.source.chat_id, content=event,
                                                reply_to=reply_to, metadata=metadata)
                    elif isinstance(event, dict) and event.get("type") == "display.failed":
                        failed = True
                    elif isinstance(event, dict) and event.get("type") in {"tool.started", "tool.completed", "turn.finished"}:
                        self.state.apply(event)
                        dirty = True
                    # __reset__ is an interim-answer boundary, not a finished turn.
                    if self.state.status != "working":
                        break
                now = time.monotonic()
                finished = self.state.status != "working"
                if dirty and self.state.total and not failed and (finished or now - last_attempt >= 1):
                    last_attempt = now
                    card = render_progress_card(self.state.visible(self.max_items), turn_status=self.state.status,
                        total_calls=self.state.total, error_count=self.state.failed, edit_display=self.edit_display,
                        max_items=self.max_items, max_chars=self.max_chars)
                    if self._message_id:
                        result = await self.adapter.update_coding_progress_card(self._message_id, card)
                    else:
                        result = await self.adapter.send_coding_progress_card(self.source.chat_id, card,
                                                                             reply_to=reply_to, metadata=metadata)
                    if result.success and (self._message_id or result.message_id):
                        if self._message_id is None:
                            self._message_id = result.message_id
                            on_delivery(result)
                        dirty = False
                    else:
                        logger.warning("Native progress stopped for this turn: %s", result.error)
                        failed = True
                        self._disabled = True
                        self._started.clear()
                        self._snapshots.clear()
                await asyncio.sleep(0 if not events.empty() else 0.1)
        except asyncio.CancelledError:
            raise
        except Exception:
            # Presentation failures must never prevent tools or the final answer.
            logger.exception("Native progress sender failed")
        finally:
            self._closed = True
            self._started.clear()
            self._snapshots.clear()
