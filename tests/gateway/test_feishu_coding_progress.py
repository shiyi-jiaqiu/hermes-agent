"""Native progress lifecycle, bounded display state and the redaction/privacy boundary."""
import asyncio
import json
import queue

import pytest

from gateway.config import Platform
from gateway.display_config import resolve_display_setting
from gateway.platforms.base import SendResult
from gateway.session import SessionSource
from plugins.platforms.feishu.progress.renderer import render_progress_card
from plugins.platforms.feishu.progress.sender import FeishuProgress, ProgressState, display_value


def content(card):
    return "\n".join(item["content"] for item in card["body"]["elements"] if item["tag"] == "markdown")


SOURCE = SessionSource(Platform.FEISHU, "oc_chat", chat_type="dm", user_id="ou_owner")
CONFIG = {"display": {"platforms": {"feishu": {"tool_progress_style": "card", "tool_edit_display": "diff",
                                              "tool_diff_visibility": "private"}}}}


def test_feishu_coding_progress_config_normalizes_values():
    config = {"display": {"platforms": {"feishu": {"tool_progress_style": "CARD", "tool_diff_max_lines": "42"}}}}
    assert resolve_display_setting(config, "feishu", "tool_progress_style") == "card"
    assert resolve_display_setting(config, "feishu", "tool_diff_max_lines") == 42
    assert resolve_display_setting({}, "feishu", "tool_progress_max_items") == 4


@pytest.mark.parametrize("turn_status,item_status,title", [
    ("working", "success", "Working"), ("finished", "success", "Completed"),
    ("finished", "running", "Interrupted"), ("interrupted", "running", "Interrupted"),
    ("failed", "error", "Failed"),
])
def test_turn_boundary_controls_completion_not_idle_tools(turn_status, item_status, title):
    card = render_progress_card([{"tool_name": "terminal", "status": item_status, "args": {"command": "echo test"}}],
                                turn_status=turn_status)
    assert card["header"]["title"]["content"].endswith(title)
    if title == "Interrupted":
        assert "Running" not in content(card)


def test_display_state_retains_only_active_calls_recent_history_and_totals():
    state = ProgressState(4)
    for index in range(10000):
        call_id = str(index)
        state.apply({"type": "tool.started", "tool_call_id": call_id, "tool_name": "read_file"})
        state.apply({"type": "tool.completed", "tool_call_id": call_id, "is_error": index == 0})
    assert len(state.recent) == 4 and not state.active
    assert (state.total, state.succeeded, state.failed, state.status) == (10000, 9999, 1, "working")
    state.apply({"type": "turn.finished", "status": "finished"})
    card = render_progress_card(state.visible(4), total_calls=state.total, error_count=state.failed,
                                turn_status=state.status)
    assert "9996 earlier" in content(card)
    assert card["header"]["title"]["content"].endswith("Completed with errors")


@pytest.mark.parametrize("chat_type,visible", [("dm", True), ("group", False)])
def test_callbacks_correlate_write_diff_and_enforce_group_visibility(tmp_path, chat_type, visible):
    source = SessionSource(Platform.FEISHU, "oc_chat", chat_type=chat_type, user_id="ou_owner")
    progress = FeishuProgress(None, source, CONFIG)
    events = queue.Queue()
    path = tmp_path / "new.py"
    args = {"path": str(path), "content": "print('hello')\n"}
    progress.start(events, "call", "write_file", args)
    path.write_text(args["content"])
    progress.complete(events, "call", "write_file", args, json.dumps({"success": True}))
    started, completed = events.get_nowait(), events.get_nowait()
    assert started["tool_call_id"] == completed["tool_call_id"] == "call"
    assert "content" not in started["args"] and "content" not in completed["args"]
    summary = completed["diff"]
    assert summary.files[0].status == "added" and summary.additions == 1
    assert bool(summary.files[0].lines) == visible
    assert completed["duration"] >= 0


def test_arguments_are_redacted_before_queueing_and_renderer_keeps_whole_fences():
    secret = "sk-" + "a" * 32
    command = f"curl 'https://user:password@example.test/v1?token={secret}' -H 'Authorization: Bearer {secret}'"
    progress, events = FeishuProgress(None, SOURCE, CONFIG), queue.Queue()
    progress.start(events, "terminal", "terminal", {"command": command, "workdir": "/tmp"})
    item = events.get_nowait()
    assert secret not in json.dumps(item)
    assert "user:password" not in json.dumps(item)
    assert secret not in json.dumps(display_value({"api_key": secret}))
    card = render_progress_card([{**item, "status": "running"}], max_chars=500)
    assert content(card).count("```") % 2 == 0
    assert secret not in content(card)


class Adapter:
    def __init__(self, after_send=None, fail=False):
        self.cards, self.updates, self.text = [], [], []
        self.after_send, self.fail = after_send, fail

    async def send_coding_progress_card(self, chat_id, card, **kwargs):
        self.cards.append((card, kwargs))
        if self.after_send:
            self.after_send()
        return SendResult(not self.fail, message_id="card" if not self.fail else None, error="unavailable" if self.fail else None)

    async def update_coding_progress_card(self, message_id, card):
        self.updates.append((message_id, card))
        return SendResult(True, message_id=message_id)

    async def send(self, **kwargs):
        self.text.append(kwargs)
        return SendResult(True, message_id="text")


@pytest.mark.asyncio
@pytest.mark.parametrize("fail", [False, True])
async def test_one_native_card_explicit_finish_and_no_text_fallback(fail):
    events = queue.Queue()
    delivered = []
    progress = None
    def complete():
        progress.complete(events, "call", "terminal", {"command": "echo done"}, '{"exit_code": 0, "output": "done"}')
        progress.finish(events, "finished")
    adapter = Adapter(complete, fail=fail)
    progress = FeishuProgress(adapter, SOURCE, CONFIG)
    progress.start(events, "call", "terminal", {"command": "echo done"})
    await progress.send_events(events, reply_to="message", metadata={"thread_id": "topic"}, on_delivery=delivered.append)
    assert len(adapter.cards) == 1 and not adapter.text
    assert adapter.cards[0][1] == {"reply_to": "message", "metadata": {"thread_id": "topic"}}
    if fail:
        assert not adapter.updates and not delivered
    else:
        assert len(delivered) == 1 and len(adapter.updates) == 1
        assert adapter.updates[0][0] == "card"
        assert adapter.updates[0][1]["header"]["title"]["content"].endswith("Completed")
    assert progress._closed and not progress._snapshots and not progress._started


@pytest.mark.asyncio
async def test_burst_processing_yields_while_preserving_all_completions():
    events = queue.Queue()
    for index in range(1000):
        events.put({"type": "tool.started", "tool_call_id": str(index), "tool_name": "read_file"})
        events.put({"type": "tool.completed", "tool_call_id": str(index), "tool_name": "read_file", "is_error": False})
    events.put({"type": "turn.finished", "status": "finished"})
    queued_at_yield = []
    adapter = Adapter(lambda: asyncio.get_running_loop().call_soon(lambda: queued_at_yield.append(events.qsize())))
    progress = FeishuProgress(adapter, SOURCE, CONFIG)
    await progress.send_events(events, reply_to=None, metadata=None, on_delivery=lambda result: None)
    assert queued_at_yield[0] > 0
    assert progress.state.total == progress.state.succeeded == 1000
    assert len(progress.state.recent) == 4


@pytest.mark.asyncio
async def test_redactor_failure_stops_only_presentation(monkeypatch):
    import plugins.platforms.feishu.progress.sender as sender
    monkeypatch.setattr(sender, "display_value", lambda value: (_ for _ in ()).throw(RuntimeError("redactor unavailable")))
    progress, events = FeishuProgress(Adapter(), SOURCE, CONFIG), queue.Queue()
    progress.start(events, "call", "terminal", {"command": "echo secret"})
    progress.complete(events, "call", "terminal", {}, "done")
    progress.finish(events, "finished")
    await progress.send_events(events, reply_to=None, metadata=None, on_delivery=lambda result: None)
    assert not progress.adapter.cards and progress._closed


@pytest.mark.asyncio
async def test_turn_cleanup_budget_releases_session_and_rejects_late_delivery(monkeypatch):
    from types import SimpleNamespace
    from gateway.run import GatewayRunner
    import gateway.run_turn as turn_module
    started, release = asyncio.Event(), asyncio.Event()
    class StalledAdapter:
        async def send_coding_progress_card(self, *args, **kwargs):
            started.set()
            while not release.is_set():
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    pass
            return SendResult(True, message_id='late')
    progress = FeishuProgress(StalledAdapter(), SOURCE, CONFIG)
    events = queue.Queue()
    progress.start(events, 'id', 'terminal', {'command': 'true'})
    delivered = []
    sender = asyncio.create_task(progress.send_events(events, reply_to=None, metadata=None, on_delivery=delivered.append))
    await started.wait()
    runner = object.__new__(GatewayRunner)
    runner._draining = False
    released = []
    runner._release_running_agent_state = lambda *a, **kw: released.append(kw)
    tracking = asyncio.create_task(asyncio.sleep(30))
    ctx = SimpleNamespace(native_progress=progress, progress_outcome='finished', progress_queue=events,
                          session_key='key', run_generation=1, stream_consumer_holder=[None],
                          streaming_tts_consumer_holder=[None])
    monkeypatch.setattr(turn_module, '_PROGRESS_FINALIZE_TIMEOUT', .01, raising=False)
    cleanup = asyncio.create_task(runner._run_agent_cleanup_turn_tasks(
        ctx, progress_task=sender, log_task=None, interrupt_monitor=None, _notify_task=None,
        tracking_task=tracking, stream_task=None))
    try:
        done, _ = await asyncio.wait({cleanup}, timeout=2)
        assert done and released
        assert progress._closed and sender in runner._background_tasks
    finally:
        release.set()
        await asyncio.gather(cleanup, sender, tracking, return_exceptions=True)
    assert not delivered and progress._message_id is None
