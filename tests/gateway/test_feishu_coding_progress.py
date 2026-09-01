import asyncio
import json
import queue
from types import SimpleNamespace

from typing import Any, cast

import pytest

from gateway.display_config import resolve_display_setting
from gateway.platforms.base import SendResult
from gateway.tool_progress_diff import build_edit_diff_summary, parse_unified_diff
from gateway.turn_context import TurnContext
from plugins.platforms.feishu.progress import render_progress_card


def test_feishu_coding_progress_config_normalizes_values():
    config = {
        "display": {
            "platforms": {
                "feishu": {
                    "tool_progress_style": "CARD",
                    "tool_edit_display": "diff",
                    "tool_diff_visibility": "all",
                    "tool_diff_max_lines": "42",
                }
            }
        }
    }
    assert resolve_display_setting(config, "feishu", "tool_progress_style") == "card"
    assert resolve_display_setting(config, "feishu", "tool_edit_display") == "diff"
    assert resolve_display_setting(config, "feishu", "tool_diff_visibility") == "all"
    assert resolve_display_setting(config, "feishu", "tool_diff_max_lines") == 42


def test_parse_unified_diff_classifies_files_and_counts_changes():
    diff = """--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+one
+two
--- a/old.py
+++ /dev/null
@@ -1 +0,0 @@
-gone
--- a/edit.py
+++ b/edit.py
@@ -1 +1 @@
-old
+new
"""
    summary = parse_unified_diff(diff, redact=False)
    assert summary is not None
    assert summary.total_files == 3
    assert summary.additions == 3
    assert summary.deletions == 2
    assert [(item.status, item.path) for item in summary.files] == [
        ("added", "new.py"),
        ("deleted", "old.py"),
        ("modified", "edit.py"),
    ]


def test_parse_unified_diff_caps_visible_files_lines_and_chars():
    diff = "".join(
        f"--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1 +1 @@\n-old-{i}\n+new-{i}\n"
        for i in range(4)
    )
    summary = parse_unified_diff(
        diff, redact=False, max_files=2, max_lines=3, max_chars=1000
    )
    assert summary is not None
    assert summary.total_files == 4
    assert len(summary.files) == 2
    assert summary.omitted_files == 2
    assert summary.omitted_lines > 0
    assert summary.truncated is True


def test_parse_unified_diff_redacts_secret_shaped_content():
    secret = "sk-" + "a" * 32
    summary = parse_unified_diff(
        f"--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-old\n+OPENAI_API_KEY={secret}\n"
    )
    assert summary is not None
    visible = "\n".join(line for file in summary.files for line in file.lines)
    assert secret not in visible
    assert "..." in visible


def test_build_edit_diff_summary_reuses_patch_result_diff(tmp_path):
    target = tmp_path / "demo.py"
    raw = json.dumps(
        {
            "success": True,
            "diff": (
                f"--- a/{target}\n+++ b/{target}\n"
                "@@ -1 +1 @@\n-old = 1\n+new = 2\n"
            ),
        }
    )
    summary = build_edit_diff_summary("patch", raw, function_args={"path": str(target)})
    assert summary is not None
    assert summary.additions == 1
    assert summary.deletions == 1
    assert summary.files[0].status == "modified"


def test_build_edit_diff_summary_marks_snapshot_backed_new_file_added(tmp_path):
    from agent.display import capture_local_edit_snapshot

    target = tmp_path / "new.py"
    snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})
    target.write_text("created = True\n", encoding="utf-8")
    summary = build_edit_diff_summary(
        "write_file",
        '{"verified": true}',
        function_args={"path": str(target)},
        snapshot=snapshot,
    )
    assert summary is not None
    assert summary.files[0].status == "added"
    assert summary.additions == 1


def test_progress_card_renders_terminal_status_and_diff():
    summary = parse_unified_diff(
        "--- a/app.py\n+++ b/app.py\n@@ -1 +1 @@\n-old\n+new\n",
        redact=False,
    )
    card = render_progress_card(
        [
            {
                "call_id": "a",
                "tool_name": "terminal",
                "args": {"command": "pytest -q"},
                "status": "success",
                "duration": 0.25,
                "exit_code": 0,
            },
            {
                "call_id": "b",
                "tool_name": "patch",
                "args": {"path": "app.py"},
                "status": "success",
                "duration": 0.1,
                "diff": summary,
            },
        ],
        finalized=True,
        edit_display="diff",
    )
    assert card["header"]["template"] == "green"
    content = card["elements"][0]["text"]["content"]
    assert "pytest -q" in content
    assert "exit 0" in content
    assert "app.py" in content
    assert "+1 -1" in content
    assert "```diff" in content


class _CardAdapter:
    def __init__(self):
        self.cards = []
        self.updates = []
        self.sent = []

    async def send_coding_progress_card(self, chat_id, card, *, reply_to=None, metadata=None):
        self.cards.append((chat_id, card, reply_to, metadata))
        return SendResult(success=True, message_id="om_progress")

    async def update_coding_progress_card(self, message_id, card):
        self.updates.append((message_id, card))
        return SendResult(success=True, message_id=message_id)

    async def send(self, chat_id, content, reply_to=None, metadata=None):
        self.sent.append((chat_id, content, reply_to, metadata))
        return SendResult(success=True, message_id="om_fallback")

    async def edit_message(self, chat_id, message_id, content):
        self.sent.append((chat_id, content, None, None))
        return SendResult(success=True, message_id=message_id)


@pytest.mark.asyncio
async def test_feishu_adapter_sends_coding_card_as_interactive_with_thread_data():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    adapter._client = object()
    captured = {}

    async def fake_send(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace()

    adapter._feishu_send_with_retry = fake_send
    adapter._finalize_send_result = lambda response, default_message: SendResult(
        success=True, message_id="om_card"
    )
    card = render_progress_card([], finalized=False)
    result = await adapter.send_coding_progress_card(
        "oc_test",
        card,
        reply_to="om_user",
        metadata={"thread_id": "omt_topic"},
    )

    assert result.success is True
    assert captured["msg_type"] == "interactive"
    assert captured["reply_to"] == "om_user"
    assert captured["metadata"] == {"thread_id": "omt_topic"}
    assert json.loads(captured["payload"])["header"]["template"] == "blue"


class _FailingCardAdapter(_CardAdapter):
    async def send_coding_progress_card(self, chat_id, card, *, reply_to=None, metadata=None):
        self.cards.append((chat_id, card, reply_to, metadata))
        return SendResult(success=False, error="interactive unavailable", retryable=False)


@pytest.mark.asyncio
async def test_native_feishu_sender_falls_back_to_editable_post():
    from gateway.run import TurnRunner

    adapter = _FailingCardAdapter()
    q = queue.Queue()
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="oc_test", chat_type="dm"),
        _run_still_current=lambda: True,
        progress_queue=q,
        _native_feishu_progress_card=True,
        _tool_edit_display="diff",
    )
    turn = TurnRunner(cast(Any, SimpleNamespace()), ctx)
    q.put(
        {
            "type": "tool.started",
            "tool_call_id": "call-a",
            "tool_name": "terminal",
            "args": {"command": "pytest -q"},
            "preview": "pytest -q",
        }
    )
    task = asyncio.create_task(turn._send_native_feishu_progress(adapter))
    await asyncio.sleep(0.25)
    task.cancel()
    await task

    assert len(adapter.cards) == 1
    assert adapter.sent
    assert "Hermes Coding Progress" in adapter.sent[0][1]
    assert "pytest -q" in adapter.sent[0][1]


@pytest.mark.asyncio
async def test_feishu_adapter_patches_coding_card_with_card_specific_api():
    from plugins.platforms.feishu.adapter import FeishuAdapter

    adapter = object.__new__(FeishuAdapter)
    patch_operation = object()
    adapter._client = SimpleNamespace(
        im=SimpleNamespace(
            v1=SimpleNamespace(message=SimpleNamespace(patch=patch_operation))
        )
    )
    captured = {}

    async def fake_run(func, *args):
        captured["operation"] = func
        captured["request"] = args[0]
        return object()

    adapter._run_blocking = fake_run
    adapter._finalize_send_result = lambda response, default_message: SendResult(
        success=True, message_id=None
    )
    card = render_progress_card(
        [{"tool_name": "read_file", "status": "success", "preview": "demo.py"}],
        finalized=True,
    )
    result = await adapter.update_coding_progress_card("om_123", card)

    assert result.success is True
    assert result.message_id == "om_123"
    assert captured["operation"] is patch_operation
    request = captured["request"]
    assert request.message_id == "om_123"
    payload = json.loads(request.request_body.content)
    assert payload["config"]["update_multi"] is True


@pytest.mark.asyncio
async def test_native_callbacks_correlate_ids_and_capture_write_file_diff(tmp_path):
    from gateway.run import TurnRunner

    target = tmp_path / "note.txt"
    target.write_text("old\n", encoding="utf-8")
    q = queue.Queue()
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="oc_test", chat_type="dm"),
        _run_still_current=lambda: True,
        progress_queue=q,
        _native_feishu_progress_card=True,
        _tool_edit_display="diff",
        _tool_diff_visibility="private",
    )
    turn = TurnRunner(cast(Any, SimpleNamespace()), ctx)

    turn.native_tool_start_callback(
        "call-write", "write_file", {"path": str(target), "content": "new\n"}
    )
    started = q.get_nowait()
    assert started["tool_call_id"] == "call-write"
    target.write_text("new\n", encoding="utf-8")
    turn.native_tool_complete_callback(
        "call-write", "write_file", {"path": str(target)}, '{"verified": true}'
    )
    completed = q.get_nowait()
    assert completed["tool_call_id"] == "call-write"
    assert completed["diff"] is not None
    assert completed["diff"].additions == 1
    assert completed["diff"].deletions == 1


@pytest.mark.asyncio
async def test_group_chat_diff_visibility_private_keeps_summary_but_hides_lines(tmp_path):
    from gateway.run import TurnRunner

    target = tmp_path / "group.py"
    result = json.dumps(
        {
            "success": True,
            "diff": (
                f"--- a/{target}\n+++ b/{target}\n"
                "@@ -1 +1 @@\n-secret_old\n+secret_new\n"
            ),
        }
    )
    q = queue.Queue()
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="oc_group", chat_type="group"),
        _run_still_current=lambda: True,
        progress_queue=q,
        _native_feishu_progress_card=True,
        _tool_edit_display="diff",
        _tool_diff_visibility="private",
    )
    turn = TurnRunner(cast(Any, SimpleNamespace()), ctx)
    turn.native_tool_start_callback("call-patch", "patch", {"path": str(target)})
    q.get_nowait()
    turn.native_tool_complete_callback(
        "call-patch", "patch", {"path": str(target)}, result
    )
    completed = q.get_nowait()
    assert completed["diff"] is not None
    assert completed["diff"].additions == 1
    assert completed["diff"].deletions == 1
    assert completed["diff"].files[0].lines == ()


@pytest.mark.asyncio
async def test_native_feishu_sender_accumulates_and_finalizes_one_card():
    from gateway.run import TurnRunner

    adapter = _CardAdapter()
    q = queue.Queue()
    ctx = TurnContext(
        source=SimpleNamespace(chat_id="oc_test", chat_type="dm"),
        _run_still_current=lambda: True,
        progress_queue=q,
        _native_feishu_progress_card=True,
        _tool_edit_display="summary",
        _tool_progress_max_items=8,
        _tool_progress_card_max_chars=7200,
    )
    ctx._progress_reply_to = "om_user"
    ctx._progress_metadata = {"thread_id": "omt_topic"}
    turn = TurnRunner(cast(Any, SimpleNamespace()), ctx)

    q.put(
        {
            "type": "tool.started",
            "tool_call_id": "call-a",
            "tool_name": "terminal",
            "args": {"command": "pytest -q"},
            "preview": "pytest -q",
        }
    )
    task = asyncio.create_task(turn._send_native_feishu_progress(adapter))
    await asyncio.sleep(0.25)
    q.put(
        {
            "type": "tool.completed",
            "tool_call_id": "call-a",
            "tool_name": "terminal",
            "is_error": False,
            "duration": 0.2,
            "exit_code": 0,
        }
    )
    await asyncio.sleep(0.2)
    task.cancel()
    await task

    assert len(adapter.cards) == 1
    assert adapter.cards[0][2] == "om_user"
    assert adapter.cards[0][3] == {"thread_id": "omt_topic"}
    assert adapter.updates
    final_card = adapter.updates[-1][1]
    assert final_card["header"]["template"] == "green"
    assert "exit 0" in final_card["elements"][0]["text"]["content"]
