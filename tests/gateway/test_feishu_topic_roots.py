"""Topic creation must not become an AI turn; replies and commands still dispatch."""
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from gateway.config import PlatformConfig
from gateway.platforms.event import MessageType
from plugins.platforms.feishu.adapter import FeishuAdapter


@pytest.fixture
def adapter():
    instance = FeishuAdapter(PlatformConfig(extra={"ignore_topic_roots": True}))
    instance.get_chat_info = AsyncMock(return_value={"name": "Workspace", "chat_mode": "topic"})
    instance._fetch_message_text = AsyncMock(return_value="Topic title")
    instance._resolve_sender_profile = AsyncMock(return_value={
        "user_id": "ou_user", "user_name": "User", "user_id_alt": None})
    instance._dispatch_inbound_event = AsyncMock()
    return instance


async def process(adapter, *, text="Topic title", root_id=None, parent_id=None,
                  upper_message_id=None, chat_type="group", post=False):
    content = ({"zh_cn": {"title": text, "content": []}} if post else {"text": text})
    message = SimpleNamespace(message_id="om_current", chat_id="oc_chat",
        message_type="post" if post else "text", content=json.dumps(content), mentions=[],
        root_id=root_id, parent_id=parent_id, upper_message_id=upper_message_id, thread_id="omt_topic")
    await adapter._process_inbound_message(data=message, message=message,
        sender_id=SimpleNamespace(open_id="ou_user"), chat_type=chat_type, message_id=message.message_id)


@pytest.mark.asyncio
@pytest.mark.parametrize("post,root_id", [(False, None), (True, None), (True, "om_current")])
async def test_topic_title_does_not_reach_ai_dispatch(adapter, post, root_id):
    await process(adapter, post=post, root_id=root_id)
    adapter._dispatch_inbound_event.assert_not_awaited()
    adapter._resolve_sender_profile.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("link", ["root_id", "parent_id", "upper_message_id"])
async def test_topic_reply_still_dispatches_without_mention(adapter, link):
    await process(adapter, text="Please investigate", **{link: "om_root"})
    event = adapter._dispatch_inbound_event.call_args.args[0]
    assert event.text == "Please investigate"
    assert event.source.thread_id == "omt_topic"
    assert event.reply_to_message_id == "om_root"


@pytest.mark.asyncio
async def test_panel_command_at_topic_root_still_dispatches(adapter):
    await process(adapter, text="/panel")
    event = adapter._dispatch_inbound_event.call_args.args[0]
    assert event.message_type == MessageType.COMMAND
    assert event.text == "/panel"


@pytest.mark.asyncio
@pytest.mark.parametrize("chat_mode,chat_type", [("group", "group"), ("topic", "p2p"), (None, "group")])
async def test_ordinary_or_unknown_chat_is_not_silently_dropped(adapter, chat_mode, chat_type):
    adapter.get_chat_info.return_value["chat_mode"] = chat_mode
    await process(adapter, chat_type=chat_type)
    adapter._dispatch_inbound_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_existing_topic_behavior_is_retained_without_opt_in(adapter):
    adapter._apply_settings(adapter._load_settings({}))
    await process(adapter)
    adapter._dispatch_inbound_event.assert_awaited_once()


@pytest.mark.asyncio
async def test_chat_lookup_preserves_and_caches_topic_mode():
    adapter = FeishuAdapter(PlatformConfig())
    adapter._client = Mock()
    adapter._build_get_chat_request = Mock(return_value=object())
    adapter._run_blocking = AsyncMock(return_value=SimpleNamespace(success=lambda: True,
        data=SimpleNamespace(name="Workspace", chat_type="private", chat_mode="topic")))
    assert (await adapter.get_chat_info("oc_chat"))["chat_mode"] == "topic"
    assert (await adapter.get_chat_info("oc_chat"))["chat_mode"] == "topic"
    adapter._run_blocking.assert_awaited_once()
