"""Contracts against the installed Feishu SDK; no SDK source rewriting or text fallback."""
import asyncio
import base64
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

pytest.importorskip("lark_oapi")

from lark_oapi.api.im.v1 import PatchMessageRequest
from lark_oapi.ws import Client
from lark_oapi.ws.const import HEADER_MESSAGE_ID, HEADER_SEQ, HEADER_SUM, HEADER_TYPE
from lark_oapi.ws.enum import FrameType, MessageType
from lark_oapi.ws.pb.pbbp2_pb2 import Frame
from gateway.config import PlatformConfig
from gateway.platforms.base import SendResult
from plugins.platforms.feishu import adapter as module
from plugins.platforms.feishu.adapter_websocket import FeishuCardClient
from plugins.platforms.feishu.panel.controller import PanelCallbackResult
from plugins.platforms.feishu.progress.renderer import render_progress_card


@pytest.fixture
def adapter():
    assert module._load_lark_oapi()
    return module.FeishuAdapter(PlatformConfig())


@pytest.mark.asyncio
async def test_native_send_and_patch_use_interactive_card_sdk_contract(adapter):
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(patch=object()))))
    adapter._feishu_send_with_retry = AsyncMock(return_value=object())
    adapter._run_blocking = AsyncMock(return_value=object())
    adapter._finalize_send_result = lambda response, default_message: SendResult(True, message_id="card")
    card = render_progress_card([], turn_status="working")
    sent = await adapter.send_coding_progress_card("chat", card, reply_to="anchor", metadata={"thread_id": "topic"})
    assert sent.success
    sent_args = adapter._feishu_send_with_retry.await_args.kwargs
    assert sent_args["msg_type"] == "interactive" and sent_args["reply_to"] == "anchor"
    assert sent_args["metadata"] == {"thread_id": "topic"}
    assert json.loads(sent_args["payload"])["schema"] == "2.0"
    updated = await adapter.update_coding_progress_card("card", card)
    request = adapter._run_blocking.await_args.args[1]
    assert isinstance(request, PatchMessageRequest)
    assert request.message_id == updated.message_id == "card"
    assert json.loads(request.request_body.content)["config"]["update_multi"] is True


@pytest.mark.asyncio
@pytest.mark.parametrize("namespace_value", [False, True])
async def test_webhook_and_sdk_values_normalize_once_and_return_toast_only(adapter, namespace_value):
    adapter._loop = asyncio.get_running_loop()
    captured = []
    def handle(value, **kwargs):
        captured.append((value, kwargs))
        return PanelCallbackResult("accepted", "info")
    adapter._panel_controller = SimpleNamespace(handle_sync=handle)
    value = {"panel_action": True, "v": 1, "panel": "id", "rev": 0, "nonce": "nonce", "op": "home"}
    event = SimpleNamespace(operator=SimpleNamespace(open_id="ou_owner"), context=SimpleNamespace(open_chat_id="oc_chat"),
                            action=SimpleNamespace(value=SimpleNamespace(**value) if namespace_value else value))
    response = adapter._on_card_action_trigger(SimpleNamespace(event=event))
    serialized = adapter._serialize_card_action_response(response)
    assert captured[0][0] == value
    assert captured[0][1]["open_id"] == "ou_owner"
    assert serialized == {"toast": {"type": "info", "content": "accepted"}}


def frame(payload, index, count, kind=MessageType.CARD):
    result = Frame(SeqID=index, LogID=0, service=1, method=FrameType.DATA.value, payload=payload)
    for key, value in ((HEADER_TYPE, kind.value), (HEADER_MESSAGE_ID, "event"), (HEADER_SEQ, str(index)), (HEADER_SUM, str(count))):
        header = result.headers.add()
        header.key, header.value = key, value
    return result


@pytest.mark.asyncio
async def test_card_fragments_dispatch_once_with_native_ack_and_noncard_delegation(monkeypatch):
    module._load_lark_oapi()
    calls = []
    response = module.FeishuAdapter._build_panel_callback_response(PanelCallbackResult("accepted"))
    handler = SimpleNamespace(_do_without_validation=lambda payload: (calls.append(payload), response)[1])
    client = FeishuCardClient("app", "secret", event_handler=handler)
    writes = []
    async def write(data):
        writes.append(data)
    monkeypatch.setattr(client, "_write_message", write)
    payload = b'{"card":"payload"}'
    await client._handle_data_frame(frame(payload[:7], 0, 2))
    assert not calls and not writes
    await client._handle_data_frame(frame(payload[7:], 1, 2))
    assert calls == [payload] and len(writes) == 1
    ack_frame = Frame()
    ack_frame.ParseFromString(writes[0])
    ack = json.loads(ack_frame.payload)
    assert ack["code"] == 200
    assert json.loads(base64.b64decode(ack["data"]))["toast"]["content"] == "accepted"
    upstream = AsyncMock()
    monkeypatch.setattr(Client, "_handle_data_frame", upstream)
    event = frame(b"{}", 0, 1, MessageType.EVENT)
    await client._handle_data_frame(event)
    upstream.assert_awaited_once_with(event)


@pytest.mark.asyncio
async def test_card_cancellation_joins_owned_io_before_returning(adapter):
    started, release = asyncio.Event(), asyncio.Event()
    async def request():
        started.set()
        await release.wait()
        return "delivered"
    task = asyncio.create_task(adapter._card_io(request()))
    await started.wait()
    task.cancel()
    await asyncio.sleep(0)
    assert not task.done()
    release.set()
    with pytest.raises(asyncio.CancelledError):
        await task


@pytest.mark.asyncio
async def test_profile_namespaced_plugin_uses_current_sdk_without_canonical_module_state():
    from pathlib import Path
    from hermes_cli.plugins_loader import PluginLoaderMixin
    from hermes_cli.plugins_manifest import PluginManifest
    plugin = PluginLoaderMixin()._load_directory_module(
        PluginManifest("feishu", path=str(Path(module.__file__).parent)), module_name="hermes_plugins.feishu_transport_test")
    loaded = plugin.adapter
    assert loaded._load_lark_oapi()
    adapter = loaded.FeishuAdapter(PlatformConfig())
    response = adapter._build_panel_callback_response(PanelCallbackResult("accepted"))
    assert adapter._serialize_card_action_response(response)["toast"]["content"] == "accepted"
    adapter._client = SimpleNamespace(im=SimpleNamespace(v1=SimpleNamespace(message=SimpleNamespace(patch=object()))))
    adapter._run_blocking = AsyncMock(return_value=object())
    adapter._finalize_send_result = lambda *_: SendResult(True)
    result = await adapter.patch_interactive_message(message_id="card", card={"schema": "2.0"})
    assert result.success and result.message_id == "card"
