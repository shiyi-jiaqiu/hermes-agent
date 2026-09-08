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
async def test_native_send_and_patch_use_interactive_card_sdk_contract(adapter, monkeypatch):
    from lark_oapi.core.http import Transport
    from lark_oapi.core.model import RawResponse
    adapter._client = module._build_lark_client('test-app', 'test-secret', 'https://sdk.test')
    calls = []
    async def exchange(config, request, option=None):
        calls.append(request)
        assert config.enable_set_token and config.timeout is not None
        response = RawResponse()
        if 'tenant_access_token' in request.uri:
            payload = {'code': 0, 'tenant_access_token': 'test-token', 'expire': 7200}
        else:
            assert option.tenant_access_token == 'test-token'
            payload = {'code': 0, 'data': {'message_id': 'card'}}
        response.content = json.dumps(payload).encode()
        return response
    def no_sync(*args, **kwargs):
        pytest.fail('Native card I/O must not use synchronous authentication or HTTP')
    monkeypatch.setattr(Transport, 'aexecute', exchange)
    monkeypatch.setattr(Transport, 'execute', no_sync)
    card = render_progress_card([], turn_status='working')
    sent = await adapter.send_coding_progress_card('chat', card, reply_to='anchor', metadata={'thread_id': 'topic'})
    assert sent.success
    reply = calls[-1]
    assert reply.request_body.msg_type == 'interactive' and reply.message_id == 'anchor'
    assert reply.request_body.reply_in_thread is True
    assert json.loads(reply.request_body.content)['schema'] == '2.0'
    updated = await adapter.update_coding_progress_card('card', card)
    request = calls[-1]
    assert isinstance(request, PatchMessageRequest)
    assert request.message_id == updated.message_id == 'card'
    assert json.loads(request.request_body.content)['config']['update_multi'] is True
    assert len(calls) == 3  # token + reply + patch; token reused on this adapter


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
async def test_card_cancellation_stops_io_without_waiting_for_remote_release(adapter):
    started, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()
    async def request():
        started.set()
        try:
            await release.wait()
        finally:
            cancelled.set()
    task = asyncio.create_task(adapter._card_io(request()))
    await started.wait()
    task.cancel()
    try:
        done, _ = await asyncio.wait({task}, timeout=2)
        assert done and cancelled.is_set()
    finally:
        release.set()
        await asyncio.gather(task, return_exceptions=True)


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
    adapter._client = object()
    adapter._card_message_request = AsyncMock(return_value=object())
    adapter._finalize_send_result = lambda *_: SendResult(True)
    result = await adapter.patch_interactive_message(message_id="card", card={"schema": "2.0"})
    assert result.success and result.message_id == "card"


@pytest.mark.asyncio
async def test_card_total_budget_closes_real_stalled_http_connection(adapter, monkeypatch):
    """Exercise SDK auth and message HTTP over a local socket, including cancellation cleanup."""
    monkeypatch.setenv('NO_PROXY', '127.0.0.1')
    pending = set()
    requested, disconnected = asyncio.Event(), asyncio.Event()
    async def serve(reader, writer):
        pending.add(asyncio.current_task())
        try:
            header = await reader.readuntil(b'\r\n\r\n')
            fields = dict(line.split(b':', 1) for line in header.split(b'\r\n')[1:] if b':' in line)
            size = int(fields.get(b'Content-Length', fields.get(b'content-length', b'0')))
            await reader.readexactly(size)
            if b'tenant_access_token' in header.split(b'\r\n')[0]:
                body = json.dumps({'code': 0, 'tenant_access_token': 'test-token', 'expire': 7200}).encode()
                writer.write(b'HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nConnection: close\r\nContent-Length: '
                             + str(len(body)).encode() + b'\r\n\r\n' + body)
                await writer.drain()
            else:
                requested.set()
                assert await reader.read() == b''
                disconnected.set()
        finally:
            writer.close()
            await writer.wait_closed()
            pending.discard(asyncio.current_task())
    server = await asyncio.start_server(serve, '127.0.0.1', 0)
    adapter._client = module._build_lark_client('test-app', 'test-secret',
        f'http://127.0.0.1:{server.sockets[0].getsockname()[1]}')
    adapter.card_io_timeout = 2
    operation = asyncio.create_task(adapter.patch_interactive_message(message_id='card', card={'schema': '2.0'}))
    try:
        await asyncio.wait_for(requested.wait(), 5)
        result = await asyncio.wait_for(operation, 5)
        assert not result.success
        await asyncio.wait_for(disconnected.wait(), 5)
    finally:
        operation.cancel()
        await asyncio.gather(operation, return_exceptions=True)
        server.close()
        await server.wait_closed()
        for task in pending:
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True)
