"""Current Feishu card transport and Panel/menu entry points."""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from types import SimpleNamespace

from gateway.platforms.base import MessageEvent, MessageType, SendResult

logger = logging.getLogger(__name__)


class FeishuCardsMixin:
    card_io_timeout = 5.0

    def create_tool_progress(self, source, config):
        from gateway.display_config import resolve_display_setting
        from .progress.sender import FeishuProgress
        if resolve_display_setting(config, source.platform.value, "tool_progress_style") == "card":
            return FeishuProgress(self, source, config)
        return None

    async def _card_io(self, operation):
        # Includes async token acquisition, thread lookup, retry backoff and delivery.
        # SDK async HTTP owns/closes its client on cancellation; there is no worker to join.
        try:
            async with asyncio.timeout(self.card_io_timeout):
                return await operation
        except TimeoutError as exc:
            raise TimeoutError(f"Card I/O exceeded its {self.card_io_timeout:g}s total budget") from exc

    async def _card_message_request(self, name, request):
        from lark_oapi import RequestOption
        from lark_oapi.api.auth.v3 import InternalTenantAccessTokenRequest, InternalTenantAccessTokenRequestBody
        # SDK async message methods still run synchronous token discovery unless an
        # explicit token is supplied. Acquire it asynchronously under the same budget.
        async with self._card_token_lock:
            if time.monotonic() >= self._card_token_expires:
                body = InternalTenantAccessTokenRequestBody.builder().app_id(self._app_id).app_secret(self._app_secret).build()
                token_request = InternalTenantAccessTokenRequest.builder().request_body(body).build()
                response = await self._client.auth.v3.tenant_access_token.ainternal(token_request)
                # This SDK response exposes token fields only in its raw JSON body.
                payload = json.loads(response.raw.content)
                if not response.success() or not payload.get("tenant_access_token"):
                    raise RuntimeError("Feishu card authentication failed")
                self._card_token = payload["tenant_access_token"]
                self._card_token_expires = time.monotonic() + max(0, payload["expire"] - 60)
            token = self._card_token
        option = RequestOption.builder().tenant_access_token(token).build()
        return await getattr(self._client.im.v1.message, "a" + name)(request, option)

    async def send_coding_progress_card(self, chat_id, card, *, reply_to=None, metadata=None):
        if not self._client:
            return SendResult(success=False, error="Not connected")
        try:
            response = await self._card_io(self._feishu_send_with_retry(
                chat_id=chat_id, msg_type="interactive", payload=json.dumps(card, ensure_ascii=False),
                reply_to=reply_to, metadata=metadata, message_call=self._card_message_request))
            return self._finalize_send_result(response, "interactive card send failed")
        except Exception as exc:
            logger.warning("Feishu card send failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def patch_interactive_message(self, *, message_id, card):
        from lark_oapi.api.im.v1 import PatchMessageRequest, PatchMessageRequestBody
        if not self._client:
            return SendResult(success=False, error="Not connected")
        try:
            body = PatchMessageRequestBody.builder().content(json.dumps(card, ensure_ascii=False)).build()
            request = PatchMessageRequest.builder().message_id(message_id).request_body(body).build()
            response = await self._card_io(self._card_message_request("patch", request))
            result = self._finalize_send_result(response, "interactive card update failed")
            if result.success:
                result.message_id = message_id
            return result
        except Exception as exc:
            logger.warning("Feishu card update failed: %s", exc)
            return SendResult(success=False, error=str(exc))

    async def update_coding_progress_card(self, message_id, card):
        return await self.patch_interactive_message(message_id=message_id, card=card)

    def is_control_panel_operator_authorized(self, source, open_id):
        if not self._is_interactive_operator_authorized(open_id):
            return False
        if self._is_sender_authorized(source.user_id, source.chat_type, source.chat_id,
                                      thread_id=source.thread_id) is not True:
            return False
        if source.chat_type == "dm":
            return True
        sender = SimpleNamespace(open_id=open_id, user_id=source.user_id)
        return self._allow_group_message(sender, source.chat_id, is_bot=False)

    async def open_control_panel(self, event, *, session_key, metadata, initial_view):
        owner = self.control_panel_owner_id(event)
        if not owner:
            return SendResult(success=False, error="Feishu operator identity is missing")
        return await self.send_control_panel(chat_id=event.source.chat_id, session_key=session_key,
            source=event.source, owner_open_id=owner, initial_view=initial_view, metadata=metadata)

    async def send_control_panel(self, *, chat_id, session_key, source, owner_open_id,
                                 status_text="", metadata=None, initial_view="home"):
        from gateway.control import HermesPanelControlService
        from .panel import FeishuPanelController
        if self._panel_controller is None:
            self._panel_controller = FeishuPanelController(self, HermesPanelControlService(self.gateway_runner))
        return await self._panel_controller.open(
            source=source, session_key=session_key, owner_open_id=owner_open_id,
            status_text=status_text, metadata=metadata, initial_view=initial_view)

    @staticmethod
    def control_panel_owner_id(event: MessageEvent) -> str:
        raw_event = getattr(event.raw_message, "event", None)
        sender_id = getattr(getattr(raw_event, "sender", None), "sender_id", None)
        open_id = str(getattr(sender_id, "open_id", "") or "")
        # Menu events already carry the authorized operator as source.user_id.
        return open_id or (event.source.user_id if event.source.user_id.startswith("ou_") else "")

    @staticmethod
    def _build_panel_callback_response(result):
        from lark_oapi.event.callback.model.p2_card_action_trigger import CallBackToast, P2CardActionTriggerResponse
        response = P2CardActionTriggerResponse()
        if result.toast:
            toast = CallBackToast()
            toast.type, toast.content = result.toast_type, result.toast
            response.toast = toast
        return response

    @staticmethod
    def _serialize_card_action_response(response):
        from lark_oapi.core.json import JSON
        return json.loads(JSON.marshal(response)) if response is not None else {"code": 0, "msg": "ok"}

    def _on_bot_menu_event(self, data):
        if self._loop_accepts_callbacks(self._loop):
            self._submit_on_loop(self._loop, self._handle_bot_menu_event(data))

    async def _handle_bot_menu_event(self, data):
        event = data.event
        command = self._menu_routes.get(event.event_key)
        if not command:
            return
        open_id = event.operator.operator_id.open_id
        if not self._is_interactive_operator_authorized(open_id):
            return
        if not self._menu_default_chat_id:
            logger.error("menu_default_chat_id is required for bot menu routing")
            return
        await self._route_menu_command(chat_id=self._menu_default_chat_id, open_id=open_id,
                                       command=command, raw_message=data, thread_id=None,
                                       chat_type_hint="p2p")

    async def _route_menu_command(self, *, chat_id, open_id, command, raw_message,
                                  thread_id, chat_type_hint="group"):
        sender_id = SimpleNamespace(open_id=open_id, user_id=None, union_id=None)
        chat_info = await self.get_chat_info(chat_id)
        chat_type = self._resolve_source_chat_type(chat_info=chat_info, event_chat_type=chat_type_hint)
        if chat_type != "dm" and not self._allow_group_message(sender_id, chat_id, is_bot=False):
            return
        sender = await self._resolve_sender_profile(sender_id)
        source = self.build_source(chat_id=chat_id, chat_name=chat_info.get("name") or chat_id,
            chat_type=chat_type, user_id=sender["user_id"], user_name=sender["user_name"],
            thread_id=thread_id, user_id_alt=sender["user_id_alt"])
        event = MessageEvent(text=command, message_type=MessageType.COMMAND, source=source,
            raw_message=raw_message, message_id="", channel_prompt=self._resolve_channel_prompt(chat_id),
            timestamp=datetime.now())
        await self._handle_message_with_guards(event)
