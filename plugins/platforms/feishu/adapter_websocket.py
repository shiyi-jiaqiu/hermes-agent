"""Feishu CARD frames use the same callback/ack protocol as EVENT frames.

The current SDK ignores CARD frames. Keep that transport extension on our own
client class; never alter the installed SDK or another adapter's client class.
"""
import base64
import http
import logging
import time

from lark_oapi.core.const import UTF_8
from lark_oapi.core.json import JSON
from lark_oapi.ws import Client
from lark_oapi.ws.client import _get_by_key
from lark_oapi.ws.const import HEADER_BIZ_RT, HEADER_MESSAGE_ID, HEADER_SEQ, HEADER_SUM, HEADER_TYPE
from lark_oapi.ws.enum import MessageType
from lark_oapi.ws.model import Response

logger = logging.getLogger(__name__)


class FeishuCardClient(Client):
    async def _handle_data_frame(self, frame):
        if _get_by_key(frame.headers, HEADER_TYPE) != MessageType.CARD.value:
            return await super()._handle_data_frame(frame)
        headers = frame.headers
        message_id = _get_by_key(headers, HEADER_MESSAGE_ID)
        total = int(_get_by_key(headers, HEADER_SUM))
        payload = frame.payload
        if total > 1:
            payload = self._combine(message_id, total, int(_get_by_key(headers, HEADER_SEQ)), payload)
            if payload is None:
                return
        response = Response(code=http.HTTPStatus.OK)
        try:
            started = time.monotonic()
            result = self._event_handler._do_without_validation(payload)
            header = headers.add()
            header.key, header.value = HEADER_BIZ_RT, str(int((time.monotonic() - started) * 1000))
            if result is not None:
                response.data = base64.b64encode(JSON.marshal(result).encode(UTF_8))
        except Exception:
            logger.exception("Feishu CARD callback failed")
            response = Response(code=http.HTTPStatus.INTERNAL_SERVER_ERROR)
        frame.payload = JSON.marshal(response).encode(UTF_8)
        await self._write_message(frame.SerializeToString())
