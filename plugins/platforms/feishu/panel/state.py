"""Process-local presentation state for a Feishu control panel."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, replace
from typing import Any


@dataclass
class PanelState:
    panel_id: str
    app_id: str
    owner_open_id: str
    chat_id: str
    thread_id: str
    session_key: str
    profile: str = "default"
    chat_type: str = "group"
    view: str = "home"
    view_stack: list[str] = field(default_factory=list)
    page: int = 0
    filters: dict[str, str] = field(default_factory=dict)
    revision: int = 0
    busy_action_id: str = ""
    active: bool = True
    lifecycle: str = "active"
    data: dict[str, Any] = field(default_factory=dict)
    handled_nonces: list[str] = field(default_factory=list)
    expires_at: float = field(default_factory=lambda: time.time() + 7 * 24 * 60 * 60)

    @property
    def scope_key(self) -> str:
        return json.dumps(
            [self.app_id, self.chat_id, self.thread_id, self.owner_open_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def clone(self) -> "PanelState":
        # View payloads are immutable snapshots. Navigation copies only UI fields,
        # never the shared provider inventory or the session's business state.
        return replace(self, view_stack=list(self.view_stack), filters=dict(self.filters),
                       data=dict(self.data), handled_nonces=list(self.handled_nonces))

    def remember_nonce(self, nonce: str) -> None:
        if nonce and nonce not in self.handled_nonces:
            self.handled_nonces.append(nonce)
            del self.handled_nonces[:-64]
