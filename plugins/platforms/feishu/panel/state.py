"""Serializable server-owned state for a Feishu control panel."""

from __future__ import annotations

import copy
import json
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class PanelState:
    panel_id: str
    message_id: str
    app_id: str
    owner_open_id: str
    chat_id: str
    thread_id: str
    session_key: str
    profile: str = "default"
    chat_type: str = "group"
    user_id: str = ""
    user_id_alt: str = ""
    user_name: str = ""
    view: str = "home"
    view_stack: list[str] = field(default_factory=list)
    page: int = 0
    filters: dict[str, str] = field(default_factory=dict)
    revision: int = 0
    busy_action_id: str = ""
    busy_started_at: float = 0.0
    active: bool = True
    lifecycle: str = "active"
    data: dict[str, Any] = field(default_factory=dict)
    handled_nonces: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    expires_at: float = field(default_factory=lambda: time.time() + 7 * 24 * 60 * 60)

    @property
    def scope_key(self) -> str:
        return json.dumps(
            [self.app_id, self.chat_id, self.thread_id, self.owner_open_id],
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def clone(self) -> "PanelState":
        return copy.deepcopy(self)

    def touch(self) -> None:
        self.updated_at = time.time()

    def remember_nonce(self, nonce: str) -> None:
        if nonce and nonce not in self.handled_nonces:
            self.handled_nonces.append(nonce)
            del self.handled_nonces[:-64]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "PanelState":
        fields = cls.__dataclass_fields__
        return cls(**{key: item for key, item in value.items() if key in fields})
