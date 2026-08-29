"""Strict parsing for untrusted Feishu panel button payloads."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Mapping


PANEL_ACTION_VERSION = 1
_ALLOWED_OPS = frozenset({"nav", "back", "home", "page", "refresh", "select", "exec", "close"})
_MAX_TOKEN_LENGTH = 96


class PanelActionError(ValueError):
    """Raised when an untrusted card action does not match the panel contract."""


def normalize_mapping(value: Any) -> Any:
    """Recursively turn SDK/webhook namespaces into ordinary JSON values."""
    if isinstance(value, SimpleNamespace):
        return {key: normalize_mapping(item) for key, item in vars(value).items()}
    if isinstance(value, Mapping):
        return {str(key): normalize_mapping(item) for key, item in value.items()}
    if isinstance(value, list):
        return [normalize_mapping(item) for item in value]
    return value


def _token(value: Any, field: str, *, required: bool = True) -> str:
    result = str(value or "").strip()
    if required and not result:
        raise PanelActionError(f"missing {field}")
    if len(result) > _MAX_TOKEN_LENGTH:
        raise PanelActionError(f"invalid {field}")
    return result


@dataclass(frozen=True)
class PanelAction:
    panel_id: str
    revision: int
    op: str
    nonce: str
    target: str = ""
    page: int | None = None
    index: int | None = None


def parse_panel_action(raw_value: Any) -> PanelAction:
    """Parse only references/indices; never accept commands or trusted values."""
    value = normalize_mapping(raw_value)
    if not isinstance(value, dict) or value.get("panel_action") not in {True, 1, "1"}:
        raise PanelActionError("not a panel action")
    try:
        raw_version = value.get("v")
        raw_revision = value.get("rev")
        if raw_version is None or raw_revision is None:
            raise ValueError
        version = int(raw_version)
        revision = int(raw_revision)
    except (TypeError, ValueError) as exc:
        raise PanelActionError("invalid version or revision") from exc
    if version != PANEL_ACTION_VERSION or revision < 0:
        raise PanelActionError("unsupported version or revision")
    op = _token(value.get("op"), "op")
    if op not in _ALLOWED_OPS:
        raise PanelActionError("unsupported operation")

    def optional_int(name: str) -> int | None:
        raw = value.get(name)
        if raw is None:
            return None
        try:
            parsed = int(raw)
        except (TypeError, ValueError) as exc:
            raise PanelActionError(f"invalid {name}") from exc
        if parsed < 0 or parsed > 10000:
            raise PanelActionError(f"invalid {name}")
        return parsed

    return PanelAction(
        panel_id=_token(value.get("panel"), "panel"),
        revision=revision,
        op=op,
        nonce=_token(value.get("nonce"), "nonce"),
        target=_token(value.get("target"), "target", required=False),
        page=optional_int("page"),
        index=optional_int("index"),
    )
