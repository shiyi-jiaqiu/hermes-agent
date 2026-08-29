#!/usr/bin/env python3
"""Configure and verify the WSL2 Hermes bot's Feishu DM control menu.

Credentials are loaded from ~/.hermes/.env and are never printed. The script
uses only official Application v7 write/publish APIs and Application v6
version readback APIs.
"""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

BASE = "https://open.feishu.cn"
ENV_PATH = Path.home() / ".hermes" / ".env"
MENU_EVENT = "application.bot.menu_v6"
CARD_ACTION_EVENT = "card.action.trigger"


def load_env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def request_json(path: str, *, token: str | None = None, method: str = "GET", body: Any = None) -> dict:
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    payload = None if body is None else json.dumps(body, ensure_ascii=False).encode()
    request = urllib.request.Request(BASE + path, data=payload, method=method, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            result = json.load(response)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")
        raise RuntimeError(f"{method} {path} failed: HTTP {exc.code}: {detail[:2000]}") from exc
    if result.get("code") not in (None, 0):
        raise RuntimeError(f"{method} {path} failed: {result.get('code')} {result.get('msg')}")
    return result


def tenant_token(env: dict[str, str]) -> str:
    result = request_json(
        "/open-apis/auth/v3/tenant_access_token/internal",
        method="POST",
        body={"app_id": env["FEISHU_APP_ID"], "app_secret": env["FEISHU_APP_SECRET"]},
    )
    return str(result["tenant_access_token"])


def node(menu_id: str, name: str, sort: int, *, parent: str = "", event_key: str = "") -> dict:
    item = {
        "menu_id": menu_id,
        "sort": sort,
        "default_name": name,
        "i18n_name": {"zh_cn": name, "en_us": name},
        "menu_content_type": 2 if event_key else 3,
    }
    if parent:
        item["parent_menu_id"] = parent
    if event_key:
        item["event_key"] = event_key
    return item


def menu_nodes() -> list[dict]:
    # Feishu permits at most three top-level bot menus. Keep them as navigation
    # only: every actionable item creates/replaces the operator's single
    # stateful card at a logical page; no setting is changed by the menu.
    return [
        node("open_panel", "🎛 控制面板", 1, event_key="panel"),
        node("root_settings", "⚙️ 设置", 2),
        node("select_model", "🤖 模型", 1, parent="root_settings", event_key="select_model"),
        node("select_reasoning", "🧠 推理", 2, parent="root_settings", event_key="select_reasoning"),
        node("root_context", "📚 查看", 3),
        node("sessions", "📂 会话", 1, parent="root_context", event_key="sessions"),
        node("status", "📌 状态", 2, parent="root_context", event_key="status"),
    ]


def online_version(app_id: str, token: str) -> dict:
    app = request_json(f"/open-apis/application/v6/applications/{app_id}?lang=zh_cn", token=token)["data"]["app"]
    version_id = app.get("online_version_id")
    if not version_id:
        return {"app": app, "version": {}}
    version = request_json(
        f"/open-apis/application/v6/applications/{app_id}/app_versions/{version_id}?lang=zh_cn",
        token=token,
    )["data"]["app_version"]
    return {"app": app, "version": version}


def verify(app_id: str, token: str, expected_version_id: str | None = None) -> bool:
    current = online_version(app_id, token)
    app = current["app"]
    version = current["version"]
    event_types = {item.get("event_type") for item in version.get("event_infos", [])}
    callback_info = app.get("callback_info") or {}
    subscribed_callbacks = set(callback_info.get("subscribed_callbacks") or [])
    bot = ((version.get("ability") or {}).get("bot") or {})
    menus = bot.get("bot_menus") or []
    def semantic_menu_contract(items: list[dict]) -> set[tuple[str, str, str]]:
        names_by_id = {
            str(item.get("menu_id") or ""): str(item.get("default_name") or "")
            for item in items
        }
        return {
            (
                str(item.get("default_name") or ""),
                str(item.get("event_key") or ""),
                names_by_id.get(str(item.get("parent_menu_id") or ""), ""),
            )
            for item in items
        }

    expected_menu_contract = semantic_menu_contract(menu_nodes())
    online_menu_contract = semantic_menu_contract(menus)
    menu_contract_verified = online_menu_contract == expected_menu_contract
    version_id = version.get("version_id")
    ok = (
        MENU_EVENT in event_types
        and callback_info.get("callback_type") == "websocket"
        and CARD_ACTION_EVENT in subscribed_callbacks
        and bot.get("bot_menu_enable") is True
        and menu_contract_verified
        and (not expected_version_id or version_id == expected_version_id)
    )
    print(json.dumps({
        "online_version": version.get("version"),
        "online_version_id": version_id,
        "menu_event_subscribed": MENU_EVENT in event_types,
        "card_action_event_subscribed": CARD_ACTION_EVENT in subscribed_callbacks,
        "callback_type": callback_info.get("callback_type"),
        "bot_menu_enabled": bot.get("bot_menu_enable") is True,
        "menu_node_count": len(menus),
        "menu_contract_verified": menu_contract_verified,
        "verified": ok,
    }, ensure_ascii=False, indent=2))
    return ok


def apply(app_id: str, token: str) -> str:
    request_json(
        f"/open-apis/application/v7/applications/{app_id}/config",
        token=token,
        method="PATCH",
        body={
            "event": {
                "subscription_type": "websocket",
                "add_events": [MENU_EVENT],
            },
            "callback": {
                "callback_type": "websocket",
                "add_callbacks": [CARD_ACTION_EVENT],
            }
        },
    )
    request_json(
        f"/open-apis/application/v7/applications/{app_id}/ability",
        token=token,
        method="PATCH",
        body={"bot": {
            "enable": True,
            "bot_menu_enable": True,
            "bot_menus": menu_nodes(),
            "bot_menu_display_strategy": 1,
        }},
    )
    published = request_json(
        f"/open-apis/application/v7/applications/{app_id}/publish",
        token=token,
        method="POST",
        body={
            "pc_default_ability": "bot",
            "mobile_default_ability": "bot",
            "remark": "Hermes Feishu control panel",
            "changelog": "Use DM menu as first-level navigation for the single-card stateful Control Panel.",
        },
    )
    data = published.get("data") or {}
    print(json.dumps({"published_version": data.get("version"), "version_id": data.get("version_id")}, ensure_ascii=False))
    return str(data.get("version_id") or "")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true", help="patch, publish, and verify")
    parser.add_argument("--wait", type=int, default=180, help="seconds to wait for online readback")
    args = parser.parse_args()

    env = load_env(ENV_PATH)
    app_id = env["FEISHU_APP_ID"]
    token = tenant_token(env)
    if not args.apply:
        return 0 if verify(app_id, token) else 1

    version_id = apply(app_id, token)
    deadline = time.time() + max(0, args.wait)
    while True:
        if verify(app_id, token, expected_version_id=version_id or None):
            return 0
        if time.time() >= deadline:
            return 2
        time.sleep(5)


if __name__ == "__main__":
    raise SystemExit(main())
