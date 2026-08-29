"""Render :class:`PanelState` into Feishu Card JSON 2.0."""

from __future__ import annotations

import math
import uuid
from typing import Any, Iterable

from .state import PanelState

PROVIDER_PAGE_SIZE = 6
MODEL_PAGE_SIZE = 6
SESSION_PAGE_SIZE = 5


def _plain(value: Any, limit: int = 160) -> str:
    text = " ".join(str(value or "").replace("`", "'").split())
    return text[:limit]


def _markdown(content: str) -> dict[str, Any]:
    return {"tag": "markdown", "content": content}


def _action_value(
    state: PanelState,
    op: str,
    *,
    target: str = "",
    page: int | None = None,
    index: int | None = None,
) -> dict[str, Any]:
    value: dict[str, Any] = {
        "panel_action": True,
        "v": 1,
        "panel": state.panel_id,
        "rev": state.revision,
        "op": op,
        "nonce": f"a_{uuid.uuid4().hex}",
    }
    if target:
        value["target"] = target
    if page is not None:
        value["page"] = page
    if index is not None:
        value["index"] = index
    return value


def _button(
    state: PanelState,
    label: str,
    op: str,
    *,
    target: str = "",
    page: int | None = None,
    index: int | None = None,
    kind: str = "default",
    disabled: bool = False,
) -> dict[str, Any]:
    button: dict[str, Any] = {
        "tag": "button",
        "text": {"tag": "plain_text", "content": _plain(label, 40)},
        "type": kind,
        "behaviors": [
            {
                "type": "callback",
                "value": _action_value(
                    state,
                    op,
                    target=target,
                    page=page,
                    index=index,
                ),
            }
        ],
    }
    if disabled:
        button["disabled"] = True
    return button


def _rows(buttons: Iterable[dict[str, Any]], row_size: int = 3) -> list[dict[str, Any]]:
    materialized = list(buttons)
    return [
        {
            "tag": "column_set",
            "flex_mode": "flow",
            "horizontal_spacing": "default",
            "columns": [
                {
                    "tag": "column",
                    "width": "weighted",
                    "weight": 1,
                    "elements": [button],
                }
                for button in materialized[start : start + row_size]
            ],
        }
        for start in range(0, len(materialized), row_size)
    ]


def _navigation(state: PanelState, *, include_cancel: bool = False) -> list[dict[str, Any]]:
    buttons = [
        _button(state, "← 返回", "back"),
        _button(state, "⌂ 首页", "home"),
    ]
    if include_cancel:
        buttons.append(_button(state, "取消", "back"))
    return _rows(buttons)


def _page_controls(state: PanelState, page: int, pages: int) -> list[dict[str, Any]]:
    return _rows(
        [
            _button(state, "‹ 上一页", "page", page=max(0, page - 1), disabled=page <= 0),
            _button(state, f"{page + 1} / {pages}", "page", page=page, disabled=True),
            _button(state, "下一页 ›", "page", page=min(pages - 1, page + 1), disabled=page >= pages - 1),
        ]
    )


def _pending_view(
    state: PanelState,
    view: str,
    breadcrumb: str,
    noun: str,
) -> list[dict[str, Any]]:
    """Render a bounded loading/error state with an explicit retry."""
    errors = dict(state.data.get("load_errors") or {})
    loading = view in set(state.data.get("loading_views") or [])
    error = _plain(errors.get(view) or "", 180)
    if loading:
        message = f"⏳ 正在加载{noun}…"
        retry_disabled = True
    elif error:
        message = f"❌ {error}"
        retry_disabled = False
    else:
        message = f"尚未加载{noun}。"
        retry_disabled = False
    return [
        _markdown(f"⌂ 首页 / {breadcrumb}\n\n{message}"),
        *_rows(
            [
                _button(
                    state,
                    "↻ 立即重试" if error else "↻ 加载",
                    "refresh",
                    target="snapshot",
                    disabled=retry_disabled,
                )
            ]
        ),
        *_navigation(state),
    ]


def _home(state: PanelState) -> list[dict[str, Any]]:
    data = state.data
    running = bool(data.get("running"))
    model = _plain(data.get("effective_model") or "unknown", 80)
    reasoning = _plain(data.get("effective_reasoning") or "default", 30)
    fast = "on" if data.get("fast_mode") else "off"
    model_source = _plain(data.get("model_source") or data.get("value_source") or "global default", 40)
    reasoning_source = _plain(data.get("reasoning_source") or data.get("value_source") or "global default", 40)
    preset = _plain(data.get("current_preset") or "Custom", 40)
    session_suffix = _plain(state.session_key[-10:], 12)
    session_scope = "Topic" if state.thread_id else ("DM" if state.chat_type == "dm" else "群聊")
    session = f"{_plain(state.profile or 'default', 20)} · {session_scope} · …{session_suffix}"
    source_line = (
        f"**作用域：** {model_source}"
        if model_source == reasoning_source
        else f"**来源：** 模型 {model_source} · 推理 {reasoning_source}"
    )
    elements: list[dict[str, Any]] = [
        _markdown(
            f"**{'Working' if running else 'Idle'}** · `{session}`\n"
            f"**当前预设：** {preset}\n"
            f"**模型：** `{model}` · **Reasoning：** `{reasoning}` · **Fast：** `{fast}`\n"
            f"{source_line}"
        )
    ]
    if state.busy_action_id:
        elements.append(_markdown("⏳ **正在处理控制操作…** 导航仍可使用。"))
    home_loading = "home" in set(data.get("loading_views") or [])
    if home_loading:
        elements.append(_markdown("⏳ 正在刷新首页状态；其他按钮仍可使用。"))
    flash = _plain(data.get("flash") or "", 240)
    if flash:
        elements.append(_markdown(f"> {flash}"))

    presets = list(data.get("preset_options") or [])
    preset_buttons = []
    for index, item in enumerate(presets[:3]):
        name = _plain(item.get("name") if isinstance(item, dict) else item, 30)
        label = _plain(item.get("label") if isinstance(item, dict) else name, 30)
        selected = bool(name and name == str(data.get("current_preset") or ""))
        preset_buttons.append(
            _button(
                state,
                f"{'✓ ' if selected else ''}{label}",
                "exec",
                target="preset",
                index=index,
                kind="primary" if selected else "default",
                disabled=bool(state.busy_action_id) or selected,
            )
        )
    if preset_buttons:
        elements.extend([_markdown("**组合预设**"), *_rows(preset_buttons)])

    if data.get("fast_supported"):
        fast_mode = bool(data.get("fast_mode"))
        elements.extend(
            [
                _markdown("**性能模式（当前会话）**"),
                *_rows(
                    [
                        _button(
                            state,
                            f"{'✓ ' if fast_mode else ''}⚡ Fast",
                            "exec",
                            target="fast",
                            index=0,
                            kind="primary" if fast_mode else "default",
                            disabled=bool(state.busy_action_id) or fast_mode,
                        ),
                        _button(
                            state,
                            f"{'✓ ' if not fast_mode else ''}正常",
                            "exec",
                            target="fast",
                            index=1,
                            kind="primary" if not fast_mode else "default",
                            disabled=bool(state.busy_action_id) or not fast_mode,
                        ),
                    ],
                    row_size=2,
                ),
            ]
        )

    elements.extend(
        [
            _markdown("**高级设置**"),
            *_rows(
                [
                    _button(state, "🤖 模型", "nav", target="model"),
                    _button(state, "🧠 推理", "nav", target="reasoning"),
                    _button(state, "🗂 会话", "nav", target="sessions"),
                    _button(state, "📌 状态", "nav", target="status"),
                ],
                row_size=2,
            ),
        ]
    )
    runtime_buttons = [
        _button(
            state,
            "↻ 刷新",
            "refresh",
            target="snapshot",
            disabled=bool(state.busy_action_id) or home_loading,
        ),
        _button(state, "关闭面板", "close"),
    ]
    if running:
        runtime_buttons.insert(
            0,
            _button(state, "⛔ 停止当前任务", "exec", target="stop", kind="danger"),
        )
    elements.extend(_rows(runtime_buttons, row_size=3))
    return elements


def _model(state: PanelState) -> list[dict[str, Any]]:
    """Render level one of the picker: authenticated/configured providers."""
    if "model" not in set(state.data.get("loaded_views") or []):
        return _pending_view(state, "model", "模型", "模型目录")
    providers = list(state.data.get("model_providers") or [])
    pages = max(1, math.ceil(len(providers) / PROVIDER_PAGE_SIZE))
    page = min(max(0, state.page), pages - 1)
    start = page * PROVIDER_PAGE_SIZE
    current = str(state.data.get("effective_model") or "")
    current_provider = str(state.data.get("effective_provider") or "")
    default = str(state.data.get("global_model") or "")
    default_provider = str(state.data.get("global_provider") or "")
    source = _plain(state.data.get("model_source") or state.data.get("value_source") or "global default", 50)
    elements: list[dict[str, Any]] = [
        _markdown(
            f"⌂ 首页 / 模型 / 供应商 · 第 {page + 1} / {pages} 页\n"
            f"**当前：** `{_plain(current_provider or 'unknown', 50)} / {_plain(current, 100)}`\n"
            f"**来源：** {source}\n"
            f"**全局默认：** `{_plain(default_provider or 'unknown', 50)} / {_plain(default, 100)}`\n\n"
            "请选择供应商："
        )
    ]
    buttons = []
    for index, item in enumerate(
        providers[start : start + PROVIDER_PAGE_SIZE], start=start
    ):
        if not isinstance(item, dict):
            continue
        slug = str(item.get("slug") or "")
        name = _plain(item.get("name") or slug or "provider", 25)
        available = int(item.get("available_models") or len(item.get("model_indices") or []))
        total = max(available, int(item.get("total_models") or 0))
        count = f"{available}/{total}" if total > available else str(available)
        selected = bool(slug and slug == current_provider)
        buttons.append(
            _button(
                state,
                f"{'✓ ' if selected else ''}{name} ({count})",
                "select",
                target="model_provider",
                index=index,
                kind="primary" if selected else "default",
            )
        )
    elements.extend(
        _rows(buttons, row_size=2)
        if buttons
        else [_markdown("没有可用的已认证或已配置供应商。")]
    )
    if pages > 1:
        elements.extend(_page_controls(state, page, pages))
    elements.extend(_navigation(state))
    return elements


def _model_provider(state: PanelState) -> list[dict[str, Any]]:
    """Render level two: models belonging to the selected provider only."""
    if "model" not in set(state.data.get("loaded_views") or []):
        return _pending_view(state, "model", "模型", "模型目录")
    providers = list(state.data.get("model_providers") or [])
    options = list(state.data.get("model_options") or [])
    selected_slug = str(state.filters.get("model_provider") or "")
    provider = next(
        (
            item
            for item in providers
            if isinstance(item, dict) and str(item.get("slug") or "") == selected_slug
        ),
        None,
    )
    if not isinstance(provider, dict):
        return [
            _markdown("⌂ 首页 / 模型\n\n所选供应商已不在当前模型目录中，请返回重新选择。"),
            *_navigation(state),
        ]

    model_indices = [
        index
        for index in (provider.get("model_indices") or [])
        if isinstance(index, int) and 0 <= index < len(options)
    ]
    pages = max(1, math.ceil(len(model_indices) / MODEL_PAGE_SIZE))
    page = min(max(0, state.page), pages - 1)
    start = page * MODEL_PAGE_SIZE
    current = str(state.data.get("effective_model") or "")
    current_provider = str(state.data.get("effective_provider") or "")
    slug = str(provider.get("slug") or "")
    name = _plain(provider.get("name") or slug or "provider", 60)
    elements: list[dict[str, Any]] = [
        _markdown(
            f"⌂ 首页 / 模型 / {name} · 第 {page + 1} / {pages} 页\n"
            f"**供应商：** `{_plain(slug, 60)}`\n"
            f"**当前：** `{_plain(current_provider or 'unknown', 50)} / {_plain(current, 100)}`\n\n"
            "请选择模型："
        )
    ]
    buttons = []
    for option_index in model_indices[start : start + MODEL_PAGE_SIZE]:
        item = options[option_index]
        if not isinstance(item, dict):
            continue
        value = str(item.get("model") or item.get("target") or "")
        label = _plain(item.get("label") or value, 36)
        selected = bool(value and value == current and slug == current_provider)
        buttons.append(
            _button(
                state,
                f"{'✓ ' if selected else ''}{label}",
                "exec",
                target="model",
                index=option_index,
                kind="primary" if selected else "default",
                disabled=bool(state.busy_action_id) or selected,
            )
        )
    elements.extend(
        _rows(buttons, row_size=2)
        if buttons
        else [_markdown("该供应商当前没有可用模型。")]
    )
    if pages > 1:
        elements.extend(_page_controls(state, page, pages))
    elements.extend(_navigation(state))
    return elements


def _reasoning(state: PanelState, *, global_scope: bool = False) -> list[dict[str, Any]]:
    options = list(state.data.get("reasoning_options") or [])
    current_key = "global_reasoning" if global_scope else "effective_reasoning"
    current = str(state.data.get(current_key) or "medium")
    source = "全局默认" if global_scope else _plain(state.data.get("reasoning_source") or "global default", 50)
    elements: list[dict[str, Any]] = [
        _markdown(
            f"⌂ 首页 / {'修改全局默认' if global_scope else '推理设置'}\n"
            f"**当前有效值：** `{_plain(state.data.get('effective_reasoning') or 'medium', 30)}`\n"
            f"**值来源：** {_plain(state.data.get('reasoning_source') or 'global default', 50)}\n"
            f"**全局默认：** `{_plain(state.data.get('global_reasoning') or 'medium', 30)}`"
        )
    ]
    if global_scope:
        elements.append(_markdown("⚠️ 选择后还需确认；修改会影响本 Profile 后续创建的会话。"))
    buttons = []
    for index, item in enumerate(options):
        value = str(item.get("value") if isinstance(item, dict) else item)
        label = _plain(item.get("label") if isinstance(item, dict) else value, 30)
        selected = value == current
        buttons.append(
            _button(
                state,
                f"{'✓ ' if selected else ''}{label}",
                "select" if global_scope else "exec",
                target="global_reasoning" if global_scope else "reasoning",
                index=index,
                kind="primary" if selected else "default",
                disabled=bool(state.busy_action_id) or selected,
            )
        )
    elements.extend([_markdown("**推理强度**"), *_rows(buttons, row_size=3)])
    if not global_scope:
        display = bool(state.data.get("show_reasoning"))
        elements.extend(
            [
                _markdown("**显示设置（Profile 平台默认）**"),
                *_rows(
                    [
                        _button(state, f"{'✓ ' if display else ''}显示思考过程", "exec", target="reasoning_display", index=0, disabled=bool(state.busy_action_id) or display),
                        _button(state, f"{'✓ ' if not display else ''}隐藏", "exec", target="reasoning_display", index=1, disabled=bool(state.busy_action_id) or not display),
                    ],
                    row_size=2,
                ),
                *_rows(
                    [
                        _button(state, "重置本会话覆盖", "exec", target="reasoning_reset", kind="danger", disabled=bool(state.busy_action_id)),
                        _button(state, "修改全局默认", "nav", target="reasoning_global"),
                    ],
                    row_size=2,
                ),
            ]
        )
    elements.extend(_navigation(state))
    return elements


def _confirm_global_reasoning(state: PanelState) -> list[dict[str, Any]]:
    index = state.data.get("pending_global_reasoning_index")
    options = list(state.data.get("reasoning_options") or [])
    target = options[index].get("value") if isinstance(index, int) and 0 <= index < len(options) else ""
    return [
        _markdown(
            "⚠️ **修改全局默认配置**\n"
            "这会影响此 Profile 后续创建的会话；现有会话覆盖项不会自动清除。\n\n"
            f"`{_plain(state.data.get('global_reasoning') or 'medium', 30)}` → `{_plain(target, 30)}`"
        ),
        *_rows(
            [
                _button(state, "确认修改", "exec", target="global_reasoning", index=index if isinstance(index, int) else 0, kind="danger", disabled=bool(state.busy_action_id)),
                _button(state, "取消", "back"),
            ],
            row_size=2,
        ),
    ]


def _sessions(state: PanelState) -> list[dict[str, Any]]:
    if "sessions" not in set(state.data.get("loaded_views") or []):
        return _pending_view(state, "sessions", "会话", "会话列表")
    sessions = list(state.data.get("sessions") or [])
    pages = max(1, math.ceil(len(sessions) / SESSION_PAGE_SIZE))
    page = min(max(0, state.page), pages - 1)
    start = page * SESSION_PAGE_SIZE
    elements: list[dict[str, Any]] = [
        _markdown(f"⌂ 首页 / 会话 · 第 {page + 1} / {pages} 页\n仅显示当前 Profile、用户、chat 与 Topic/Thread。")
    ]
    for index, item in enumerate(sessions[start : start + SESSION_PAGE_SIZE], start=start):
        if not isinstance(item, dict):
            continue
        title = _plain(item.get("title") or item.get("preview") or f"Session {index + 1}", 80)
        preview = _plain(item.get("preview") or "", 100)
        if preview == title:
            preview = ""
        elements.append(_markdown(f"**{index + 1}. {title}**\n{preview}" if preview else f"**{index + 1}. {title}**"))
        elements.extend(
            _rows(
                [_button(state, "恢复", "exec", target="resume", index=index, disabled=bool(state.busy_action_id))]
            )
        )
    if not sessions:
        elements.append(_markdown("没有可恢复的历史会话。"))
    if pages > 1:
        elements.extend(_page_controls(state, page, pages))
    elements.extend(
        _rows([_button(state, "＋ 新建会话", "nav", target="confirm_new", kind="primary")])
    )
    elements.extend(_navigation(state))
    return elements


def _status(state: PanelState) -> list[dict[str, Any]]:
    if "status" not in set(state.data.get("loaded_views") or []):
        return _pending_view(state, "status", "状态", "状态信息")
    status = str(state.data.get("status_text") or "No status available.")[:3000]
    return [
        _markdown(f"⌂ 首页 / 状态\n\n{status}"),
        *_rows(
            [
                _button(state, "↻ 刷新", "refresh", target="snapshot", disabled=bool(state.busy_action_id)),
                *(
                    [_button(state, "⛔ 停止当前任务", "exec", target="stop", kind="danger")]
                    if state.data.get("running")
                    else []
                ),
            ],
            row_size=2,
        ),
        *_navigation(state),
    ]


def _confirm_new(state: PanelState) -> list[dict[str, Any]]:
    return [
        _markdown("⚠️ **新建会话**\n这会结束当前会话上下文并创建一个空白会话。"),
        *_rows(
            [
                _button(state, "确认新建", "exec", target="new", kind="danger", disabled=bool(state.busy_action_id)),
                _button(state, "取消", "back"),
            ],
            row_size=2,
        ),
        *_navigation(state),
    ]


def render_panel(state: PanelState) -> dict[str, Any]:
    """Render a complete replacement card for the current logical view."""
    if state.lifecycle == "replaced":
        elements = [_markdown("此面板已被同一用户的新面板替代。请使用最新的 Hermes Control Panel。")]
        template = "grey"
        title = "Hermes Control · 已替代"
    elif not state.active or state.lifecycle == "closed":
        elements = [_markdown("此控制面板已关闭。重新发送 `/panel` 可创建新面板。")]
        template = "grey"
        title = "Hermes Control · 已关闭"
    else:
        renderers = {
            "home": _home,
            "model": _model,
            "model_provider": _model_provider,
            "reasoning": _reasoning,
            "reasoning_global": lambda value: _reasoning(value, global_scope=True),
            "confirm_global_reasoning": _confirm_global_reasoning,
            "sessions": _sessions,
            "status": _status,
            "confirm_new": _confirm_new,
        }
        renderer = renderers.get(state.view, _home)
        elements = renderer(state)
        template = "turquoise" if state.view == "home" else "blue"
        view_titles = {
            "home": "首页",
            "model": "模型",
            "model_provider": "选择模型",
            "reasoning": "推理设置",
            "reasoning_global": "全局推理默认",
            "confirm_global_reasoning": "确认全局修改",
            "sessions": "会话",
            "status": "状态",
            "confirm_new": "确认新建会话",
        }
        title = f"Hermes Control · {view_titles.get(state.view, '首页')}"
    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "header": {
            "template": template,
            "title": {"tag": "plain_text", "content": title},
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": elements,
        },
    }
