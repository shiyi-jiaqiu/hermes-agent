"""Render ID-correlated tool lifecycle state as a Feishu Card JSON 2.0 card.

The renderer is deliberately pure: no SDK calls, no gateway state, and no
conversation-history writes.  Presentation failures therefore remain isolated
from the agent turn.  Card content is emitted as separate native Markdown
components so Feishu can render inline code, fenced code blocks, copy controls,
and syntax highlighting instead of flattening every tool into one ``lark_md``
string.
"""

from __future__ import annotations

import json
from typing import Any, Iterable

from gateway.tool_progress_diff import EditDiffSummary

_STATUS = {
    "running": ("blue", "Running", "→"),
    "success": ("green", "Success", "✓"),
    "error": ("red", "Error", "✗"),
}
_DIFF_ICON = {
    "added": "🟢 A",
    "modified": "🟡 M",
    "deleted": "🔴 D",
}
_TOOL_LABELS = {
    "terminal": "Terminal",
    "read_file": "Read",
    "search_files": "Search",
    "patch": "Patch",
    "write_file": "Write",
    "web_search": "Web Search",
    "web_extract": "Web Extract",
    "execute_code": "Execute Code",
    "browser_exec": "Browser Script",
}
_DETAIL_FIELDS: dict[str, tuple[tuple[str, str], ...]] = {
    "read_file": (("path", "File"), ("offset", "Start line"), ("limit", "Line limit")),
    "search_files": (
        ("pattern", "Pattern"),
        ("path", "Path"),
        ("file_glob", "File filter"),
        ("target", "Target"),
        ("output_mode", "Output"),
    ),
    "patch": (("path", "File"),),
    "write_file": (("path", "File"),),
    "web_search": (("query", "Query"), ("limit", "Result limit")),
    "web_extract": (("urls", "URLs"),),
    "process": (("action", "Action"), ("session_id", "Session"), ("timeout", "Timeout")),
    "read_url": (("url", "URL"),),
    "browser_navigate": (("url", "URL"),),
}
_CODE_TOOLS = {
    "terminal": ("command", "bash"),
    "execute_code": ("code", "python"),
    "browser_exec": ("code", "javascript"),
}


def _safe_inline(value: Any, limit: int = 1000) -> str:
    """Return compact text safe inside a Markdown inline-code span."""
    text = " ".join(str(value or "").replace("`", "ˋ").split())
    if len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _safe_code(value: Any, limit: int = 1000) -> str:
    """Return bounded fenced-code content without allowing fence injection."""
    text = str(value or "").strip().replace("```", "``\u200b`")
    if len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _safe_diff_line(value: Any, limit: int = 2000) -> str:
    """Escape a diff line while preserving its significant leading whitespace."""
    text = str(value or "").replace("```", "``\u200b`")
    if len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _display_value(value: Any) -> str:
    if isinstance(value, (dict, list, tuple)):
        try:
            return json.dumps(value, ensure_ascii=False, indent=2, default=str)
        except (TypeError, ValueError):
            pass
    return str(value or "")


def _redact_display_value(value: Any) -> Any:
    """Apply the mandatory egress redactor recursively to card arguments."""
    if isinstance(value, dict):
        return {str(key): _redact_display_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact_display_value(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_redact_display_value(item) for item in value)
    if not isinstance(value, str):
        return value
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(value, force=True, redact_url_credentials=True)


def _display_args(item: dict[str, Any]) -> dict[str, Any]:
    raw_args = item.get("args")
    if not isinstance(raw_args, dict):
        return {}
    return _redact_display_value(raw_args)


def _duration(item: dict[str, Any]) -> str:
    value = item.get("duration")
    if not isinstance(value, (int, float)):
        return ""
    if value < 1:
        return f" · {value * 1000:.0f}ms"
    return f" · {value:.1f}s"


def _tool_title(item: dict[str, Any]) -> str:
    from agent.display import get_tool_emoji

    name = str(item.get("tool_name") or "tool")
    emoji = get_tool_emoji(name, default="⚙️")
    return f"{emoji} {_TOOL_LABELS.get(name, name)}"


def _status_heading(item: dict[str, Any]) -> str:
    status = str(item.get("status") or "running")
    color, label, icon = _STATUS.get(status, ("grey", "Status", "·"))
    suffix = _duration(item)
    if item.get("exit_code") is not None:
        suffix += f" · exit {item['exit_code']}"
    # text_tag is native Card JSON 2.0 Markdown.  The plain icon remains useful
    # on older clients that ignore the tag styling.
    return (
        f"<text_tag color='{color}'>{label}</text_tag> "
        f"**{icon} {_tool_title(item)}**{suffix}"
    )


def _inline_or_block(label: str, value: Any, *, language: str = "text") -> list[str]:
    rendered = _display_value(value).strip()
    if not rendered:
        return []
    compact = _safe_inline(rendered, 1000)
    if "\n" not in rendered and len(compact) <= 100:
        return [f"**{label}**", f"`{compact}`"]
    return [f"**{label}**", f"```{language}", _safe_code(rendered), "```"]


def _tool_detail_lines(item: dict[str, Any]) -> list[str]:
    name = str(item.get("tool_name") or "")
    args = _display_args(item)
    code_spec = _CODE_TOOLS.get(name)
    if code_spec:
        key, language = code_spec
        value = args.get(key)
        if value in (None, ""):
            value = _redact_display_value(item.get("preview") or "")
        code = _safe_code(value)
        lines = [f"```{language}", code, "```"] if code else []
        if name == "terminal" and args.get("workdir"):
            lines.extend(_inline_or_block("Working directory", args["workdir"]))
        return lines

    fields = _DETAIL_FIELDS.get(name)
    lines: list[str] = []
    if fields:
        for key, label in fields:
            value = args.get(key)
            if value in (None, "", [], ()):
                continue
            language = "regex" if key == "pattern" else "text"
            lines.extend(_inline_or_block(label, value, language=language))
        if lines:
            return lines

    preview = _redact_display_value(item.get("preview") or "")
    if preview:
        lines.extend(_inline_or_block("Details", preview))
    return lines


def _diff_summary_lines(
    summary: EditDiffSummary,
    *,
    include_body: bool,
) -> list[str]:
    lines = [
        f"**📝 {summary.total_files} file(s) changed · +{summary.additions} -{summary.deletions}**"
    ]
    for file in summary.files:
        marker = _DIFF_ICON.get(file.status, "🟡 M")
        lines.append(
            f"{marker} `{_safe_inline(file.path, 240)}`  **+{file.additions} -{file.deletions}**"
        )
        if include_body and file.lines:
            lines.extend(("```diff", *(_safe_diff_line(line) for line in file.lines), "```"))
        if file.omitted_lines:
            lines.append(f"_… {file.omitted_lines} more diff line(s) omitted_")
    if summary.omitted_files:
        lines.append(f"_… {summary.omitted_files} additional file(s) omitted_")
    elif summary.omitted_lines and not any(f.omitted_lines for f in summary.files):
        lines.append(f"_… {summary.omitted_lines} diff line(s) omitted_")
    return lines


def _render_tool_markdown(
    item: dict[str, Any],
    *,
    edit_display: str,
    include_details: bool = True,
) -> str:
    lines = [_status_heading(item)]
    if include_details:
        details = _tool_detail_lines(item)
        if details:
            lines.extend(("", *details))
    diff = item.get("diff")
    if isinstance(diff, EditDiffSummary) and edit_display in {"summary", "diff"}:
        lines.extend(("", *_diff_summary_lines(diff, include_body=edit_display == "diff")))
    if item.get("status") == "error" and item.get("error"):
        error = _safe_inline(_redact_display_value(item["error"]), 300)
        lines.extend(("", f"**Error**", f"> {error}"))
    return "\n".join(lines).strip()


def _visible_items(items: list[dict[str, Any]], max_items: int) -> tuple[list[dict[str, Any]], int]:
    omitted = max(0, len(items) - max_items)
    return items[-max_items:], omitted


def _render_blocks(
    items: list[dict[str, Any]],
    *,
    edit_display: str,
    max_items: int,
    max_chars: int,
) -> tuple[list[str], int]:
    visible, omitted = _visible_items(items, max_items)
    blocks = [_render_tool_markdown(item, edit_display=edit_display) for item in visible]

    def content_size(values: list[str]) -> int:
        return sum(len(value) for value in values)

    # Prefer summaries over cutting through a code fence.
    if content_size(blocks) > max_chars and edit_display == "diff":
        blocks = [_render_tool_markdown(item, edit_display="summary") for item in visible]

    # If previews still exceed the card budget, compact the oldest calls first
    # while preserving every visible lifecycle state and the newest details.
    index = 0
    while content_size(blocks) > max_chars and index < max(0, len(blocks) - 1):
        blocks[index] = _render_tool_markdown(
            visible[index], edit_display="summary", include_details=False
        )
        index += 1

    # The final block can still contain a very long error or preview.  Replace
    # it with a complete status heading rather than slicing Markdown syntax.
    if content_size(blocks) > max_chars and blocks:
        blocks[-1] = _render_tool_markdown(
            visible[-1], edit_display="summary", include_details=False
        )
    return blocks, omitted


def _render_markdown(
    items: list[dict[str, Any]],
    *,
    edit_display: str,
    max_items: int,
    max_chars: int,
) -> str:
    """Render the portable text/post fallback as one Markdown document."""
    blocks, omitted = _render_blocks(
        items,
        edit_display=edit_display,
        max_items=max_items,
        max_chars=max_chars,
    )
    parts = ["\n\n---\n\n".join(blocks) or "Preparing tools…"]
    if omitted:
        parts.append(f"_… {omitted} earlier tool call(s) hidden_")
    return "\n\n".join(parts)


def render_progress_card(
    items: Iterable[dict[str, Any]],
    *,
    finalized: bool = False,
    edit_display: str = "diff",
    max_items: int = 4,
    max_chars: int = 7200,
) -> dict[str, Any]:
    """Build a full replacement Card JSON 2.0 interactive card."""
    materialized = [dict(item) for item in items]
    has_error = any(item.get("status") == "error" for item in materialized)
    still_running = any(item.get("status") == "running" for item in materialized)
    if has_error:
        template, state = "red", "Completed with errors"
    elif finalized or (materialized and not still_running):
        template, state = "green", "Completed"
    else:
        template, state = "blue", "Working"

    normalized_max_items = max(1, int(max_items or 1))
    normalized_max_chars = max(500, int(max_chars or 7200))
    blocks, omitted = _render_blocks(
        materialized,
        edit_display=edit_display,
        max_items=normalized_max_items,
        max_chars=normalized_max_chars,
    )
    elements: list[dict[str, Any]] = []
    if omitted:
        elements.append(
            {
                "tag": "markdown",
                "content": f"_… {omitted} earlier tool call(s) hidden; showing the latest {len(blocks)}._",
            }
        )
    for index, block in enumerate(blocks):
        if index or elements:
            elements.append({"tag": "hr"})
        elements.append({"tag": "markdown", "content": block})
    if not elements:
        elements.append({"tag": "markdown", "content": "Preparing tools…"})

    return {
        "schema": "2.0",
        "config": {"update_multi": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"⚕ Hermes Coding Progress · {state}",
            },
            "template": template,
        },
        "body": {
            "direction": "vertical",
            "padding": "12px 12px 12px 12px",
            "elements": elements,
        },
    }


def render_progress_fallback(
    items: Iterable[dict[str, Any]],
    *,
    edit_display: str = "summary",
    max_items: int = 4,
    max_chars: int = 7000,
) -> str:
    """Text/post fallback when interactive-card transport is unavailable."""
    return "**⚕ Hermes Coding Progress**\n\n" + _render_markdown(
        [dict(item) for item in items],
        edit_display=edit_display,
        max_items=max(1, int(max_items or 1)),
        max_chars=max(500, int(max_chars or 7000)),
    )
