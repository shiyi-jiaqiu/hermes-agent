"""Render ID-correlated tool lifecycle state as a compact Feishu card.

The renderer is deliberately pure: no SDK calls, no gateway state, and no
conversation-history writes.  This keeps failures in presentation isolated
from the agent turn.
"""

from __future__ import annotations

from typing import Any, Iterable

from gateway.tool_progress_diff import EditDiffSummary

_STATUS_ICON = {
    "running": "→",
    "success": "✓",
    "error": "✗",
}
_DIFF_ICON = {
    "added": "🟢 A",
    "modified": "🟡 M",
    "deleted": "🔴 D",
}


def _safe_inline(value: Any, limit: int = 1000) -> str:
    text = " ".join(str(value or "").replace("`", "ˋ").split())
    if len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


def _safe_code(value: Any, limit: int = 1000) -> str:
    text = str(value or "").strip().replace("```", "``\u200b`")
    if len(text) > limit:
        return text[: max(0, limit - 3)].rstrip() + "..."
    return text


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
    labels = {
        "terminal": "Terminal",
        "read_file": "Read",
        "search_files": "Search",
        "patch": "Patch",
        "write_file": "Write",
        "web_search": "Web Search",
        "web_extract": "Web Extract",
    }
    return f"{emoji} {labels.get(name, name)}"


def _tool_preview(item: dict[str, Any]) -> tuple[str, bool]:
    name = str(item.get("tool_name") or "")
    raw_args = item.get("args")
    args: dict[str, Any] = raw_args if isinstance(raw_args, dict) else {}
    if name == "terminal" and isinstance(args.get("command"), str):
        return _safe_code(args["command"]), True
    preview = item.get("preview")
    if preview:
        return _safe_inline(preview), False
    for key in ("path", "query", "url"):
        if args.get(key):
            return _safe_inline(args[key]), False
    return "", False


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
            lines.extend(("```diff", *file.lines, "```"))
        if file.omitted_lines:
            lines.append(f"_… {file.omitted_lines} more diff line(s) omitted_ ")
    if summary.omitted_files:
        lines.append(f"_… {summary.omitted_files} additional file(s) omitted_ ")
    elif summary.omitted_lines and not any(f.omitted_lines for f in summary.files):
        lines.append(f"_… {summary.omitted_lines} diff line(s) omitted_ ")
    return lines


def _render_markdown(
    items: list[dict[str, Any]],
    *,
    edit_display: str,
    max_items: int,
    max_chars: int,
) -> str:
    omitted_items = max(0, len(items) - max_items)
    visible = items[-max_items:]
    lines: list[str] = []
    for item in visible:
        status = str(item.get("status") or "running")
        icon = _STATUS_ICON.get(status, "·")
        title = _tool_title(item)
        suffix = _duration(item)
        if item.get("exit_code") is not None:
            suffix += f" · exit {item['exit_code']}"
        lines.append(f"**{icon} {title}**{suffix}")
        preview, is_code = _tool_preview(item)
        if preview:
            if is_code:
                lines.extend(("```", preview, "```"))
            else:
                lines.append(f"`{preview}`")
        diff = item.get("diff")
        if isinstance(diff, EditDiffSummary) and edit_display in {"summary", "diff"}:
            lines.extend(
                _diff_summary_lines(diff, include_body=edit_display == "diff")
            )
        if status == "error" and item.get("error"):
            lines.append(f"**Error:** {_safe_inline(item['error'], 300)}")
        lines.append("")
    if omitted_items:
        lines.append(f"_… {omitted_items} earlier tool call(s) hidden_ ")
    rendered = "\n".join(lines).strip() or "Preparing tools…"
    if len(rendered) <= max_chars:
        return rendered

    # Preserve task state and file summaries when the full diff would exceed the
    # card budget.  This is deterministic and preferable to cutting a code fence.
    if edit_display == "diff":
        return _render_markdown(
            items,
            edit_display="summary",
            max_items=max_items,
            max_chars=max_chars,
        )
    return rendered[: max(0, max_chars - 30)].rstrip() + "\n\n_… card content truncated_"


def render_progress_card(
    items: Iterable[dict[str, Any]],
    *,
    finalized: bool = False,
    edit_display: str = "diff",
    max_items: int = 8,
    max_chars: int = 7200,
) -> dict[str, Any]:
    """Build a full replacement interactive card."""
    materialized = [dict(item) for item in items]
    has_error = any(item.get("status") == "error" for item in materialized)
    still_running = any(item.get("status") == "running" for item in materialized)
    if has_error:
        template, state = "red", "Completed with errors"
    elif finalized or (materialized and not still_running):
        template, state = "green", "Completed"
    else:
        template, state = "blue", "Working"
    return {
        "config": {"wide_screen_mode": True},
        "header": {
            "title": {
                "tag": "plain_text",
                "content": f"⚕ Hermes Coding Progress · {state}",
            },
            "template": template,
        },
        "elements": [
            {
                "tag": "markdown",
                "content": _render_markdown(
                    materialized,
                    edit_display=edit_display,
                    max_items=max(1, int(max_items or 1)),
                    max_chars=max(500, int(max_chars or 7200)),
                ),
            }
        ],
    }


def render_progress_fallback(
    items: Iterable[dict[str, Any]],
    *,
    edit_display: str = "summary",
    max_items: int = 8,
    max_chars: int = 7000,
) -> str:
    """Text/post fallback when interactive-card transport is unavailable."""
    return "**⚕ Hermes Coding Progress**\n\n" + _render_markdown(
        [dict(item) for item in items],
        edit_display=edit_display,
        max_items=max_items,
        max_chars=max_chars,
    )
