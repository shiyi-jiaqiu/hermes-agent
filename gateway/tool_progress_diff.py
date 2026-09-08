"""Presentation-only unified-diff extraction for gateway tool progress.

The agent and file tools already produce everything needed for inline edit
previews.  This module turns that data into a small, platform-neutral model so
messaging adapters can render it without changing conversation history or tool
results.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
import re
from typing import Any


_EDIT_TOOLS = frozenset({"patch", "write_file", "skill_manage"})


@dataclass(frozen=True)
class DiffFile:
    """One file section from a unified diff."""

    path: str
    status: str  # added | modified | deleted
    additions: int
    deletions: int
    lines: tuple[str, ...] = ()
    omitted_lines: int = 0


@dataclass(frozen=True)
class EditDiffSummary:
    """Bounded display model for one edit-tool completion."""

    files: tuple[DiffFile, ...] = ()
    additions: int = 0
    deletions: int = 0
    total_files: int = 0
    omitted_files: int = 0
    omitted_lines: int = 0
    truncated: bool = False


@dataclass
class _RawSection:
    old_path: str = ""
    new_path: str = ""
    body: list[str] = field(default_factory=list)


def _clean_path(raw: str) -> str:
    value = str(raw or "").strip().split("\t", 1)[0]
    if value == "/dev/null":
        return value
    if value.startswith(("a/", "b/")):
        value = value[2:]
    # difflib labels absolute paths as a//tmp/... and b//tmp/....
    if value.startswith("/"):
        return value
    return value or "?"


def _section_status(old_path: str, new_path: str) -> tuple[str, str]:
    old_clean = _clean_path(old_path)
    new_clean = _clean_path(new_path)
    if old_clean == "/dev/null":
        return "added", new_clean
    if new_clean == "/dev/null":
        return "deleted", old_clean
    return "modified", new_clean if new_clean != "?" else old_clean


_HUNK_HEADER = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@")


def _split_sections(diff: str) -> list[_RawSection]:
    sections: list[_RawSection] = []
    current: _RawSection | None = None
    lines = diff.splitlines()
    old_remaining = new_remaining = 0
    idx = 0
    while idx < len(lines):
        line = lines[idx]
        if old_remaining or new_remaining:
            if line.startswith("-") and old_remaining:
                old_remaining -= 1
            elif line.startswith("+") and new_remaining:
                new_remaining -= 1
            elif line.startswith(" ") and old_remaining and new_remaining:
                old_remaining -= 1
                new_remaining -= 1
            elif not line.startswith("\\ No newline at end of file"):
                raise ValueError("Invalid unified diff hunk body")
            current.body.append(line)
        elif line.startswith("--- ") and idx + 1 < len(lines) and lines[idx + 1].startswith("+++ "):
            current = _RawSection(old_path=line[4:], new_path=lines[idx + 1][4:])
            sections.append(current)
            idx += 1
        elif current is not None:
            header = _HUNK_HEADER.match(line)
            if header:
                old_remaining = int(header.group(1) or "1")
                new_remaining = int(header.group(2) or "1")
                current.body.append(line)
            elif line.startswith("\\ No newline at end of file"):
                current.body.append(line)
        idx += 1
    if old_remaining or new_remaining:
        raise ValueError("Incomplete unified diff hunk")
    return sections


def _redact_diff_text(diff: str) -> str:
    """A failed redactor aborts this preview at its single presentation boundary."""
    from agent.redact import redact_sensitive_text
    return redact_sensitive_text(diff, force=True, redact_url_credentials=True)


def parse_unified_diff(
    diff: str,
    *,
    max_files: int = 6,
    max_lines: int = 80,
    max_chars: int = 6000,
    redact: bool = True,
) -> EditDiffSummary | None:
    """Parse and cap a unified diff for chat display.

    Totals describe the complete diff.  ``files`` and each file's ``lines`` are
    the bounded visible subset.  File headers are represented structurally and
    therefore do not consume the line budget.
    """
    text = str(diff or "")
    if not text.strip():
        return None

    raw_sections = _split_sections(text)
    if not raw_sections:
        return None

    max_files = max(1, int(max_files or 1))
    max_lines = max(0, int(max_lines or 0))
    max_chars = max(0, int(max_chars or 0))

    total_additions = 0
    total_deletions = 0
    visible: list[DiffFile] = []
    remaining_lines = max_lines
    remaining_chars = max_chars
    omitted_lines = 0

    for section_index, section in enumerate(raw_sections):
        status, path = _section_status(section.old_path, section.new_path)
        additions = sum(
            1 for line in section.body if line.startswith("+")
        )
        deletions = sum(
            1 for line in section.body if line.startswith("-")
        )
        total_additions += additions
        total_deletions += deletions

        if section_index >= max_files:
            omitted_lines += len(section.body)
            continue

        kept: list[str] = []
        section_omitted = 0
        # Count raw hunks first. Multi-line redaction may collapse lines, but must
        # never change the actual addition/deletion totals or break parsing.
        display_body = (_redact_diff_text("\n".join(section.body)).splitlines() if redact else section.body)
        if redact:
            path = _redact_diff_text(path)
        for safe_line in display_body:
            line_cost = len(safe_line) + 1
            if remaining_lines <= 0 or line_cost > remaining_chars:
                section_omitted += 1
                continue
            kept.append(safe_line)
            remaining_lines -= 1
            remaining_chars = max(0, remaining_chars - line_cost)
        omitted_lines += section_omitted
        visible.append(
            DiffFile(
                path=path,
                status=status,
                additions=additions,
                deletions=deletions,
                lines=tuple(kept),
                omitted_lines=section_omitted,
            )
        )

    omitted_files = max(0, len(raw_sections) - len(visible))
    return EditDiffSummary(
        files=tuple(visible),
        additions=total_additions,
        deletions=total_deletions,
        total_files=len(raw_sections),
        omitted_files=omitted_files,
        omitted_lines=omitted_lines,
        truncated=bool(omitted_files or omitted_lines),
    )


def build_edit_diff_summary(
    tool_name: str,
    result: str | None,
    *,
    function_args: dict[str, Any] | None = None,
    snapshot: Any = None,
    max_files: int = 6,
    max_lines: int = 80,
    max_chars: int = 6000,
) -> EditDiffSummary | None:
    """Extract and parse a successful edit tool's diff."""
    if str(tool_name or "") not in _EDIT_TOOLS:
        return None
    from agent.display import extract_edit_diff
    diff = extract_edit_diff(str(tool_name), result, function_args=function_args, snapshot=snapshot)
    if not diff:
        return None
    summary = parse_unified_diff(
        diff,
        max_files=max_files,
        max_lines=max_lines,
        max_chars=max_chars,
        redact=True,
    )
    if summary is None or snapshot is None:
        return summary

    # ``difflib.unified_diff`` uses ordinary a/path and b/path labels for a
    # snapshot-backed create/delete, so /dev/null is unavailable to the parser.
    # Recover those statuses from the captured before-state and current disk.
    status_by_path: dict[str, str] = {}
    for raw_path in snapshot.paths:
        path = Path(raw_path)
        key = str(path.resolve(strict=False))
        before = snapshot.before.get(str(path))
        exists_after = path.exists()
        if before is None and exists_after:
            status_by_path[key] = "added"
        elif before is not None and not exists_after:
            status_by_path[key] = "deleted"
    if not status_by_path:
        return summary
    rewritten = tuple(replace(file, status=status_by_path.get(
        str(Path(file.path).resolve(strict=False)), file.status)) for file in summary.files)
    return replace(summary, files=rewritten)
