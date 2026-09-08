from gateway.tool_progress_diff import parse_unified_diff


def test_hunk_body_is_counted_even_when_it_looks_like_file_headers():
    result = parse_unified_diff(
        '--- a/file.txt\n+++ b/file.txt\n@@ -1 +1 @@\n--- old\n+++ new\n'
        '--- a/second.txt\n+++ b/second.txt\n@@ -0,0 +1 @@\n+added\n',
        redact=False,
    )
    assert [(f.path, f.additions, f.deletions) for f in result.files] == [
        ('file.txt', 1, 1), ('second.txt', 1, 0),
    ]
    assert (result.additions, result.deletions) == (2, 1)

import json
from gateway.tool_progress_diff import build_edit_diff_summary

def test_parse_unified_diff_classifies_files_and_counts_changes():
    diff = """--- /dev/null
+++ b/new.py
@@ -0,0 +1,2 @@
+one
+two
--- a/old.py
+++ /dev/null
@@ -1 +0,0 @@
-gone
--- a/edit.py
+++ b/edit.py
@@ -1 +1 @@
-old
+new
"""
    summary = parse_unified_diff(diff, redact=False)
    assert summary is not None
    assert summary.total_files == 3
    assert summary.additions == 3
    assert summary.deletions == 2
    assert [(item.status, item.path) for item in summary.files] == [
        ("added", "new.py"),
        ("deleted", "old.py"),
        ("modified", "edit.py"),
    ]

def test_parse_unified_diff_caps_visible_files_lines_and_chars():
    diff = "".join(
        f"--- a/f{i}.py\n+++ b/f{i}.py\n@@ -1 +1 @@\n-old-{i}\n+new-{i}\n"
        for i in range(4)
    )
    summary = parse_unified_diff(
        diff, redact=False, max_files=2, max_lines=3, max_chars=1000
    )
    assert summary is not None
    assert summary.total_files == 4
    assert len(summary.files) == 2
    assert summary.omitted_files == 2
    assert summary.omitted_lines > 0
    assert summary.truncated is True

def test_parse_unified_diff_redacts_secret_shaped_content():
    secret = "sk-" + "a" * 32
    summary = parse_unified_diff(
        f"--- a/.env\n+++ b/.env\n@@ -1 +1 @@\n-old\n+OPENAI_API_KEY={secret}\n"
    )
    assert summary is not None
    visible = "\n".join(line for file in summary.files for line in file.lines)
    assert secret not in visible

def test_build_edit_diff_summary_reuses_patch_result_diff(tmp_path):
    target = tmp_path / "demo.py"
    raw = json.dumps(
        {
            "success": True,
            "diff": (
                f"--- a/{target}\n+++ b/{target}\n"
                "@@ -1 +1 @@\n-old = 1\n+new = 2\n"
            ),
        }
    )
    summary = build_edit_diff_summary("patch", raw, function_args={"path": str(target)})
    assert summary is not None
    assert summary.additions == 1
    assert summary.deletions == 1
    assert summary.files[0].status == "modified"

def test_build_edit_diff_summary_marks_snapshot_backed_new_file_added(tmp_path):
    from agent.display import capture_local_edit_snapshot

    target = tmp_path / "new.py"
    snapshot = capture_local_edit_snapshot("write_file", {"path": str(target)})
    target.write_text("created = True\n", encoding="utf-8")
    summary = build_edit_diff_summary(
        "write_file",
        '{"verified": true}',
        function_args={"path": str(target)},
        snapshot=snapshot,
    )
    assert summary is not None
    assert summary.files[0].status == "added"
    assert summary.additions == 1
