"""Kanban housekeeping must borrow the process-wide state.db writer."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def _posix_locks_on(path: Path) -> set[tuple[str, ...]]:
    if not sys.platform.startswith("linux"):
        pytest.skip("lock-table probe requires /proc/locks")
    inode = path.stat().st_ino
    held = set()
    for line in Path("/proc/locks").read_text().splitlines():
        parts = line.split()
        try:
            owner = int(parts[4])
            locked_inode = int(parts[5].split(":")[2])
        except (IndexError, ValueError):
            continue
        if owner == os.getpid() and locked_inode == inode:
            held.add(tuple(parts[1:8]))
    return held


def test_legacy_retag_does_not_close_a_second_writer_or_lose_wal_generation(tmp_path, monkeypatch):
    """Exercise the field failure: helper close -> lost DMS lock -> external SELECT unlinks WAL."""
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))

    from hermes_cli import kanban_db_dispatch as dispatch
    from hermes_state_registry import acquire, close_all

    dispatch._retagged_workspace_roots.clear()
    db_path = tmp_path / "state.db"
    workspaces = tmp_path / "kanban" / "workspaces"
    db = acquire(db_path)
    try:
        db.create_session("legacy", source="cli", cwd=str(workspaces / "task"))
        wal_path = Path(str(db_path) + "-wal")
        wal_identity = (wal_path.stat().st_dev, wal_path.stat().st_ino)
        before = _posix_locks_on(db_path)
        assert before, "expected the live WAL writer to hold a DMS lock on state.db"

        dispatch._retag_legacy_worker_sessions(str(workspaces))

        after = _posix_locks_on(db_path)
        assert not (before - after), "kanban housekeeping cancelled the live writer's POSIX locks"
        subprocess.run(
            [sys.executable, "-c", "import sqlite3,sys; sqlite3.connect(sys.argv[1]).execute('SELECT 1').fetchone()", str(db_path)],
            check=True,
        )
        assert wal_path.exists()
        assert (wal_path.stat().st_dev, wal_path.stat().st_ino) == wal_identity
        db.append_message("legacy", role="user", content="post-housekeeping")
    finally:
        close_all()
        dispatch._retagged_workspace_roots.clear()
