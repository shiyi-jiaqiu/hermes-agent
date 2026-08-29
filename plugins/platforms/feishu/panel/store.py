"""SQLite persistence and optimistic concurrency for Feishu panels."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Optional

from .state import PanelState


class PanelStateStore:
    """Small synchronous store suitable for the SDK's callback thread.

    SQLite transactions provide cross-thread/process compare-and-set semantics.
    Calls only read or write one compact JSON row and stay out of the Agent lane.
    """

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        # One adapter owns one store for the process lifetime. Reusing a single
        # lock-protected connection avoids opening SQLite and renegotiating WAL
        # mode several times for every button callback.
        self._connection = sqlite3.connect(
            str(self.path),
            timeout=0.25,
            check_same_thread=False,
        )
        self._connection.execute("PRAGMA busy_timeout=250")
        self._connection.execute("PRAGMA journal_mode=WAL")
        self._connection.execute("PRAGMA synchronous=NORMAL")
        self._initialize()

    def _initialize(self) -> None:
        with self._lock, self._connection as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS feishu_panels (
                    panel_id TEXT PRIMARY KEY,
                    revision INTEGER NOT NULL,
                    active INTEGER NOT NULL,
                    lifecycle TEXT NOT NULL,
                    updated_at REAL NOT NULL,
                    expires_at REAL NOT NULL,
                    state_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS feishu_active_panels (
                    scope_key TEXT PRIMARY KEY,
                    panel_id TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_feishu_panels_expiry
                    ON feishu_panels(expires_at);
                """
            )

    @staticmethod
    def _serialize(state: PanelState) -> str:
        return json.dumps(state.to_dict(), ensure_ascii=False, separators=(",", ":"))

    @staticmethod
    def _deserialize(raw: str) -> PanelState:
        return PanelState.from_dict(json.loads(raw))

    def get(self, panel_id: str) -> Optional[PanelState]:
        with self._lock, self._connection as connection:
            row = connection.execute(
                "SELECT state_json FROM feishu_panels WHERE panel_id = ?",
                (panel_id,),
            ).fetchone()
        return self._deserialize(row[0]) if row else None

    def get_active_id(self, scope_key: str) -> str:
        with self._lock, self._connection as connection:
            row = connection.execute(
                "SELECT panel_id FROM feishu_active_panels WHERE scope_key = ?",
                (scope_key,),
            ).fetchone()
        return str(row[0]) if row else ""

    def is_active(self, state: PanelState) -> bool:
        return bool(state.active and state.lifecycle == "active" and self.get_active_id(state.scope_key) == state.panel_id)

    def create_active(self, state: PanelState) -> Optional[PanelState]:
        """Install state as active and atomically replace the prior panel."""
        replaced: Optional[PanelState] = None
        now = time.time()
        state.active = True
        state.lifecycle = "active"
        state.updated_at = now
        with self._lock, self._connection as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT panel_id FROM feishu_active_panels WHERE scope_key = ?",
                (state.scope_key,),
            ).fetchone()
            old_id = str(row[0]) if row else ""
            if old_id and old_id != state.panel_id:
                old_row = connection.execute(
                    "SELECT state_json FROM feishu_panels WHERE panel_id = ?",
                    (old_id,),
                ).fetchone()
                if old_row:
                    replaced = self._deserialize(old_row[0])
                    replaced.active = False
                    replaced.lifecycle = "replaced"
                    replaced.busy_action_id = ""
                    replaced.busy_started_at = 0.0
                    replaced.revision += 1
                    replaced.touch()
                    connection.execute(
                        "UPDATE feishu_panels SET revision=?, active=0, lifecycle=?, updated_at=?, state_json=? WHERE panel_id=?",
                        (
                            replaced.revision,
                            replaced.lifecycle,
                            replaced.updated_at,
                            self._serialize(replaced),
                            replaced.panel_id,
                        ),
                    )
            connection.execute(
                "INSERT OR REPLACE INTO feishu_panels(panel_id, revision, active, lifecycle, updated_at, expires_at, state_json) VALUES(?,?,?,?,?,?,?)",
                (
                    state.panel_id,
                    state.revision,
                    1,
                    state.lifecycle,
                    state.updated_at,
                    state.expires_at,
                    self._serialize(state),
                ),
            )
            connection.execute(
                "INSERT OR REPLACE INTO feishu_active_panels(scope_key, panel_id) VALUES(?,?)",
                (state.scope_key, state.panel_id),
            )
            connection.commit()
        return replaced

    def compare_and_set(self, expected_revision: int, state: PanelState) -> bool:
        state.touch()
        with self._lock, self._connection as connection:
            cursor = connection.execute(
                """
                UPDATE feishu_panels
                   SET revision=?, active=?, lifecycle=?, updated_at=?, expires_at=?, state_json=?
                 WHERE panel_id=? AND revision=?
                """,
                (
                    state.revision,
                    int(state.active),
                    state.lifecycle,
                    state.updated_at,
                    state.expires_at,
                    self._serialize(state),
                    state.panel_id,
                    expected_revision,
                ),
            )
            if cursor.rowcount != 1:
                return False
            if not state.active:
                connection.execute(
                    "DELETE FROM feishu_active_panels WHERE scope_key=? AND panel_id=?",
                    (state.scope_key, state.panel_id),
                )
            connection.commit()
            return True

    def delete(self, panel_id: str) -> None:
        with self._lock, self._connection as connection:
            row = connection.execute(
                "SELECT state_json FROM feishu_panels WHERE panel_id=?", (panel_id,)
            ).fetchone()
            if row:
                state = self._deserialize(row[0])
                connection.execute(
                    "DELETE FROM feishu_active_panels WHERE scope_key=? AND panel_id=?",
                    (state.scope_key, panel_id),
                )
            connection.execute("DELETE FROM feishu_panels WHERE panel_id=?", (panel_id,))

    def prune(self, now: float | None = None) -> int:
        cutoff = float(now or time.time())
        with self._lock, self._connection as connection:
            rows = connection.execute(
                "SELECT panel_id, state_json FROM feishu_panels WHERE expires_at <= ?",
                (cutoff,),
            ).fetchall()
            for panel_id, raw in rows:
                state = self._deserialize(raw)
                connection.execute(
                    "DELETE FROM feishu_active_panels WHERE scope_key=? AND panel_id=?",
                    (state.scope_key, panel_id),
                )
            connection.execute("DELETE FROM feishu_panels WHERE expires_at <= ?", (cutoff,))
            connection.commit()
        return len(rows)

    def close(self) -> None:
        with self._lock:
            self._connection.close()
