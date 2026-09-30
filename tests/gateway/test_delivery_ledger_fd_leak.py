"""Delivery operations release leases without closing the live transcript writer."""
from gateway import delivery_ledger as dl
from hermes_state_registry import acquire, release, stats


def test_ledger_operations_release_every_borrowed_connection(monkeypatch, tmp_path):
    monkeypatch.setattr(dl, "_db_path", lambda: tmp_path / "state.db")
    db = acquire(tmp_path / "state.db")
    conn = db._conn
    try:
        oid = dl.compute_obligation_id("sess", "msg", "content")
        dl.record_obligation(obligation_id=oid, session_key="sess", platform="telegram",
                             chat_id="123", thread_id=None, content="hello")
        dl.mark_attempting(oid)
        dl.mark_delivered(oid)
        dl.sweep_recoverable()
        assert db._conn is conn
        assert stats()["total_refcounts"] == 1
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
    finally:
        release(db)
    assert db._conn is None
    assert stats()["total_refcounts"] == 0
