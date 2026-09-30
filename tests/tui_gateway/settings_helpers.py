"""Complete session fixtures for the deferred runtime-settings transaction."""
import threading


def bind_settings_session(monkeypatch, server, sid, session, db):
    session.setdefault("session_key", sid)
    session.setdefault("history", [])
    session.setdefault("history_lock", threading.Lock())
    session.setdefault("agent_build_lock", threading.Lock())
    db.ensure_session(session["session_key"], source="desktop")
    monkeypatch.setitem(server._sessions, sid, session)
    monkeypatch.setattr(server, "_get_db", lambda: db)
    monkeypatch.setattr(server, "_emit", lambda *a, **k: None)
    monkeypatch.setattr(server, "_session_info", lambda *a, **k: {})
    return session
