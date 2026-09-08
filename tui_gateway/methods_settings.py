"""Shared session settings RPCs for TUI and Desktop (registered on the server facade)."""
from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


class _TUISettingsEndpoint:
    def __init__(self, sid, session):
        self.sid, self.session = sid, session
        self.key = session["session_key"]
        self.retired_agent = self.retired_worker = None

    def read(self):
        from hermes_cli.runtime_settings import RuntimeSettings, reasoning_name
        session = self.session
        agent = session.get("agent")
        cfg = _load_cfg()
        model_cfg = cfg.get("model") or {}
        if isinstance(model_cfg, str):
            model_cfg = {"default": model_cfg}
        route = session.get("model_override") or {}
        if isinstance(route, str):
            route = {"model": route}
        def field(name, default=""):
            return route.get(name, getattr(agent, name, default)) or ""
        resume = session.get("resume_runtime_overrides") or {}
        rc = session.get("create_reasoning_override", resume.get("reasoning_config_override"))
        reasoning_inherited = rc is None
        if rc is None:
            from hermes_constants import resolve_reasoning_config
            rc = resolve_reasoning_config(cfg, field("model", model_cfg.get("default", "")))
        tier = session.get("create_service_tier_override", resume.get("service_tier_override"))
        if tier is None:
            tier = getattr(agent, "service_tier", _load_service_tier())
        return RuntimeSettings(field("model", model_cfg.get("default", "")),
                               field("provider", model_cfg.get("provider", "")),
                               field("base_url", model_cfg.get("base_url", "")),
                               field("api_mode", model_cfg.get("api_mode", "")),
                               reasoning_name(rc), tier or "normal", field("api_key"),
                               route.get("request_overrides", getattr(agent, "request_overrides", None)),
                               route.get("capabilities", getattr(agent, "capabilities", None)),
                               reasoning_inherited,
                               runtime_resolved=bool(agent or route),
                               temporary=session.get("one_turn_model_restore") is not None)

    def validate(self):
        session = self.session
        if _sessions.get(self.sid) is not session or session["session_key"] != self.key:
            raise ValueError("Session changed during settings resolution")
        if session.get("running") or session.get("_closing"):
            raise ValueError("Stop the running turn before changing settings")
        ready = session.get("agent_ready")
        if ready is not None and not ready.is_set() and session.get("agent_build_started"):
            raise ValueError("Agent is initializing; retry after it is ready")

    def persist(self, settings):
        session = self.session
        with _session_db(session) as db:
            if db is None:
                raise RuntimeError("Session database is unavailable")
            db.ensure_session(self.key, source=_session_source(session), model=self.read().model)
            db.update_runtime_settings(self.key, settings.persisted())

    def publish(self, settings):
        import threading
        session = self.session
        self.retired_agent = session.get("agent")
        self.retired_worker = session.get("slash_worker")
        session.update(model_override=settings.route(), create_reasoning_override=(
                           None if settings.reasoning_inherited else settings.reasoning_config()),
                       create_service_tier_override="" if settings.service_tier == "normal" else settings.service_tier,
                       agent=None, agent_error=None, slash_worker=None, agent_ready=threading.Event(), lazy=True)
        for key in ("agent_build_started", "_agent_build_thread", "pending_model_switch", "one_turn_model_restore"):
            session.pop(key, None)
        # A resumed session's build kwargs are read before the creation overrides.
        if "resume_runtime_overrides" in session:
            session["resume_runtime_overrides"] = {
                **session["resume_runtime_overrides"], "model_override": settings.route(),
                "provider_override": settings.provider, "reasoning_config_override": (
                    None if settings.reasoning_inherited else settings.reasoning_config()),
                "service_tier_override": "" if settings.service_tier == "normal" else settings.service_tier}

    def release_retired(self):
        for resource, method_name in ((self.retired_agent, "release_clients"), (self.retired_worker, "close")):
            if resource is not None:
                try:
                    getattr(resource, method_name)()
                except Exception:
                    logger.warning("Failed to release retired settings resource", exc_info=True)


def _apply_session_settings(sid, session, request, cfg, *, resolver=None):
    from hermes_cli.runtime_settings import SettingsResult, commit_settings, prepare_settings, settings_error
    endpoint = _TUISettingsEndpoint(sid, session)
    with _session_profile_runtime_scope(session):
        baseline = endpoint.read()
        try:
            candidate = prepare_settings(baseline, request, cfg, resolver=resolver)
        except Exception as exc:
            return SettingsResult(endpoint.read(), False, settings_error(exc))
        # Serialize publication with prompt claims, teardown and agent initialization.
        with _sessions_lock, session["history_lock"], session.setdefault("agent_build_lock", threading.Lock()):
            result = commit_settings(endpoint, baseline, candidate)
        endpoint.release_retired()
        if result.applied and result.changed:
            _emit("session.info", sid, _session_info(None, session))
        return result


@method("mode.apply")
@_profile_scoped
def _mode_apply(rid, params):
    from hermes_cli.runtime_settings import mode_request, parse_mode_command
    session, error = _sess_nowait(params, rid)
    if error:
        return error
    try:
        with _session_profile_runtime_scope(session):
            cfg = _load_cfg()
        name, modifier = parse_mode_command(str(params.get("command") or "/mode " + str(params.get("name") or "")))
        result = _apply_session_settings(params["session_id"], session, mode_request(cfg, name, modifier), cfg)
    except ValueError as exc:
        return _err(rid, 4002, str(exc))
    return _ok(rid, {"applied": result.applied, "actual": result.actual.persisted(),
                     "error": result.error, "output": result.text()})


def register(server):
    bind_module(globals(), server)
