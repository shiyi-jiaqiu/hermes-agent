"""Pure navigation transitions for the Feishu panel state machine."""

from __future__ import annotations

from .actions import PanelAction, PanelActionError
from .state import PanelState


NAVIGABLE_VIEWS = frozenset({
    "home",
    "model",
    "model_provider",
    "reasoning",
    "reasoning_global",
    "confirm_global_reasoning",
    "sessions",
    "status",
    "confirm_new",
})
PAGED_VIEWS = frozenset({"model", "model_provider", "sessions"})


def reduce_panel_state(state: PanelState, action: PanelAction) -> PanelState:
    """Return a cloned state after a navigation-only operation."""
    new_state = state.clone()
    if action.op == "nav":
        if action.target not in NAVIGABLE_VIEWS:
            raise PanelActionError("invalid navigation target")
        if action.target != new_state.view:
            new_state.view_stack.append(new_state.view)
        new_state.view = action.target
        new_state.page = 0
    elif action.op == "back":
        new_state.view = new_state.view_stack.pop() if new_state.view_stack else "home"
        new_state.page = 0
    elif action.op == "home":
        new_state.view = "home"
        new_state.view_stack.clear()
        new_state.page = 0
    elif action.op == "page":
        if new_state.view not in PAGED_VIEWS or action.page is None:
            raise PanelActionError("page operation is not valid here")
        new_state.page = action.page
    elif action.op == "close":
        new_state.active = False
        new_state.lifecycle = "closed"
        new_state.busy_action_id = ""
        new_state.busy_started_at = 0.0
        new_state.view = "home"
        new_state.view_stack.clear()
    elif action.op not in {"refresh", "select"}:
        raise PanelActionError("operation is not a navigation transition")
    return new_state
