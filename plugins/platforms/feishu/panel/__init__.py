"""Stateful Feishu control-panel implementation.

The panel package intentionally contains no LLM or generic message-pipeline
integration. Card navigation is reduced and rendered synchronously; trusted
Hermes controls are executed through the gateway control service.
"""

from .actions import PanelAction, PanelActionError, parse_panel_action
from .controller import FeishuPanelController, PanelCallbackResult
from .state import PanelState
from .store import PanelStateStore

__all__ = [
    "FeishuPanelController",
    "PanelAction",
    "PanelActionError",
    "PanelCallbackResult",
    "PanelState",
    "PanelStateStore",
    "parse_panel_action",
]
