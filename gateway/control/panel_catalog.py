"""Visibility policy for the Feishu model catalog."""

from __future__ import annotations

from typing import Any


HIDDEN_PANEL_PROVIDER_SLUGS = frozenset(
    {
        "gemini",
        "google",
        "google-ai-studio",
        "copilot",
        "github",
        "github-copilot",
        "moa",
        "mixture",
        "mixture-of-agents",
    }
)
HIDDEN_OPENROUTER_MODEL_VENDORS = frozenset({"anthropic", "openai", "google"})


def is_hidden_panel_provider(provider: Any) -> bool:
    """Return whether a provider row must be hidden from the Panel."""
    return str(provider or "").strip().lower() in HIDDEN_PANEL_PROVIDER_SLUGS


def is_hidden_openrouter_model(provider: Any, model: Any) -> bool:
    """Return whether an OpenRouter model belongs to a hidden vendor namespace."""
    if str(provider or "").strip().lower() != "openrouter":
        return False
    vendor, separator, _model_name = str(model or "").strip().partition("/")
    return bool(separator and vendor.strip().lower() in HIDDEN_OPENROUTER_MODEL_VENDORS)
