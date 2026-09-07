"""Shared resolution for configured model/reasoning mode presets.

The preset name is part of the user's configuration, while ``/quick`` is the
stable command for the panel's Quick model. Older configurations stored that
panel preset under ``mode_presets.fast`` because ``fast`` was originally the
internal name. Keep that compatibility mapping here so gateway, panel, TUI,
and desktop all resolve the same target.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class ModePreset:
    """A validated preset plus the resolved model metadata."""

    requested_name: str
    config_name: str
    model_target: str
    expected_model: str
    expected_provider: str
    expected_base_url: str
    reasoning: str
    fast_mode: bool

    @property
    def expected_reasoning(self) -> str:
        """Return the effective reasoning value used by verification."""
        if self.reasoning in {
            "provider",
            "provider-managed",
            "provider_managed",
            "auto",
        }:
            return "none"
        return self.reasoning


def _mode_presets(config: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if not isinstance(config, Mapping):
        return {}
    raw = config.get("mode_presets")
    return raw if isinstance(raw, Mapping) else {}


def resolve_model_reference(
    config: Mapping[str, Any] | None,
    target: str,
) -> tuple[str, str, str]:
    """Resolve a configured model alias to ``(model, provider, base_url)``."""
    model_target = str(target or "").strip()
    if not isinstance(config, Mapping):
        return model_target, "", ""
    raw_aliases = config.get("model_aliases")
    aliases = raw_aliases if isinstance(raw_aliases, Mapping) else {}
    alias_spec = aliases.get(model_target)
    if alias_spec is None:
        alias_spec = next(
            (
                value
                for name, value in aliases.items()
                if str(name).strip().lower() == model_target.lower()
            ),
            None,
        )
    if isinstance(alias_spec, Mapping):
        return (
            str(alias_spec.get("model") or model_target).strip(),
            str(alias_spec.get("provider") or "").strip(),
            str(alias_spec.get("base_url") or "").strip(),
        )
    if alias_spec:
        return str(alias_spec).strip(), "", ""
    return model_target, "", ""


def available_mode_names(config: Mapping[str, Any] | None) -> list[str]:
    """Return configured names plus the stable ``quick`` compatibility name."""
    presets = _mode_presets(config)
    names = [str(name) for name in presets]
    if "quick" not in {name.lower() for name in names} and "fast" in {
        name.lower() for name in names
    }:
        # Put the public alias next to the configured Quick/fast entry without
        # changing the order users see for the actual config keys.
        names.insert(0, "quick")
    return names


def resolve_mode_preset(
    config: Mapping[str, Any] | None,
    requested_name: str,
) -> ModePreset | None:
    """Resolve a mode name, including the legacy panel Quick alias.

    Exact configured names win. Only ``quick`` falls back to ``fast``; the
    latter remains the independent `/fast` command at the slash-dispatch level.
    """
    requested = str(requested_name or "").strip().lower()
    if not requested:
        return None

    presets = _mode_presets(config)
    config_name = requested
    selected = presets.get(config_name)
    if not isinstance(selected, Mapping) and requested == "quick":
        legacy_name = next(
            (str(name) for name in presets if str(name).strip().lower() == "fast"),
            None,
        )
        if legacy_name is not None and isinstance(presets.get(legacy_name), Mapping):
            config_name = legacy_name
            selected = presets[legacy_name]

    if not isinstance(selected, Mapping):
        return None

    model_target = str(selected.get("model") or "").strip()
    reasoning = str(selected.get("reasoning") or "").strip().lower()
    if not model_target or not reasoning:
        return None

    expected_model, expected_provider, expected_base_url = resolve_model_reference(
        config, model_target
    )

    return ModePreset(
        requested_name=requested,
        config_name=str(config_name),
        model_target=model_target,
        expected_model=expected_model,
        expected_provider=expected_provider,
        expected_base_url=expected_base_url,
        reasoning=reasoning,
        fast_mode=bool(selected.get("fast_mode", False)),
    )


def mode_command_fast_value(preset: ModePreset, argument: str | None = None) -> bool:
    """Resolve an optional ``fast|normal|off`` modifier for a preset."""
    raw = str(argument or "").strip().lower()
    if not raw:
        return preset.fast_mode
    return raw == "fast"


def format_mode_verification(
    *,
    expected_model: str,
    expected_provider: str,
    expected_reasoning: str,
    expected_fast: bool,
    actual_model: str,
    actual_provider: str,
    actual_reasoning: str,
    actual_fast: bool,
) -> str:
    """Format safe model-state diagnostics for a failed mode verification."""
    expected_route = expected_model
    if expected_provider:
        expected_route += f"/{expected_provider}"
    actual_route = actual_model or "unknown"
    if actual_provider:
        actual_route += f"/{actual_provider}"
    return (
        "expected "
        f"{expected_route}/{expected_reasoning}/Fast {'on' if expected_fast else 'off'}; "
        "actual "
        f"{actual_route}/{actual_reasoning}/Fast {'on' if actual_fast else 'off'}"
    )
