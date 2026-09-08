"""Read configured presets without resolving or rewriting their model aliases."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class ModePreset:
    name: str
    model_target: str
    reasoning: str
    fast_mode: bool


def available_mode_names(config: Mapping) -> list[str]:
    presets = config.get("mode_presets") or {}
    return list(presets) if isinstance(presets, Mapping) else []


def resolve_mode_preset(config: Mapping, requested_name: str) -> ModePreset | None:
    name = requested_name.strip().lower()
    presets = config.get("mode_presets") or {}
    if not isinstance(presets, Mapping):
        return None
    selected = next((value for key, value in presets.items() if str(key).lower() == name), None)
    if not isinstance(selected, Mapping):
        return None
    model, reasoning = selected.get("model"), selected.get("reasoning")
    fast = selected.get("fast_mode", False)
    if not isinstance(model, str) or not model.strip() or not isinstance(reasoning, str) or not reasoning.strip():
        return None
    if not isinstance(fast, bool):
        return None
    return ModePreset(name, model.strip(), reasoning.strip().lower(), fast)
