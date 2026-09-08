#!/usr/bin/env python3
"""One-time migration of the old Quick preset and personal Panel visibility policy.

This edits only local YAML. It does not contact Feishu or publish the bot menu.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
import shutil
from datetime import datetime, timezone

import yaml


# These values belonged to the old implementation. They are migration input,
# never runtime policy: subsequent edits are read solely from the user's YAML.
_LEGACY_HIDDEN_PROVIDERS = ["gemini", "google", "google-ai-studio", "copilot", "github",
                            "github-copilot", "moa", "mixture", "mixture-of-agents"]
_LEGACY_HIDDEN_PREFIXES = {"openrouter": ["anthropic/", "openai/", "google/"]}


def migrate(config: dict) -> tuple[dict, list[str]]:
    updated = deepcopy(config)
    changes = []
    presets = updated.get("mode_presets") or {}
    if "fast" in presets and "quick" not in presets:
        updated["mode_presets"] = {"quick" if key == "fast" else key: value for key, value in presets.items()}
        changes.append("mode_presets.fast → mode_presets.quick")
    panel = updated.setdefault("feishu_panel", {})
    for key, value in (("hidden_providers", _LEGACY_HIDDEN_PROVIDERS), ("hidden_model_prefixes", _LEGACY_HIDDEN_PREFIXES)):
        if key not in panel:
            panel[key] = deepcopy(value)
            changes.append(f"feishu_panel.{key}")
    return updated, changes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path.home() / ".hermes" / "config.yaml")
    parser.add_argument("--apply", action="store_true", help="write the migration (default: show changed key names)")
    args = parser.parse_args()
    raw = yaml.safe_load(args.config.read_text()) or {}
    if not isinstance(raw, dict):
        raise ValueError("config.yaml must contain a mapping")
    updated, changes = migrate(raw)
    if not changes:
        print("Already migrated; no changes.")
        return
    print("\n".join(changes))
    if args.apply:
        from hermes_cli.config import atomic_config_write
        backup = args.config.with_name(args.config.name + ".before-feishu-refactor-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        shutil.copy2(args.config, backup)
        atomic_config_write(args.config, updated)
        print(f"Saved; backup: {backup}")


if __name__ == "__main__":
    main()
