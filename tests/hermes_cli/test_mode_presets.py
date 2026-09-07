from hermes_cli.mode_presets import available_mode_names, resolve_mode_preset


def test_quick_resolves_to_the_panel_quick_preset_when_stored_as_fast():
    config = {
        "mode_presets": {
            "fast": {"model": "flash-cpa", "reasoning": "high", "fast_mode": False},
            "daily": {"model": "luna", "reasoning": "max", "fast_mode": False},
        },
        "model_aliases": {
            "flash-cpa": {
                "model": "gemini-3.8-flash-high",
                "provider": "cpa-gemini",
            }
        },
    }

    preset = resolve_mode_preset(config, "quick")

    assert preset is not None
    assert preset.requested_name == "quick"
    assert preset.config_name == "fast"
    assert preset.model_target == "flash-cpa"
    assert preset.expected_model == "gemini-3.8-flash-high"
    assert preset.expected_provider == "cpa-gemini"
    assert preset.expected_reasoning == "high"
    assert available_mode_names(config) == ["quick", "fast", "daily"]


def test_explicit_quick_preset_takes_precedence_over_legacy_fast():
    config = {
        "mode_presets": {
            "fast": {"model": "old", "reasoning": "low"},
            "quick": {"model": "new", "reasoning": "medium"},
        }
    }

    preset = resolve_mode_preset(config, "quick")

    assert preset is not None
    assert preset.config_name == "quick"
    assert preset.expected_model == "new"
