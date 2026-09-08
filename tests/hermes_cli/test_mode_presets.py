"""The old name is migrated once on disk; runtime resolution preserves aliases."""
import pytest

from hermes_cli.mode_presets import available_mode_names, resolve_mode_preset
from hermes_cli.runtime_settings import mode_request
from scripts.migrate_feishu_settings import migrate


def test_migration_is_idempotent_and_preserves_user_choices():
    config = {"mode_presets": {"fast": {"model": "flash-cpa", "reasoning": "high"}},
              "feishu_panel": {"hidden_providers": ["mine"]}, "unrelated": {"keep": True}}
    migrated, changes = migrate(config)
    assert changes and "fast" in config["mode_presets"]  # input untouched
    assert available_mode_names(migrated) == ["quick"]
    assert migrated["feishu_panel"]["hidden_providers"] == ["mine"]
    assert migrated["unrelated"] == {"keep": True}
    assert migrate(migrated) == (migrated, [])
    preset = resolve_mode_preset(migrated, "quick")
    assert preset.model_target == "flash-cpa" and preset.reasoning == "high"
    assert mode_request(migrated, "quick").model_target == "flash-cpa"


@pytest.mark.parametrize("modifier", ["typo", "true", "low"])
def test_unknown_modifier_is_rejected_instead_of_silently_disabling_fast(modifier):
    with pytest.raises(ValueError, match="Usage"):
        mode_request({"mode_presets": {"quick": {"model": "alias", "reasoning": "high"}}}, "quick", modifier)
