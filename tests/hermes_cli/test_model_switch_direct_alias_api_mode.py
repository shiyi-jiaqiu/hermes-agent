"""Regression tests for transport selection on configured direct aliases."""

from unittest.mock import patch
import pytest

from hermes_cli.model_switch import DirectAlias, switch_model


_VALIDATION = {
    "accepted": True,
    "persist": True,
    "recognized": True,
    "message": None,
}


@pytest.mark.parametrize("alias_mode,provider_mode", [("", "codex_responses"), ("codex_responses", "chat_completions")])
def test_direct_alias_keeps_named_provider_api_mode(alias_mode, provider_mode):
    """A generic alias endpoint must keep its provider-declared transport.

    ``cpa-gemini`` is intentionally hosted at a generic OpenAI-compatible URL,
    so URL detection returns ``chat_completions``. Its configured provider,
    however, declares ``codex_responses``. A direct alias must not erase that
    explicit mode before the final URL fallback runs.
    """
    alias = DirectAlias(
        model="gemini-3.8-flash-high",
        provider="cpa-gemini",
        base_url="https://cpa.example.test/v1", api_mode=alias_mode,
    )
    runtime = {
        "api_key": "test-key",
        "base_url": alias.base_url,
        "api_mode": provider_mode,
        "capabilities": {},
    }
    user_providers = {
        "cpa-gemini": {
            "name": "CPA Gemini Responses",
            "api": alias.base_url,
            "api_mode": provider_mode,
        }
    }

    with (
        patch(
            "hermes_cli.model_switch.resolve_alias",
            return_value=(alias.provider, alias.model, "flash-cpa"),
        ),
        patch(
            "hermes_cli.model_switch.DIRECT_ALIASES",
            {"flash-cpa": alias},
        ),
        patch("hermes_cli.model_switch._ensure_direct_aliases"),
        patch(
            "hermes_cli.runtime_provider.resolve_runtime_provider",
            return_value=runtime,
        ),
        patch(
            "hermes_cli.models_validate.validate_requested_model",
            return_value=_VALIDATION,
        ),
        patch("hermes_cli.model_switch.get_model_info", return_value=None),
        patch("hermes_cli.model_switch.get_model_capabilities", return_value=None),
        patch("hermes_cli.models.detect_provider_for_model", return_value=None),
    ):
        result = switch_model(
            raw_input="flash-cpa",
            current_provider="openai-codex",
            current_model="gpt-5.6-sol",
            current_base_url="https://chatgpt.com/backend-api/codex",
            user_providers=user_providers,
            custom_providers=[],
        )

        from hermes_cli.runtime_settings import RuntimeSettings, SettingsRequest, prepare_settings
        settings = prepare_settings(RuntimeSettings("gpt-5.6-sol", "openai-codex"),
                                    SettingsRequest(model_target="flash-cpa", reasoning="high"),
                                    {"providers": user_providers})
        assert (settings.model, settings.provider, settings.base_url, settings.api_mode) == (
            result.new_model, result.target_provider, result.base_url, result.api_mode)

    assert result.success, result.error_message
    assert result.target_provider == "cpa-gemini"
    assert result.new_model == "gemini-3.8-flash-high"
    assert result.base_url == alias.base_url
    assert result.api_mode == "codex_responses"
