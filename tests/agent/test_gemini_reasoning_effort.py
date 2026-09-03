"""Gemini thinking-level mapping on Responses and native-compatible wires."""

from agent.reasoning_effort import gemini_supported_efforts
from agent.transports import get_transport
import agent.transports.chat_completions  # noqa: F401
import agent.transports.codex  # noqa: F401


def _codex_kwargs(model: str, effort: str | None = None, **params):
    reasoning_config = None
    if effort is not None:
        reasoning_config = {"enabled": True, "effort": effort}
    transport = get_transport("codex_responses")
    assert transport is not None
    return transport.build_kwargs(
        model=model,
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        provider="cpa-gemini",
        base_url="https://cpa.example.test/v1",
        reasoning_config=reasoning_config,
        **params,
    )


def test_gemini_38_and_37_flash_support_low_medium_high_only():
    expected = ("low", "medium", "high")
    assert gemini_supported_efforts("gemini-3.8-flash-high") == expected
    assert gemini_supported_efforts("google/gemini-3.7-flash") == expected


def test_gemini_35_flash_and_flash_lite_support_minimal():
    expected = ("minimal", "low", "medium", "high")
    assert gemini_supported_efforts("gemini-3.5-flash") == expected
    assert gemini_supported_efforts("gemini-3.1-flash-lite") == expected


def test_gemini_pro_and_image_lite_use_their_narrower_vocabularies():
    assert gemini_supported_efforts("gemini-3.1-pro-preview") == (
        "low",
        "medium",
        "high",
    )
    assert gemini_supported_efforts("gemini-3.1-flash-lite-image") == (
        "minimal",
        "high",
    )
    assert gemini_supported_efforts("gemini-3-pro-preview") == ("low", "high")


def test_gemini_25_is_not_guessed_as_a_level_api():
    """Gemini 2.5 needs native thinkingBudget rather than a guessed level."""
    assert gemini_supported_efforts("gemini-2.5-flash") is None


def test_cpa_responses_emits_high_for_quick_gemini_model():
    kwargs = _codex_kwargs("gemini-3.8-flash-high", "high")
    assert kwargs["reasoning"] == {"effort": "high", "summary": "auto"}


def test_cpa_responses_clamps_codex_only_levels_to_gemini_levels():
    assert _codex_kwargs("gemini-3.8-flash-high", "max")["reasoning"]["effort"] == "high"
    assert _codex_kwargs("gemini-3.8-flash-high", "ultra")["reasoning"]["effort"] == "high"
    # ``minimal`` is not accepted by Gemini 3.8/3.7 Flash; low is its
    # nearest supported thinking level.
    assert _codex_kwargs("gemini-3.8-flash-high", "minimal")["reasoning"]["effort"] == "low"


def test_cpa_responses_preserves_panel_levels_supported_by_gemini():
    for effort in ("low", "medium", "high"):
        assert _codex_kwargs("gemini-3.8-flash-high", effort)["reasoning"]["effort"] == effort


def test_cpa_responses_keeps_explicit_disable_semantics():
    transport = get_transport("codex_responses")
    assert transport is not None
    kwargs = transport.build_kwargs(
        model="gemini-3.8-flash-high",
        messages=[{"role": "user", "content": "Hi"}],
        tools=[],
        provider="cpa-gemini",
        base_url="https://cpa.example.test/v1",
        reasoning_config={"enabled": False, "effort": "none"},
    )
    assert "reasoning" not in kwargs
    assert kwargs["include"] == []
