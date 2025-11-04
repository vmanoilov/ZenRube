import json
from typing import Any, Dict, Tuple
from unittest.mock import patch

import pytest

import zenrube
from zenrube.models import SYNTHESIS_STYLES


def _mock_response(style: str) -> Tuple[str, Dict[str, Any]]:
    return (f"{style} consensus", {"stage": "synthesis"})


@pytest.mark.parametrize("style", SYNTHESIS_STYLES)
@patch("zenrube.invoke_llm")
def test_consensus_styles(mock_llm, style):
    def side_effect(prompt: str, **_: Any):
        if prompt.startswith("You are orchestrating"):
            return _mock_response(style)
        return (f"{style} expert", {"stage": "analysis"})

    mock_llm.side_effect = side_effect
    result = zenrube.zen_consensus(
        "Should we adopt event-driven architecture?",
        synthesis_style=style,
        use_cache=False,
    )

    assert result["synthesis_style"] == style
    assert result["consensus"] == f"{style} consensus"
    assert len(result["responses"]) == 3
    for response in result["responses"]:
        assert response["response"] == f"{style} expert"


@patch("zenrube.invoke_llm")
def test_parallel_execution(mock_llm):
    def side_effect(prompt: str, **_: Any):
        if prompt.startswith("You are orchestrating"):
            return ("Parallel consensus", {"stage": "synthesis"})
        return (prompt.split("\n")[0], {"stage": "analysis"})

    mock_llm.side_effect = side_effect
    result = zenrube.zen_consensus(
        "How should we roll out feature flags?",
        synthesis_style="collaborative",
        parallel=True,
        use_cache=False,
    )

    assert result["metadata"]["parallel_execution"] is True
    prompts = {response["prompt"] for response in result["responses"]}
    assert len(prompts) == 3


@patch("zenrube.invoke_llm")
def test_sequential_execution(mock_llm):
    call_order = []

    def side_effect(prompt: str, **_: Any):
        if prompt.startswith("You are orchestrating"):
            return ("Sequential consensus", {"stage": "synthesis"})
        call_order.append(prompt)
        return (prompt, {"stage": "analysis"})

    mock_llm.side_effect = side_effect
    result = zenrube.zen_consensus(
        "How do we design our deployment pipeline?",
        parallel=False,
        use_cache=False,
    )

    assert result["metadata"]["parallel_execution"] is False
    expected_prompts = [response["prompt"] for response in result["responses"]]
    assert call_order == expected_prompts


@patch("zenrube.invoke_llm")
def test_degraded_mode(mock_llm):
    call_count = 0

    def side_effect(prompt: str, **_: Any):
        nonlocal call_count
        if prompt.startswith("You are orchestrating"):
            return ("Recovered consensus", {"stage": "synthesis"})
        call_count += 1
        if call_count == 2:
            raise RuntimeError("LLM timeout")
        return (f"Expert {call_count}", {"stage": "analysis"})

    mock_llm.side_effect = side_effect
    result = zenrube.zen_consensus(
        "How do we secure our APIs?",
        synthesis_style="balanced",
        use_cache=False,
    )

    assert result["degraded"] is True
    assert any("degraded" in warning for warning in result["warnings"])
    errors = []
    for response in result["responses"]:
        if response["error"]:
            errors.append(response)
    assert len(errors) == 1


@patch("zenrube.invoke_llm")
def test_output_structure(mock_llm):
    def side_effect(prompt: str, **_: Any):
        if prompt.startswith("You are orchestrating"):
            return ("Structured consensus", {"stage": "synthesis"})
        return ("Structured response", {"stage": "analysis"})

    mock_llm.side_effect = side_effect
    result = zenrube.zen_consensus(
        "Should we use microservices?",
        synthesis_style="critical",
        use_cache=False,
    )

    assert set(result.keys()) >= {
        "execution_id",
        "question",
        "synthesis_style",
        "responses",
        "consensus",
        "timestamp",
        "warnings",
    }
    assert result["question"] == "Should we use microservices?"
    assert isinstance(result["responses"], list)
    assert all(
        {"name", "prompt", "provider"}.issubset(response.keys())
        for response in result["responses"]
    )
    json.dumps(result)  # ensure serialisable
