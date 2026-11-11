from free_claude_code.providers.openai_responses.prompt_cache import (
    add_explicit_prompt_cache_breakpoint,
)


def test_marks_stable_instructions_and_keeps_dynamic_suffix_out_of_write():
    body = {
        "model": "gpt-6.1-sol",
        "instructions": "stable " + ("reference " * 1800),
        "input": [{"role": "user", "content": "new question"}],
    }

    result = add_explicit_prompt_cache_breakpoint(body)

    assert "instructions" not in result
    assert result["input"][0] == {
        "role": "developer",
        "content": [{
            "type": "input_text",
            "text": body["instructions"],
            "prompt_cache_breakpoint": {"mode": "explicit", "ttl": "30m"},
        }],
    }
    assert result["input"][1] == body["input"][0]
    assert result["prompt_cache_options"] == {"mode": "implicit", "ttl": "30m"}


def test_does_not_add_openai_explicit_markers_for_unsupported_or_small_prefix():
    for body in (
        {"model": "gpt-4.1", "instructions": "x" * 20, "input": []},
        {"model": "gpt-6.1-sol", "instructions": "short", "input": []},
        {"model": "gemini-3.8-flash", "instructions": "x" * 5000, "input": []},
    ):
        assert add_explicit_prompt_cache_breakpoint(body) == body


def test_does_not_inject_cache_options_for_unsupported_luna_models():
    for model in ("gpt-5.6-luna", "gpt-6-luna"):
        body = {
            "model": model,
            "instructions": "reference " * 1800,
            "input": [{"role": "user", "content": "question"}],
        }

        assert add_explicit_prompt_cache_breakpoint(body) == body


def test_preserves_caller_cache_options_and_never_duplicates_instructions():
    instructions = "shared " + ("reference " * 1800)
    body = {
        "model": "gpt-6.1-sol",
        "instructions": instructions,
        "input": [{"role": "user", "content": "question"}],
        "prompt_cache_options": {"mode": "implicit", "ttl": "12h"},
    }

    result = add_explicit_prompt_cache_breakpoint(body)

    assert result["prompt_cache_options"] == body["prompt_cache_options"]
    serialized = str(result)
    assert serialized.count(instructions) == 1


def test_returns_unchanged_when_explicit_caching_is_already_selected():
    body = {
        "model": "gpt-6.1-sol",
        "instructions": "reference " * 1800,
        "input": [],
        "prompt_cache_options": {"mode": "explicit"},
    }

    assert add_explicit_prompt_cache_breakpoint(body) == body
