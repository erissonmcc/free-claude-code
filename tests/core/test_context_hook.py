import pytest

from free_claude_code.core.context_hook import (
    bypass_context_preparation,
    install_context_preparer,
    install_usage_recorder,
    prepare_outbound_context,
    record_provider_usage,
    reset_context_bypass,
    reset_context_preparer,
    reset_usage_recorder,
    set_context_usage_estimate,
)


@pytest.mark.asyncio
async def test_outbound_hook_receives_each_attempt_and_can_rewrite_body():
    calls = []

    async def prepare(protocol, body, provider):
        calls.append((protocol, dict(body), provider))
        return {**body, "messages": [*body["messages"], {"role": "user", "content": "x"}]}

    token = install_context_preparer(prepare)
    try:
        result = await prepare_outbound_context("chat", {"messages": []}, "vertex")
    finally:
        reset_context_preparer(token)
    assert result["messages"] == [{"role": "user", "content": "x"}]
    assert calls == [("chat", {"messages": []}, "vertex")]


@pytest.mark.asyncio
async def test_summary_bypass_does_not_reenter_context_hook():
    calls = []

    async def prepare(protocol, body, provider):
        calls.append(protocol)
        return dict(body)

    token = install_context_preparer(prepare)
    bypass = bypass_context_preparation()
    try:
        result = await prepare_outbound_context("messages", {"messages": []}, "vertex")
    finally:
        reset_context_bypass(bypass)
        reset_context_preparer(token)
    assert result == {"messages": []}
    assert calls == []


@pytest.mark.asyncio
async def test_provider_usage_keeps_unknown_cache_null_and_attaches_estimates():
    records = []

    async def record(values):
        records.append(dict(values))

    recorder = install_usage_recorder(record)
    set_context_usage_estimate(900, 400, 500)
    try:
        await record_provider_usage(
            provider="openai", model="model-a", protocol="responses",
            actual_input_tokens=400, normal_input_tokens=None,
            cached_input_tokens=None, output_tokens=50, total_tokens=450,
        )
    finally:
        reset_usage_recorder(recorder)

    assert records == [{
        "request_kind": "generation",
        "estimated_input_before": 900,
        "estimated_input_after": 400,
        "estimated_tokens_saved": 500,
        "estimated_messages_before": None,
        "estimated_messages_after": None,
        "internal_summary_calls": None,
        "provider": "openai",
        "model": "model-a",
        "protocol": "responses",
        "actual_input_tokens": 400,
        "normal_input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": 50,
        "total_tokens": 450,
    }]


@pytest.mark.asyncio
async def test_internal_summary_estimate_clears_main_prefix_observation():
    from free_claude_code.core.context_hook import set_context_cache_observation

    records = []

    async def record(values):
        records.append(dict(values))

    token = install_usage_recorder(record)
    try:
        set_context_cache_observation("a" * 64, 1000)
        set_context_usage_estimate(50, 50, 0)
        await record_provider_usage(provider="vertex", model="m")
    finally:
        reset_usage_recorder(token)
    assert records[0].get("cache_prefix_key") is None
