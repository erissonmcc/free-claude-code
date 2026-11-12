import pytest

from free_claude_code.application.context_runtime import RequestContextRuntime
from free_claude_code.application.context_manager import InMemoryCheckpointStore
from free_claude_code.config.settings import Settings
from free_claude_code.core.failures import ExecutionFailure, FailureKind
from free_claude_code.core.context_hook import (
    install_context_preparer,
    prepare_outbound_context,
    reset_context_preparer,
)


@pytest.mark.asyncio
async def test_google_prefix_observation_is_scoped_and_tracks_instructions_not_tail():
    from free_claude_code.core.context_hook import (
        install_usage_recorder, record_provider_usage, reset_usage_recorder,
    )

    async def resolver(_):
        raise AssertionError("observation must not call a model")

    records = []

    async def record(data):
        records.append(dict(data))

    async def observe(system, tail, *, session="thread-a", auth="secret-a", enabled=True, endpoint="https://vertex-a.example/project-a"):
        headers = {}
        if session:
            headers["x-fcc-session-id"] = session
        if auth:
            headers["authorization"] = auth
        runtime = RequestContextRuntime(
            Settings(context_management_enabled=enabled), resolver,
            request_id="req", request_headers=headers,
            model_info_lookup=lambda *_: None,
        )
        body = {"model": "gemini-test", "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": tail},
        ], "tools": [{"type": "function", "function": {"name": "read"}}]}
        assert await runtime.prepare("chat", body, "vertex", endpoint_scope=endpoint) == body
        await record_provider_usage(provider="vertex", model="gemini-test")
        return records[-1]

    token = install_usage_recorder(record)
    try:
        first = await observe("fixed instruction", "hello")
        changed_tail = await observe("fixed instruction", "new question")
        changed_system = await observe("changed instruction", "hello")
        other_session = await observe("fixed instruction", "hello", session="thread-b")
        other_auth = await observe("fixed instruction", "hello", auth="secret-b")
        other_endpoint = await observe("fixed instruction", "hello", endpoint="https://vertex-b.example/project-b")
        disabled = await observe("fixed instruction", "hello", enabled=False)
        no_session = await observe("fixed instruction", "hello", session=None)
        no_auth = await observe("fixed instruction", "hello", auth=None)
    finally:
        reset_usage_recorder(token)

    key = first.get("cache_prefix_key")
    assert isinstance(key, str) and len(key) == 64
    assert first["cache_prefix_tokens"] > 0
    assert changed_tail["cache_prefix_key"] == key == disabled["cache_prefix_key"]
    assert changed_system["cache_prefix_key"] != key
    assert other_session["cache_prefix_key"] != key
    assert other_auth["cache_prefix_key"] != key
    assert other_endpoint["cache_prefix_key"] != key
    assert first["cache_endpoint_key"] != other_endpoint["cache_endpoint_key"]
    assert first["cache_observation_version"] == 2
    assert no_session.get("cache_prefix_key") is None
    assert no_auth.get("cache_prefix_key") is None
    assert "secret-a" not in str(first) and "fixed instruction" not in str(first)


@pytest.mark.asyncio
async def test_display_provider_name_uses_catalog_context_window():
    from free_claude_code.application.model_metadata import ProviderModelInfo

    lookups = []

    def lookup(provider, model):
        lookups.append((provider, model))
        if provider == "openai":
            return ProviderModelInfo(model_id=model, context_window_tokens=100)
        return None

    async def resolver(_provider):
        raise AssertionError("pinned history must fail before any summary call")

    runtime = RequestContextRuntime(
        Settings(context_response_reserve_tokens=10),
        resolver,
        request_id="display-provider",
        request_headers={},
        model_info_lookup=lookup,
        checkpoint_store=InMemoryCheckpointStore(),
    )
    with pytest.raises(ExecutionFailure):
        await runtime.prepare("responses", {
            "model": "model-a",
            "instructions": "fixed instructions " * 100,
            "input": [{"role": "user", "content": "hello"}],
        }, "OpenAI")
    assert lookups == [("openai", "model-a")]


@pytest.mark.asyncio
async def test_display_provider_compacts_and_records_real_savings(monkeypatch):
    from free_claude_code.application.context_manager import StructuredSummary
    from free_claude_code.application.model_metadata import ProviderModelInfo
    from free_claude_code.core.context_hook import (
        install_usage_recorder, record_provider_usage, reset_usage_recorder,
    )

    async def resolver(_provider):
        raise AssertionError("use the stub summary without provider I/O")

    def lookup(provider, model):
        assert provider == "openai"
        return ProviderModelInfo(model_id=model, context_window_tokens=510)

    runtime = RequestContextRuntime(
        Settings(context_min_recent_messages=1, context_response_reserve_tokens=10),
        resolver, request_id="display-provider-summary", request_headers={},
        model_info_lookup=lookup, checkpoint_store=InMemoryCheckpointStore(),
    )

    async def summarize(provider, model, groups, target_tokens):
        assert provider == "openai"
        return StructuredSummary(current_task="keep task")

    monkeypatch.setattr(runtime, "_summarize", summarize)
    records = []

    async def record(data):
        records.append(dict(data))

    token = install_usage_recorder(record)
    try:
        body = {"model": "model-a", "input": [
            {"role": "user", "content": "old detail " * 600},
            {"role": "assistant", "content": "old result " * 600},
            {"role": "user", "content": "recent question"},
        ]}
        prepared = await runtime.prepare("responses", body, "OpenAI")
        await record_provider_usage(provider="OpenAI", model="model-a")
    finally:
        reset_usage_recorder(token)
    assert len(prepared["input"]) < len(body["input"])
    assert records[0]["estimated_messages_after"] < records[0]["estimated_messages_before"]
    assert records[0]["estimated_tokens_saved"] > 0
    assert records[0]["internal_summary_calls"] == 1


@pytest.mark.asyncio
async def test_large_responses_history_is_summarized_in_bounded_batches(monkeypatch):
    from free_claude_code.application.context_manager import StructuredSummary
    from free_claude_code.application.model_metadata import ProviderModelInfo

    async def resolver(_provider):
        raise AssertionError("use the stub summary without provider I/O")

    runtime = RequestContextRuntime(
        Settings(
            context_max_input_tokens=1000,
            context_min_recent_messages=1,
            context_response_reserve_tokens=100,
        ),
        resolver,
        request_id="large-responses-history",
        request_headers={},
        model_info_lookup=lambda _provider, model: ProviderModelInfo(
            model_id=model, context_window_tokens=1100
        ),
        checkpoint_store=InMemoryCheckpointStore(),
    )
    summary_calls = []

    async def summarize(provider, model, groups, target_tokens):
        summary_calls.append(sum(len(str(message)) for group in groups for message in group.messages))
        return StructuredSummary(
            current_task="continue the active task",
            decisions=("preserve the verified earlier decision",),
        )

    monkeypatch.setattr(runtime, "_summarize", summarize)
    body = {
        "model": "model-a",
        # The connected Codex provider strips this field upstream and sends
        # the complete input list, so FCC must still compact the visible history.
        "previous_response_id": "resp_previous",
        "input": [
            {"role": "user", "content": f"old detail {index} " + "x" * 2400}
            for index in range(60)
        ]
        + [{"role": "user", "content": "current request"}],
    }

    prepared = await runtime.prepare("responses", body, "OpenAI")

    assert len(summary_calls) >= 2
    assert len(prepared["input"]) < len(body["input"])
    assert prepared["input"][-1] == body["input"][-1]
    assert any("preserve the verified earlier decision" in str(item) for item in prepared["input"])


@pytest.mark.asyncio
async def test_runtime_leaves_short_outbound_payload_unchanged():
    async def resolver(_provider):
        raise AssertionError("summary provider must not be resolved")

    original = {"model": "model-a", "messages": [{"role": "user", "content": "hi"}]}
    runtime = RequestContextRuntime(
        Settings(context_max_input_tokens=1000),
        resolver,
        request_id="req-short",
        request_headers={},
        model_info_lookup=lambda *_args: None,
    )
    prepared = await runtime.prepare("chat", original, "test")
    assert prepared == original
    assert prepared is not original
    assert prepared["messages"] is original["messages"]


@pytest.mark.asyncio
async def test_runtime_rejects_over_budget_pinned_history_before_provider_io():
    async def resolver(_provider):
        raise AssertionError("provider must not be called when no safe group can be removed")

    runtime = RequestContextRuntime(
        Settings(context_max_input_tokens=1, context_min_recent_messages=1),
        resolver,
        request_id="req-over-budget",
        request_headers={},
        model_info_lookup=lambda *_args: None,
    )
    with pytest.raises(ExecutionFailure) as failure:
        await runtime.prepare(
            "chat",
            {"model": "model-a", "messages": [{"role": "user", "content": "large input"}]},
            "test",
        )
    assert failure.value.kind is FailureKind.CONTEXT_WINDOW_EXCEEDED


@pytest.mark.asyncio
async def test_runtime_budget_counts_stable_instructions_and_tool_schemas():
    async def resolver(_provider):
        raise AssertionError("provider must not be called")

    runtime = RequestContextRuntime(
        Settings(context_max_input_tokens=1, context_min_recent_messages=1),
        resolver,
        request_id="req-large-prefix",
        request_headers={},
        model_info_lookup=lambda *_args: None,
    )
    body = {
        "model": "model-a",
        "messages": [{"role": "user", "content": "hi"}],
        "system": "stable instruction " * 200,
        "tools": [{"name": "tool", "description": "schema " * 200}],
    }
    with pytest.raises(ExecutionFailure):
        await runtime.prepare("chat", body, "test")


@pytest.mark.asyncio
async def test_summary_call_uses_bypass_and_keeps_recent_turn():
    hook_calls = []
    summary_prompts = []

    class SummaryProvider:
        def stream_messages(self, request, **_kwargs):
            async def stream():
                summary_prompts.append(request.messages[0].content)
                await prepare_outbound_context(
                    "messages", request.model_dump(mode="json"), "test"
                )
                yield (
                    'data: {"type":"message_start","message":{"usage":{"input_tokens":1}}}\n\n'
                    'data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":"{\\"current_task\\":\\"keep task\\",\\"requirements\\":[],\\"preferences\\":[],\\"constraints\\":[],\\"decisions\\":[\\"decision\\"],\\"files\\":[],\\"changes\\":[],\\"errors_and_fixes\\":[],\\"pending_work\\":[],\\"tool_result_references\\":[]}"}}\n\n'
                    'data: {"type":"content_block_stop","index":0}\n\n'
                    'data: {"type":"message_delta","delta":{"stop_reason":"end_turn"},"usage":{"output_tokens":10}}\n\n'
                    'data: {"type":"message_stop"}\n\n'
                )
            return stream()

    async def resolver(_provider):
        return SummaryProvider()

    settings = Settings(
        context_max_input_tokens=300,
        context_min_recent_messages=1,
        context_response_reserve_tokens=10,
    )
    runtime = RequestContextRuntime(
        settings,
        resolver,
        request_id="req-summary",
        request_headers={"authorization": "Bearer test"},
        model_info_lookup=lambda *_args: None,
        checkpoint_store=InMemoryCheckpointStore(),
    )

    async def hook(protocol, body, provider):
        hook_calls.append(protocol)
        return await runtime.prepare(protocol, body, provider)

    token = install_context_preparer(hook)
    try:
        body = {
            "model": "model-a",
            "messages": [
                {"role": "user", "content": "old detail " * 600},
                {"role": "assistant", "content": "old result " * 600},
                {"role": "user", "content": "recent question"},
            ],
        }
        prepared = await runtime.prepare("chat", body, "test")
        next_turn = await runtime.prepare(
            "chat",
            {
                **prepared,
                "messages": [*prepared["messages"], {"role": "user", "content": "next"}],
            },
            "test",
        )
    finally:
        reset_context_preparer(token)
    assert prepared["messages"][-1] == body["messages"][-1]
    assert len(prepared["messages"]) == 2
    assert hook_calls == []
    assert len(next_turn["messages"]) == 3
    assert next_turn["messages"][0] == prepared["messages"][0]
    assert len(summary_prompts) == 1


@pytest.mark.asyncio
async def test_runtime_bounds_large_tool_result_even_without_model_window_metadata():
    async def resolver(_provider):
        raise AssertionError("a safe tool result bound does not need a summary call")

    runtime = RequestContextRuntime(
        Settings(context_max_tool_result_tokens=20),
        resolver,
        request_id="req-large-tool",
        request_headers={"authorization": "Bearer large-tool-test"},
        model_info_lookup=lambda *_args: None,
    )
    original = "begin " + ("x" * 1000) + " END_ERROR_MARKER"
    body = {
        "model": "model-a",
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": original},
        ],
    }
    prepared = await runtime.prepare("chat", body, "test")
    result = prepared["messages"][1]["content"]
    assert len(result) < len(original)
    assert result.startswith("begin ")
    assert result.endswith("END_ERROR_MARKER")
    assert "tool result limited" in result
    assert body["messages"][1]["content"] == original


@pytest.mark.asyncio
async def test_large_tool_transaction_fits_a_bounded_summary_batch(monkeypatch):
    from free_claude_code.application.context_manager import StructuredSummary
    from free_claude_code.application.model_metadata import ProviderModelInfo

    async def resolver(_provider):
        raise AssertionError("use the stub summary without provider I/O")

    runtime = RequestContextRuntime(
        Settings(
            context_max_input_tokens=100,
            context_max_tool_result_tokens=1000,
            context_min_recent_messages=1,
            context_response_reserve_tokens=10,
        ),
        resolver,
        request_id="large-tool-transaction",
        request_headers={},
        model_info_lookup=lambda _provider, model: ProviderModelInfo(
            model_id=model, context_window_tokens=110
        ),
        checkpoint_store=InMemoryCheckpointStore(),
    )
    summary_calls = []

    async def summarize(provider, model, groups, target_tokens):
        summary_calls.append(groups)
        return StructuredSummary(current_task="retain the tool result finding")

    monkeypatch.setattr(runtime, "_summarize", summarize)
    body = {
        "model": "model-a",
        "messages": [
            {"role": "assistant", "tool_calls": [{"id": "call-1"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "begin " + "x" * 100000 + " END_ERROR_MARKER"},
            {"role": "user", "content": "current request"},
        ],
    }

    prepared = await runtime.prepare("chat", body, "openai")

    assert len(summary_calls) == 1
    assert prepared["messages"][-1] == body["messages"][-1]
    assert len(prepared["messages"]) < len(body["messages"])
    assert "retain the tool result finding" in str(prepared["messages"])


def test_growth_budget_limits_cross_turn_growth_but_not_same_request_retry():
    async def resolver(_provider):
        raise AssertionError

    growth_estimates = {}
    runtime = RequestContextRuntime(
        Settings(context_growth_limit_percent=25),
        resolver,
        request_id="turn-next",
        request_headers={"x-fcc-session-id": "growth-test"},
        model_info_lookup=lambda *_args: None,
        growth_estimates=growth_estimates,
    )
    scope = runtime._session_key({})
    assert scope is not None
    key = (scope, "chat", "test", "model-a")
    growth_estimates[key] = ("turn-previous", 200)
    policy, prior = runtime._policy_for_request(scope, "chat", "test", "model-a")
    assert policy.max_input_tokens == 250
    assert prior == 200

    same_turn = RequestContextRuntime(
        Settings(context_growth_limit_percent=25),
        resolver,
        request_id="turn-previous",
        request_headers={"x-fcc-session-id": "growth-test"},
        model_info_lookup=lambda *_args: None,
        growth_estimates=growth_estimates,
    )
    policy, prior = same_turn._policy_for_request(scope, "chat", "test", "model-a")
    assert policy.max_input_tokens is None
    assert prior is None


def test_session_growth_identity_is_scoped_by_authentication():
    async def resolver(_provider):
        raise AssertionError

    def scope(authorization):
        return RequestContextRuntime(
            Settings(),
            resolver,
            request_id="request",
            request_headers={
                "authorization": authorization,
                "x-fcc-session-id": "reused-session-id",
            },
            model_info_lookup=lambda *_args: None,
        )._session_key({})

    assert scope("Bearer user-a") != scope("Bearer user-b")
