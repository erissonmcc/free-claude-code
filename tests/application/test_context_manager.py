from __future__ import annotations

import pytest

from free_claude_code.application.context_manager import (
    ContextManager,
    ContextPolicy,
    ContextRequest,
    ContextCompactor,
    InMemoryCheckpointStore,
    MessageGroup,
    StructuredSummary,
    deduplicate_tool_outputs,
    group_message_transactions,
    _summarize_in_bounded_batches,
    _summary_fragments,
)


@pytest.mark.asyncio
async def test_summary_batch_accepts_single_message_within_internal_prompt_limit():
    # A tool result may be capped at 20k tokens (~80k chars), so the batch
    # character guard must match the internal summarizer's 4x character cap.
    groups = (MessageGroup(messages=({"role": "tool", "content": "x" * 50000},)),)

    class Generator:
        calls = 0

        async def summarize(self, source_groups, *, target_tokens):
            self.calls += 1
            assert source_groups == groups
            return StructuredSummary(current_task="large result summarized")

    generator = Generator()
    summary, calls = await _summarize_in_bounded_batches(
        groups,
        summary_generator=generator,
        target_tokens=2048,
        max_source_chars=20000 * 4,
        max_levels=0,
    )

    assert summary.current_task == "large result summarized"
    assert calls == 1
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_summary_batches_fragment_oversized_atomic_group_without_mutating_it():
    original = MessageGroup(
        messages=(
            {"role": "assistant", "tool_calls": [{"id": "call-1", "type": "function"}]},
            {"role": "tool", "tool_call_id": "call-1", "content": "x" * 12000},
        ),
        kind="tool",
        complete=True,
    )

    class Generator:
        batches = []

        async def summarize(self, source_groups, *, target_tokens):
            self.batches.append(source_groups)
            encoded = __import__("json").dumps(
                [group.messages for group in source_groups], ensure_ascii=False
            )
            assert len(encoded) <= 4096
            return StructuredSummary(current_task="transaction summarized")

    generator = Generator()
    summary, calls = await _summarize_in_bounded_batches(
        (original,),
        summary_generator=generator,
        target_tokens=256,
        max_source_chars=4096,
        max_levels=0,
    )

    assert summary.current_task == "transaction summarized"
    assert calls == len(generator.batches)
    assert calls > 1
    assert all(
        group.kind == "summary_fragment"
        for batch in generator.batches
        for group in batch
    )
    # Fragmentation is summary-only: the source transaction remains intact.
    assert original.kind == "tool"
    assert original.complete is True
    assert len(original.messages) == 2
    assert original.messages[1]["content"] == "x" * 12000


def test_short_history_is_returned_unchanged_and_keeps_estimate_separate_from_usage():
    messages = (
        {"role": "user", "content": "Keep this exact request."},
        {"role": "assistant", "content": "Acknowledged."},
    )
    request = ContextRequest(
        provider="gemini",
        model="gemini-3.8-flash",
        protocol="chat",
        messages=messages,
        input_window_tokens=100,
        requested_output_tokens=20,
    )
    manager = ContextManager(
        ContextPolicy(max_input_tokens=80, response_reserve_tokens=10),
        estimator=lambda current: 12,
    )

    decision = manager.prepare(request)

    assert decision.messages is messages
    assert decision.input_budget_tokens == 80
    assert decision.over_budget is False
    assert decision.measurement.estimated_input_tokens == 12
    assert decision.measurement.actual_input_tokens is None
    assert decision.measurement.cached_input_tokens is None


def test_budget_reserves_the_larger_of_requested_output_and_configured_reserve():
    messages = ({"role": "user", "content": "x"},)
    manager = ContextManager(
        ContextPolicy(max_input_tokens=90, response_reserve_tokens=12),
        estimator=lambda current: 75,
    )

    decision = manager.prepare(
        ContextRequest(
            provider="vertex",
            model="gemini-model",
            protocol="chat",
            messages=messages,
            input_window_tokens=90,
            requested_output_tokens=20,
        )
    )

    assert decision.input_budget_tokens == 70
    assert decision.over_budget is True
    assert decision.messages == messages


def test_unknown_model_window_uses_only_explicit_input_budget():
    messages = ({"role": "user", "content": "x"},)
    manager = ContextManager(
        ContextPolicy(max_input_tokens=None, response_reserve_tokens=12),
        estimator=lambda current: 75,
    )

    decision = manager.prepare(
        ContextRequest(
            provider="custom",
            model="unknown-window-model",
            protocol="chat",
            messages=messages,
            input_window_tokens=None,
            requested_output_tokens=None,
        )
    )

    assert decision.input_budget_tokens is None
    assert decision.over_budget is False


@pytest.mark.asyncio
async def test_old_encrypted_responses_reasoning_does_not_block_safe_compaction():
    from free_claude_code.application.context_adapters import responses

    body = {
        "model": "model-a",
        "input": [
            {"type": "message", "role": "user", "content": "old task details " * 20},
            {"type": "reasoning", "id": "rs_old", "summary": [], "encrypted_content": "cipher" * 1000},
            {"type": "message", "role": "user", "content": "current question"},
        ],
    }
    request = responses.to_context_request(
        body, "openai", "model-a", input_window_tokens=501
    )
    policy = ContextPolicy(
        max_input_tokens=500,
        response_reserve_tokens=1,
        minimum_recent_messages=1,
        summary_target_tokens=200,
    )
    calls = []

    class Generator:
        async def summarize(self, groups, *, target_tokens):
            calls.extend(groups)
            return StructuredSummary(current_task="continue old task")

    result = await ContextCompactor(policy).compact(
        request,
        request.groups,
        estimate_context=lambda groups, summary: sum(
            len(str(message)) for group in groups for message in group.messages
        ) + (len(str(summary)) if summary is not None else 0),
        summary_generator=Generator(),
        checkpoint_store=InMemoryCheckpointStore(),
        stable_prefix_digest=request.stable_prefix_digest,
        scope_key="test",
        policy_version="1",
    )

    assert result.over_budget is False
    assert result.estimated_after_tokens < result.estimated_before_tokens
    assert body["input"][1] not in [message for group in result.groups for message in group.messages]
    assert all("cipher" not in str(group) for group in calls)
    assert body["input"][-1] in [message for group in result.groups for message in group.messages]


@pytest.mark.asyncio
async def test_unconsolidatable_summary_is_used_when_it_still_fits_context_budget():
    messages = (
        {"type": "message", "role": "user", "content": "old detail " * 80},
        {"type": "message", "role": "user", "content": "current request"},
    )
    groups = tuple(MessageGroup((message,)) for message in messages)
    request = ContextRequest(
        provider="openai",
        model="gpt-6.1-sol",
        protocol="responses",
        messages=messages,
        input_window_tokens=None,
        requested_output_tokens=None,
        groups=groups,
    )
    preserved = StructuredSummary(current_task="retain the important result")

    class Generator:
        async def summarize(self, _groups, *, target_tokens):
            return preserved

    policy = ContextPolicy(
        max_input_tokens=300,
        response_reserve_tokens=1,
        minimum_recent_messages=1,
        summary_target_tokens=1,
    )
    result = await ContextCompactor(policy).compact(
        request,
        groups,
        estimate_context=lambda current, summary: sum(
            len(str(message)) for group in current for message in group.messages
        ) + (len(str(summary)) if summary is not None else 0),
        summary_generator=Generator(),
        checkpoint_store=InMemoryCheckpointStore(),
        stable_prefix_digest="stable",
        scope_key="test",
        policy_version="1",
    )

    assert result.over_budget is False
    assert result.summary == preserved
    assert result.estimated_after_tokens <= 300
    assert messages[-1] in [message for group in result.groups for message in group.messages]


def test_chat_tool_call_and_results_form_one_complete_group():
    messages = (
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call-a", "type": "function"},
                {"id": "call-b", "type": "function"},
            ],
        },
        {"role": "tool", "tool_call_id": "call-a", "content": "A"},
        {"role": "tool", "tool_call_id": "call-b", "content": "B"},
        {"role": "assistant", "content": "Finished"},
    )

    groups = group_message_transactions("chat", messages)

    assert groups == (
        MessageGroup(messages=messages[:3], kind="tool", complete=True),
        MessageGroup(messages=messages[3:], kind="ordinary", complete=True),
    )


def test_incomplete_chat_tool_transaction_is_pinned_as_a_unit():
    messages = (
        {"role": "assistant", "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "other-call", "content": "unknown"},
    )

    groups = group_message_transactions("chat", messages)

    assert len(groups) == 1
    assert groups[0].messages == messages
    assert groups[0].complete is False
    assert groups[0].pinned is True


@pytest.mark.parametrize(
    ("protocol", "messages", "expected_type"),
    [
        (
            "messages",
            (
                {"role": "assistant", "content": [{"type": "tool_use", "id": "t1"}]},
                {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "ok"}]},
                {"role": "assistant", "content": "done"},
            ),
            "ordinary",
        ),
        (
            "responses",
            (
                {"type": "function_call", "call_id": "r1", "name": "read"},
                {"type": "function_call_output", "call_id": "r1", "output": "ok"},
                {"type": "message", "role": "assistant", "content": "done"},
            ),
            "ordinary",
        ),
    ],
)
def test_message_and_responses_protocols_group_call_result_pairs(
    protocol, messages, expected_type
):
    groups = group_message_transactions(protocol, messages)

    assert groups[0].kind == "tool"
    assert groups[0].complete is True
    assert groups[0].messages == messages[:2]
    assert groups[1].kind == expected_type


def test_deduplicating_tool_outputs_keeps_both_call_result_pairs_valid():
    messages = (
        {"role": "assistant", "tool_calls": [{"id": "call-a"}]},
        {"role": "tool", "tool_call_id": "call-a", "content": "same output"},
        {"role": "assistant", "tool_calls": [{"id": "call-b"}]},
        {"role": "tool", "tool_call_id": "call-b", "content": "same output"},
    )

    deduplicated, removed_count = deduplicate_tool_outputs("chat", messages)

    assert removed_count == 1
    assert deduplicated[1] == messages[1]
    assert deduplicated[3]["tool_call_id"] == "call-b"
    assert deduplicated[3]["content"] == "[duplicate tool result; see message 2]"
    assert messages[3]["content"] == "same output"


def test_anthropic_deduplication_keeps_unrelated_parallel_result_blocks():
    messages = (
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "a", "content": "same"},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "b", "content": "same"},
                {"type": "tool_result", "tool_use_id": "c", "content": "unique"},
            ],
        },
    )

    deduplicated, removed_count = deduplicate_tool_outputs("messages", messages)

    assert removed_count == 1
    assert deduplicated[1]["content"][0] == {
        "type": "tool_result",
        "tool_use_id": "b",
        "content": "[duplicate tool result; see message 1]",
    }
    assert deduplicated[1]["content"][1] == messages[1]["content"][1]


def test_structured_summary_preserves_required_task_state_categories():
    summary = StructuredSummary.model_validate(
        {
            "current_task": "Implement context budgeting",
            "requirements": ["Preserve recent messages"],
            "preferences": ["Keep provider prefix stable"],
            "constraints": ["Do not mutate stored history"],
            "decisions": ["Compact old completed turns"],
            "files": ["src/free_claude_code/application/context_manager.py"],
            "changes": ["Added a bounded manager"],
            "errors_and_fixes": ["No progress stops the compactor"],
            "pending_work": ["Integrate final transport hooks"],
            "tool_result_references": ["result-7"],
        }
    )

    assert summary.current_task == "Implement context budgeting"
    assert summary.requirements == ("Preserve recent messages",)
    assert summary.files == ("src/free_claude_code/application/context_manager.py",)
    assert summary.errors_and_fixes == ("No progress stops the compactor",)
    assert summary.pending_work == ("Integrate final transport hooks",)


@pytest.mark.asyncio
async def test_compactor_reuses_the_same_checkpoint_without_resummarizing():
    groups = (
        MessageGroup(messages=({"role": "user", "content": "old decision"},)),
        MessageGroup(messages=({"role": "user", "content": "current task"},)),
    )
    request = ContextRequest(
        provider="gemini",
        model="gemini-3.8-flash",
        protocol="chat",
        messages=tuple(message for group in groups for message in group.messages),
        input_window_tokens=100,
        requested_output_tokens=10,
    )

    class Generator:
        calls = 0

        async def summarize(self, source_groups, *, target_tokens):
            self.calls += 1
            return StructuredSummary(
                current_task="current task",
                decisions=("old decision",),
            )

    generator = Generator()
    store = InMemoryCheckpointStore(max_entries=8)
    compactor = ContextCompactor(
        ContextPolicy(
            max_input_tokens=60,
            response_reserve_tokens=10,
            minimum_recent_messages=1,
        )
    )

    async def estimate(source_groups, summary):
        return 100 if summary is None else 30

    first = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=store,
        stable_prefix_digest="stable-system-prefix",
        scope_key="session-1",
        policy_version="v1",
    )
    second = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=store,
        stable_prefix_digest="stable-system-prefix",
        scope_key="session-1",
        policy_version="v1",
    )

    assert first.summary == second.summary
    assert first.estimated_before_tokens == 100
    assert first.estimated_after_tokens == 30
    assert second.internal_summary_calls == 0
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_compactor_stops_when_summary_makes_no_progress():
    groups = (
        MessageGroup(messages=({"role": "user", "content": "old"},)),
        MessageGroup(messages=({"role": "user", "content": "current"},)),
    )
    request = ContextRequest(
        provider="anthropic",
        model="claude-model",
        protocol="messages",
        messages=tuple(message for group in groups for message in group.messages),
        input_window_tokens=100,
        requested_output_tokens=10,
    )

    class Generator:
        calls = 0

        async def summarize(self, source_groups, *, target_tokens):
            self.calls += 1
            return StructuredSummary(current_task="old current")

    generator = Generator()
    compactor = ContextCompactor(
        ContextPolicy(
            max_input_tokens=60,
            response_reserve_tokens=10,
            minimum_recent_messages=1,
            max_compaction_passes=4,
        )
    )

    async def estimate(source_groups, summary):
        return 100

    result = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=InMemoryCheckpointStore(max_entries=8),
        stable_prefix_digest="stable-prefix",
        scope_key="session-2",
        policy_version="v1",
    )

    assert result.over_budget is True
    assert result.stop_reason == "no_progress"
    assert result.internal_summary_calls == 1
    assert generator.calls == 1


@pytest.mark.asyncio
async def test_compactor_deduplicates_old_tool_results_before_summarizing():
    groups = (
        MessageGroup(
            messages=(
                {"role": "assistant", "tool_calls": [{"id": "call-a"}]},
                {"role": "tool", "tool_call_id": "call-a", "content": "repeat"},
            ),
            kind="tool",
        ),
        MessageGroup(
            messages=(
                {"role": "assistant", "tool_calls": [{"id": "call-b"}]},
                {"role": "tool", "tool_call_id": "call-b", "content": "repeat"},
            ),
            kind="tool",
        ),
        MessageGroup(messages=({"role": "user", "content": "current"},)),
    )
    request = ContextRequest(
        provider="gemini",
        model="gemini-model",
        protocol="chat",
        messages=tuple(message for group in groups for message in group.messages),
        input_window_tokens=100,
        requested_output_tokens=10,
    )

    class Generator:
        calls = 0

        async def summarize(self, source_groups, *, target_tokens):
            self.calls += 1
            return StructuredSummary(current_task="not needed")

    generator = Generator()
    compactor = ContextCompactor(
        ContextPolicy(
            max_input_tokens=60,
            response_reserve_tokens=10,
            minimum_recent_messages=1,
        )
    )

    async def estimate(source_groups, summary):
        content = [
            message.get("content")
            for group in source_groups
            for message in group.messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        return 100 if content.count("repeat") == 2 else 50

    result = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=InMemoryCheckpointStore(max_entries=8),
        stable_prefix_digest="stable-prefix",
        scope_key="session-3",
        policy_version="v1",
    )

    assert result.duplicate_tool_outputs_removed == 1
    assert result.estimated_after_tokens == 50
    assert result.over_budget is False
    assert result.internal_summary_calls == 0
    assert generator.calls == 0
    assert result.groups[-1] == groups[-1]
    assert groups[1].messages[1]["content"] == "repeat"


@pytest.mark.asyncio
async def test_invalid_summary_schema_keeps_all_source_groups():
    groups = (
        MessageGroup(messages=({"role": "user", "content": "old"},)),
        MessageGroup(messages=({"role": "user", "content": "current"},)),
    )
    request = ContextRequest(
        provider="anthropic",
        model="claude-model",
        protocol="messages",
        messages=tuple(message for group in groups for message in group.messages),
        input_window_tokens=100,
        requested_output_tokens=10,
    )

    class Generator:
        async def summarize(self, source_groups, *, target_tokens):
            return {"decisions": [42]}

    compactor = ContextCompactor(
        ContextPolicy(
            max_input_tokens=60,
            response_reserve_tokens=10,
            minimum_recent_messages=1,
        )
    )

    async def estimate(source_groups, summary):
        return 100 if summary is None else 20

    result = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=Generator(),
        checkpoint_store=InMemoryCheckpointStore(max_entries=8),
        stable_prefix_digest="stable-prefix",
        scope_key="session-4",
        policy_version="v1",
    )

    assert result.groups == groups
    assert result.summary is None
    assert result.over_budget is True
    assert result.stop_reason == "summary_failed"


@pytest.mark.asyncio
async def test_compaction_pass_cap_prevents_summary_after_a_tool_dedup_pass():
    groups = (
        MessageGroup(
            messages=(
                {"role": "assistant", "tool_calls": [{"id": "call-a"}]},
                {"role": "tool", "tool_call_id": "call-a", "content": "same"},
            ),
            kind="tool",
        ),
        MessageGroup(
            messages=(
                {"role": "assistant", "tool_calls": [{"id": "call-b"}]},
                {"role": "tool", "tool_call_id": "call-b", "content": "same"},
            ),
            kind="tool",
        ),
        MessageGroup(messages=({"role": "user", "content": "current"},)),
    )
    request = ContextRequest(
        provider="vertex",
        model="gemini-model",
        protocol="chat",
        messages=tuple(message for group in groups for message in group.messages),
        input_window_tokens=100,
        requested_output_tokens=10,
    )

    class Generator:
        calls = 0

        async def summarize(self, source_groups, *, target_tokens):
            self.calls += 1
            return StructuredSummary(current_task="too late")

    generator = Generator()
    compactor = ContextCompactor(
        ContextPolicy(
            max_input_tokens=60,
            response_reserve_tokens=10,
            minimum_recent_messages=1,
            max_compaction_passes=1,
        )
    )

    async def estimate(source_groups, summary):
        content = [
            message.get("content")
            for group in source_groups
            for message in group.messages
            if isinstance(message, dict) and message.get("role") == "tool"
        ]
        return 100 if content.count("same") == 2 else 80

    result = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=InMemoryCheckpointStore(max_entries=8),
        stable_prefix_digest="stable-prefix",
        scope_key="session-5",
        policy_version="v1",
    )

    assert result.stop_reason == "max_passes"
    assert result.estimated_after_tokens == 80
    assert result.duplicate_tool_outputs_removed == 1
    assert result.internal_summary_calls == 0
    assert generator.calls == 0
