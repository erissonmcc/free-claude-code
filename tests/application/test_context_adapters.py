from __future__ import annotations

import json

import pytest

from free_claude_code.application.context_manager import ContextCompactionResult
from free_claude_code.application.context_manager import StructuredSummary
from free_claude_code.application.context_adapters import chat, messages, responses


def _wire_bytes(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode()


def test_chat_stable_prefix_is_identical_when_only_the_last_turn_changes():
    base = {
        "model": "gemini-3.8-flash",
        "tools": [{"type": "function", "function": {"name": "read", "parameters": {}}}],
        "messages": [
            {"role": "system", "content": "Stable policy"},
            {"role": "developer", "content": "Stable project rules"},
            {"role": "user", "content": "first turn"},
        ],
    }
    next_turn = {
        **base,
        "messages": [*base["messages"][:2], {"role": "user", "content": "changed turn"}],
    }

    first = chat.to_context_request(base, "gemini", "gemini-3.8-flash")
    second = chat.to_context_request(next_turn, "gemini", "gemini-3.8-flash")

    assert first.stable_prefix_digest == second.stable_prefix_digest
    assert first.stable_message_count == 2
    assert _wire_bytes(base["messages"][:2]) == _wire_bytes(next_turn["messages"][:2])


def test_chat_tool_schema_change_invalidates_stable_prefix_digest():
    body = {
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "messages": [{"role": "user", "content": "task"}],
    }
    changed_tools = {
        **body,
        "tools": [{"type": "function", "function": {"name": "read", "description": "new"}}],
    }

    first = chat.to_context_request(body, "openai", "model-a")
    second = chat.to_context_request(changed_tools, "openai", "model-a")

    assert first.stable_prefix_digest != second.stable_prefix_digest


@pytest.mark.parametrize(
    ("adapter", "provider", "model", "body", "history_key"),
    [
        (
            chat,
            "openai",
            "model-a",
            {
                "tools": [],
                "messages": [
                    {"role": "assistant", "tool_calls": [{"id": "c1"}]},
                    {"role": "tool", "tool_call_id": "c1", "content": "ok"},
                ],
            },
            "messages",
        ),
        (
            messages,
            "anthropic",
            "claude-model",
            {
                "system": "rules",
                "tools": [],
                "messages": [
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "m1"}]},
                    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "m1", "content": "ok"}]},
                ],
            },
            "messages",
        ),
        (
            responses,
            "openai",
            "model-a",
            {
                "instructions": "rules",
                "tools": [],
                "input": [
                    {"type": "function_call", "call_id": "r1", "name": "read"},
                    {"type": "function_call_output", "call_id": "r1", "output": "ok"},
                ],
            },
            "input",
        ),
    ],
)
def test_protocol_adapters_keep_tool_transactions_atomic_and_ordered(
    adapter, provider, model, body, history_key
):
    request = adapter.to_context_request(body, provider, model)

    tool_groups = [group for group in request.groups if group.kind == "tool"]
    assert len(tool_groups) == 1
    assert tool_groups[0].complete is True
    assert tool_groups[0].messages[0] == body[history_key][0]
    assert tool_groups[0].messages[1] == body[history_key][1]
    decision = ContextCompactionResult(
        groups=request.groups,
        summary=None,
        estimated_before_tokens=20,
        estimated_after_tokens=20,
        over_budget=False,
    )
    output = adapter.apply_decision(body, decision)
    assert output[history_key] == body[history_key]


@pytest.mark.parametrize(
    ("adapter", "provider", "model", "body", "history_key", "retained_index"),
    [
        (
            messages,
            "anthropic",
            "claude-model",
            {
                "system": "fixed instructions",
                "messages": [
                    {"role": "user", "content": "old"},
                    {"role": "user", "content": "current"},
                ],
            },
            "messages",
            1,
        ),
        (
            responses,
            "openai",
            "model-a",
            {
                "instructions": "fixed instructions",
                "input": [
                    {"type": "message", "role": "developer", "content": "rules"},
                    {"type": "message", "role": "user", "content": "old"},
                    {"type": "message", "role": "user", "content": "current"},
                ],
            },
            "input",
            2,
        ),
    ],
)
def test_non_chat_adapters_insert_summary_after_stable_instructions(
    adapter, provider, model, body, history_key, retained_index
):
    request = adapter.to_context_request(body, provider, model)
    retained = tuple(
        group
        for index, group in enumerate(request.groups)
        if index < request.stable_message_count or index == retained_index
    )
    decision = ContextCompactionResult(
        groups=retained,
        summary=StructuredSummary(decisions=("preserve this prior decision",)),
        estimated_before_tokens=100,
        estimated_after_tokens=30,
        over_budget=False,
    )

    output = adapter.apply_decision(body, decision)

    stable_count = request.stable_message_count
    summary = output[history_key][stable_count]
    if history_key == "input":
        text = summary["content"][0]["text"]
    else:
        text = summary["content"]
    assert "preserve this prior decision" in text
    assert output[history_key][stable_count + 1] == body[history_key][-1]
    if history_key == "messages":
        assert output["system"] is body["system"]
    else:
        assert output["instructions"] is body["instructions"]


def test_responses_adapter_pins_unknown_items_but_marks_encrypted_reasoning_metadata():
    body = {
        "instructions": "stable instructions",
        "tools": [],
        "input": [
            {"type": "reasoning", "encrypted_content": "opaque-signature"},
            {"type": "vendor_future_item", "payload": {"keep": True}},
            {"type": "message", "role": "user", "content": "current"},
        ],
    }

    request = responses.to_context_request(body, "openai", "model-a")
    compacted = ContextCompactionResult(
        groups=request.groups,
        summary=None,
        estimated_before_tokens=100,
        estimated_after_tokens=100,
        over_budget=False,
    )
    output = responses.apply_decision(body, compacted)

    assert request.groups[0].kind == "reasoning_metadata"
    assert request.groups[0].pinned is False
    assert request.groups[1].kind == "opaque"
    assert output["input"] == body["input"]
    assert output["input"][0] is body["input"][0]


def test_responses_adapter_marks_native_encrypted_reasoning_as_separate_metadata():
    body = {
        "input": [
            {"type": "reasoning", "id": "rs_old", "summary": [], "encrypted_content": "cipher"},
            {"type": "message", "role": "user", "content": "current"},
        ]
    }

    request = responses.to_context_request(body, "openai", "model-a")

    assert request.groups[0].kind == "reasoning_metadata"
    assert request.groups[0].pinned is False
    assert request.groups[1].pinned is False

    other_provider = responses.to_context_request(body, "custom", "model-a")
    assert other_provider.groups[0].kind == "opaque"
    assert other_provider.groups[0].pinned is True


def test_responses_adapter_compacts_valid_messages_without_explicit_type():
    body = {
        "instructions": "stable instructions",
        "input": [
            {"role": "user", "content": "old details"},
            {"role": "assistant", "content": "old result"},
            {"role": "user", "content": "current request"},
        ],
    }

    request = responses.to_context_request(body, "openai", "model-a")
    assert all(group.kind != "opaque" for group in request.groups)
    assert all(not group.pinned for group in request.groups)

    compacted = ContextCompactionResult(
        groups=(request.groups[-1],),
        summary=StructuredSummary(current_task="continue current request"),
        estimated_before_tokens=100,
        estimated_after_tokens=30,
        over_budget=False,
    )
    output = responses.apply_decision(body, compacted)

    assert len(output["input"]) == 2
    assert output["input"][0]["type"] == "message"
    assert output["input"][0]["role"] == "assistant"
    assert "continue current request" in output["input"][0]["content"][0]["text"]
    assert output["input"][1] == body["input"][-1]


def test_responses_previous_response_id_marks_hidden_history_as_opaque():
    body = {
        "previous_response_id": "resp_previous",
        "input": [{"type": "message", "role": "user", "content": "new turn"}],
    }

    request = responses.to_context_request(body, "openai_api", "model-a")
    decision = ContextCompactionResult(
        groups=request.groups,
        summary=None,
        estimated_before_tokens=20,
        estimated_after_tokens=20,
        over_budget=False,
    )
    output = responses.apply_decision(body, decision)

    assert request.opaque_remote_context is True
    assert output["previous_response_id"] == "resp_previous"
    assert output["input"] == body["input"]


def test_connected_openai_codex_full_input_is_compactable_with_previous_response_id():
    body = {
        "previous_response_id": "resp_previous",
        "input": [
            {"type": "message", "role": "user", "content": "old turn"},
            {"type": "message", "role": "assistant", "content": "old answer"},
            {"type": "message", "role": "user", "content": "new turn"},
        ],
    }

    request = responses.to_context_request(body, "openai", "model-a")

    assert request.opaque_remote_context is False
    assert all(not group.pinned for group in request.groups)


def test_responses_string_input_is_kept_opaque_and_unchanged():
    body = {"input": "one string prompt"}

    request = responses.to_context_request(body, "openai", "model-a")
    decision = ContextCompactionResult(
        groups=request.groups,
        summary=None,
        estimated_before_tokens=5,
        estimated_after_tokens=5,
        over_budget=False,
    )
    output = responses.apply_decision(body, decision)

    assert request.groups[0].kind == "opaque"
    assert output["input"] == "one string prompt"


def test_chat_adapter_pins_messages_with_unknown_provider_extensions():
    body = {
        "tools": [],
        "messages": [
            {
                "role": "assistant",
                "content": "provider-owned state",
                "x_provider_signature": "opaque",
            },
            {"role": "user", "content": "current"},
        ],
    }

    request = chat.to_context_request(body, "custom", "model-a")

    assert request.groups[0].kind == "opaque"
    assert request.groups[0].pinned is True


def test_chat_summary_is_inserted_after_stable_prefix_without_mutating_body():
    body = {
        "tools": [{"type": "function", "function": {"name": "read"}}],
        "messages": [
            {"role": "system", "content": "stable"},
            {"role": "developer", "content": "rules"},
            {"role": "user", "content": "old"},
            {"role": "user", "content": "current"},
        ],
    }
    request = chat.to_context_request(body, "openai", "model-a")
    retained = (request.groups[0], request.groups[1], request.groups[-1])
    result = ContextCompactionResult(
        groups=retained,
        summary=StructuredSummary(decisions=("old decision",)),
        estimated_before_tokens=100,
        estimated_after_tokens=30,
        over_budget=False,
    )

    output = chat.apply_decision(body, result)

    assert _wire_bytes(output["messages"][:2]) == _wire_bytes(body["messages"][:2])
    assert output["tools"] is body["tools"]
    assert output["messages"][2]["role"] == "assistant"
    assert "old decision" in output["messages"][2]["content"]
    assert output["messages"][3] == body["messages"][3]
    assert body["messages"] == [
        {"role": "system", "content": "stable"},
        {"role": "developer", "content": "rules"},
        {"role": "user", "content": "old"},
        {"role": "user", "content": "current"},
    ]


def test_existing_fcc_summary_is_pinned_for_idempotent_followup_compaction():
    body = {
        "messages": [
            {"role": "system", "content": "stable"},
            {
                "role": "assistant",
                "content": (
                    "FCC compacted context state (prior completed turns):"
                    '{"decisions":["keep this"]}'
                ),
            },
            {"role": "user", "content": "new turn"},
        ]
    }
    request = chat.to_context_request(body, "openai", "model")
    summary_group = request.groups[1]
    assert summary_group.kind == "summary"
    assert summary_group.pinned
