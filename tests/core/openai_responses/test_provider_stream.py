import pytest

from free_claude_code.core.anthropic.stream_contracts import (
    assert_anthropic_stream_contract,
    parse_sse_text,
    thinking_content,
)
from free_claude_code.core.openai_responses.provider_stream import (
    ResponsesProviderStream,
    ResponsesStreamFailure,
)
from free_claude_code.core.openai_tool_names import OpenAIToolNameCodec


def _terminal_usage(usage: dict[str, object]) -> dict[str, object]:
    stream = ResponsesProviderStream(
        message_id="msg_test",
        model="openai/gpt-test",
        input_tokens=12,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.completed",
            {"response": {"usage": usage}},
        )
    )
    return next(
        event.data["usage"]
        for event in parse_sse_text("".join(output))
        if event.event == "message_delta"
    )


def _text_blocks(events):
    blocks = {}
    for event in events:
        block = event.data.get("content_block", {})
        delta = event.data.get("delta", {})
        if event.event == "content_block_start" and block.get("type") == "text":
            blocks[event.data["index"]] = ""
        elif event.event == "content_block_delta" and delta.get("type") == "text_delta":
            blocks[event.data["index"]] += delta["text"]
    return list(blocks.values())


def test_responses_provider_stream_preserves_reasoning_tools_usage_and_ids() -> None:
    stream = ResponsesProviderStream(
        message_id="msg_test",
        model="openai/gpt-test",
        input_tokens=12,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.reasoning_summary_text.delta",
            {"item_id": "rs_1", "delta": "reasoning"},
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "reasoning",
                    "id": "rs_1",
                    "encrypted_content": "opaque",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.function_call_arguments.delta",
            {"item_id": "fc_1", "delta": '{"q":'},
        )
    )
    output.extend(
        stream.feed(
            "response.function_call_arguments.delta",
            {"item_id": "fc_1", "delta": '"x"}'},
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_1",
                    "call_id": "call_1",
                    "name": "lookup",
                    "arguments": '{"q":"x"}',
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.completed",
            {
                "response": {
                    "usage": {
                        "input_tokens": 20,
                        "output_tokens": 8,
                        "input_tokens_details": {
                            "cached_tokens": 15,
                            "cache_write_tokens": 3,
                        },
                    }
                }
            },
        )
    )

    events = parse_sse_text("".join(output))
    assert_anthropic_stream_contract(events)
    assert thinking_content(events) == "reasoning"
    starts = [
        event.data["content_block"]
        for event in events
        if event.event == "content_block_start"
    ]
    assert {"type": "redacted_thinking", "data": "opaque"} in starts
    assert {
        "type": "tool_use",
        "id": "call_1",
        "name": "lookup",
        "input": {},
    } in starts
    argument_deltas = [
        event.data["delta"]["partial_json"]
        for event in events
        if event.data.get("delta", {}).get("type") == "input_json_delta"
    ]
    assert argument_deltas == ['{"q":', '"x"}']
    message_delta = next(event for event in events if event.event == "message_delta")
    assert message_delta.data["usage"] == {
        "input_tokens": 2,
        "output_tokens": 8,
        "cache_read_input_tokens": 15,
        "cache_creation_input_tokens": 3,
    }


def test_consecutive_message_items_keep_text_in_separate_content_blocks() -> None:
    stream = ResponsesProviderStream(
        message_id="msg_test", model="openai/gpt-test", input_tokens=0
    )
    output = stream.start()
    for item_id, text in (("msg_first", "Intermediate"), ("msg_final", "Final")):
        output.extend(
            stream.feed(
                "response.output_item.added",
                {
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "content": [],
                    }
                },
            )
        )
        output.extend(
            stream.feed(
                "response.output_text.delta",
                {
                    "item_id": item_id,
                    "output_index": 0,
                    "content_index": 0,
                    "delta": text,
                },
            )
        )
        output.extend(
            stream.feed(
                "response.output_item.done",
                {
                    "item": {
                        "id": item_id,
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text}],
                    }
                },
            )
        )
    output.extend(
        stream.feed("response.completed", {"response": {"usage": {}}})
    )

    events = parse_sse_text("".join(output))
    assert _text_blocks(events) == ["Intermediate", "Final"]


def test_message_confirmation_emits_content_when_no_text_delta_arrived() -> None:
    stream = ResponsesProviderStream(
        message_id="msg_test", model="openai/gpt-test", input_tokens=0
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "id": "msg_final",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [{"type": "output_text", "text": "Confirmed only"}],
                }
            },
        )
    )
    output.extend(
        stream.feed("response.completed", {"response": {"usage": {}}})
    )
    events = parse_sse_text("".join(output))
    text = "".join(_text_blocks(events))
    assert text == "Confirmed only"


def test_message_content_indices_keep_multiple_text_blocks_distinct() -> None:
    stream = ResponsesProviderStream(
        message_id="msg_test", model="openai/gpt-test", input_tokens=0
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.output_text.delta",
            {
                "item_id": "msg_multi",
                "output_index": 0,
                "content_index": 0,
                "delta": "First block",
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_text.delta",
            {
                "item_id": "msg_multi",
                "output_index": 0,
                "content_index": 1,
                "delta": "Second block",
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "id": "msg_multi",
                    "type": "message",
                    "role": "assistant",
                    "status": "completed",
                    "content": [
                        {"type": "output_text", "text": "First block"},
                        {"type": "output_text", "text": "Second block"},
                    ],
                }
            },
        )
    )
    output.extend(
        stream.feed("response.completed", {"response": {"usage": {}}})
    )

    events = parse_sse_text("".join(output))
    assert _text_blocks(events) == ["First block", "Second block"]


@pytest.mark.parametrize(
    ("details", "expected"),
    [
        (
            {"cached_tokens": 10, "cache_write_tokens": 5},
            {
                "input_tokens": 15,
                "output_tokens": 8,
                "cache_read_input_tokens": 10,
                "cache_creation_input_tokens": 5,
            },
        ),
        (
            {"cached_tokens": 10},
            {
                "input_tokens": 20,
                "output_tokens": 8,
                "cache_read_input_tokens": 10,
            },
        ),
        (
            {"cache_write_tokens": 5},
            {
                "input_tokens": 25,
                "output_tokens": 8,
                "cache_creation_input_tokens": 5,
            },
        ),
        (
            {"cached_tokens": 0, "cache_write_tokens": 0},
            {
                "input_tokens": 30,
                "output_tokens": 8,
                "cache_read_input_tokens": 0,
                "cache_creation_input_tokens": 0,
            },
        ),
        (
            {"cached_tokens": 10, "cache_write_tokens": -1},
            {
                "input_tokens": 20,
                "output_tokens": 8,
                "cache_read_input_tokens": 10,
            },
        ),
        (
            {"cached_tokens": 10, "cache_write_tokens": 21},
            {
                "input_tokens": 20,
                "output_tokens": 8,
                "cache_read_input_tokens": 10,
            },
        ),
        (
            {"cached_tokens": -1, "cache_write_tokens": 5},
            {
                "input_tokens": 25,
                "output_tokens": 8,
                "cache_creation_input_tokens": 5,
            },
        ),
        ({}, {"input_tokens": 30, "output_tokens": 8}),
    ],
    ids=[
        "read-and-creation",
        "read-only",
        "creation-only",
        "explicit-zeroes",
        "invalid-creation",
        "creation-over-remaining-total",
        "invalid-read-preserves-creation",
        "no-cache-details",
    ],
)
def test_responses_provider_stream_preserves_trustworthy_cache_partitions(
    details: dict[str, object],
    expected: dict[str, object],
) -> None:
    assert (
        _terminal_usage(
            {
                "input_tokens": 30,
                "output_tokens": 8,
                "input_tokens_details": details,
            }
        )
        == expected
    )


@pytest.mark.parametrize(
    ("input_tokens", "cached_tokens", "expected_input_tokens"),
    [
        (20, -1, 20),
        (20, 21, 20),
        (20, True, 20),
        (None, 5, 12),
    ],
)
def test_responses_provider_stream_ignores_invalid_cache_partitions(
    input_tokens: int | None,
    cached_tokens: int | bool,
    expected_input_tokens: int,
) -> None:
    stream = ResponsesProviderStream(
        message_id="msg_test",
        model="openai/gpt-test",
        input_tokens=12,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.completed",
            {
                "response": {
                    "usage": {
                        "input_tokens": input_tokens,
                        "output_tokens": 8,
                        "input_tokens_details": {"cached_tokens": cached_tokens},
                    }
                }
            },
        )
    )

    message_delta = next(
        event
        for event in parse_sse_text("".join(output))
        if event.event == "message_delta"
    )
    assert message_delta.data["usage"] == {
        "input_tokens": expected_input_tokens,
        "output_tokens": 8,
    }


def test_responses_provider_stream_surfaces_failed_event() -> None:
    stream = ResponsesProviderStream(
        message_id="msg_test",
        model="gpt-test",
        input_tokens=0,
    )

    with pytest.raises(ResponsesStreamFailure, match="capacity") as exc_info:
        stream.feed(
            "response.failed",
            {
                "response": {
                    "error": {
                        "code": "server_error",
                        "message": "No capacity",
                    }
                }
            },
        )

    assert exc_info.value.code == "server_error"


def test_responses_provider_stream_restores_added_and_done_only_tool_names() -> None:
    originals = (
        "mcp__responses_added__" + "x" * 70,
        "mcp__responses_done__" + "y" * 70,
    )
    codec = OpenAIToolNameCodec.from_names(originals)
    stream = ResponsesProviderStream(
        message_id="msg_test",
        model="gpt-test",
        input_tokens=0,
        tool_names=codec,
    )
    output = stream.start()
    output.extend(
        stream.feed(
            "response.output_item.added",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_added",
                    "call_id": "call_added",
                    "name": codec.encode(originals[0]),
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_added",
                    "call_id": "call_added",
                    "name": codec.encode(originals[0]),
                    "arguments": "{}",
                }
            },
        )
    )
    output.extend(
        stream.feed(
            "response.output_item.done",
            {
                "item": {
                    "type": "function_call",
                    "id": "fc_done",
                    "call_id": "call_done",
                    "name": codec.encode(originals[1]),
                    "arguments": "{}",
                }
            },
        )
    )

    event_text = "".join(output)
    starts = [
        event.data["content_block"]
        for event in parse_sse_text(event_text)
        if event.event == "content_block_start"
    ]
    assert [start["name"] for start in starts] == list(originals)
    assert all(codec.encode(name) not in event_text for name in originals)
