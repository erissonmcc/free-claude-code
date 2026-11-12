"""Shared, non-mutating helpers for protocol-specific context adapters."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict
import json
from typing import Any

from free_claude_code.application.context_manager import (
    ContextCompactionResult,
    ContextRequest,
    MessageGroup,
    StructuredSummary,
    context_digest,
    group_message_transactions,
)


def to_context_request(
    body: Mapping[str, Any],
    *,
    provider: str,
    model: str,
    protocol: str,
    history_key: str,
    stable_fields: tuple[str, ...],
    input_window_tokens: int | None,
    scope_key: str,
) -> ContextRequest:
    history = body.get(history_key)
    if isinstance(history, str) and protocol == "responses":
        history = (history,)
    if not isinstance(history, (list, tuple)):
        raise ValueError(f"{history_key} must be a list or tuple")
    messages = tuple(_jsonable(item) for item in history)
    stable_message_count = _stable_message_count(protocol, messages)
    groups = _pin_groups(
        protocol,
        group_message_transactions(protocol, messages),
        stable_message_count=stable_message_count,
        provider=provider,
    )
    prefix = {name: body[name] for name in stable_fields if name in body}
    return ContextRequest(
        provider=provider,
        model=model,
        protocol=protocol,
        messages=messages,
        input_window_tokens=input_window_tokens,
        requested_output_tokens=_requested_output_tokens(protocol, body),
        groups=groups,
        stable_prefix_digest=context_digest(prefix),
        scope_key=scope_key,
        stable_message_count=stable_message_count,
        # The connected OpenAI/Codex provider deliberately strips
        # previous_response_id before the upstream request and sends the full
        # input history. In that route the history is not hidden remotely and
        # FCC can safely compact it. Other Responses providers may rely on the
        # remote response chain, so keep treating their history as opaque.
        opaque_remote_context=(
            protocol == "responses"
            and provider.lower() != "openai"
            and isinstance(body.get("previous_response_id"), str)
        ),
    )


def apply_decision(
    body: Mapping[str, Any],
    decision: ContextCompactionResult,
    *,
    protocol: str,
    history_key: str,
    summary_role: str,
) -> dict[str, Any]:
    original = body.get(history_key)
    messages = [message for group in decision.groups for message in group.messages]
    if isinstance(original, str) and protocol == "responses":
        if decision.summary is None and len(messages) == 1 and messages[0] == original:
            return dict(body)
        raise ValueError("string Responses input is opaque and cannot be compacted")
    if not isinstance(original, (list, tuple)):
        raise ValueError(f"{history_key} must be a list or tuple")
    if decision.summary is not None:
        count = min(_stable_message_count(protocol, tuple(_jsonable(x) for x in original)), len(messages))
        while count < len(messages) and _is_fcc_summary(protocol, messages[count]):
            count += 1
        messages.insert(count, _summary_message(protocol, summary_role, decision.summary))
    if decision.summary is None and len(messages) == len(original) and all(
        new is old for new, old in zip(messages, original, strict=True)
    ):
        return dict(body)
    changed = dict(body)
    changed[history_key] = tuple(messages) if isinstance(original, tuple) else messages
    return changed


def _stable_message_count(protocol: str, messages: tuple[object, ...]) -> int:
    count = 0
    for message in messages:
        if not isinstance(message, Mapping):
            break
        role = message.get("role")
        if protocol in {"chat", "responses"} and role in {"system", "developer"}:
            count += 1
            continue
        break
    return count


def _pin_groups(
    protocol: str,
    groups: tuple[MessageGroup, ...],
    *,
    stable_message_count: int,
    provider: str,
) -> tuple[MessageGroup, ...]:
    output: list[MessageGroup] = []
    offset = 0
    for group in groups:
        reasoning_metadata = (
            provider.lower() == "openai"
            and protocol == "responses"
            and len(group.messages) == 1
            and isinstance(group.messages[0], Mapping)
            and group.messages[0].get("type") == "reasoning"
            and isinstance(group.messages[0].get("encrypted_content"), str)
        )
        opaque = any(_is_opaque(protocol, message) for message in group.messages)
        summary = any(_is_fcc_summary(protocol, message) for message in group.messages)
        stable = offset < stable_message_count
        output.append(
            MessageGroup(
                messages=group.messages,
                kind=(
                    "reasoning_metadata"
                    if reasoning_metadata
                    else "opaque"
                    if opaque
                    else "summary"
                    if summary
                    else group.kind
                ),
                complete=group.complete,
                pinned=group.pinned or (opaque and not reasoning_metadata) or stable or summary,
            )
        )
        offset += len(group.messages)
    return tuple(output)


def _is_opaque(protocol: str, message: object) -> bool:
    if not isinstance(message, Mapping):
        return True
    if protocol == "chat":
        if message.get("role") not in {"system", "developer", "user", "assistant", "tool"}:
            return True
        if any(
            key in message
            for key in (
                "reasoning_content",
                "reasoning_details",
                "signature",
                "encrypted_content",
            )
        ):
            return True
        allowed = {
            "role",
            "content",
            "name",
            "tool_calls",
            "tool_call_id",
            "refusal",
            "audio",
            "images",
            "annotations",
            "function_call",
        }
        if any(key not in allowed for key in message):
            return True
        return False
    if protocol == "messages":
        if message.get("role") not in {"user", "assistant"}:
            return True
        if any(key not in {"role", "content", "name"} for key in message):
            return True
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            for block in content:
                if not isinstance(block, Mapping):
                    return True
                block_type = block.get("type")
                allowed_block_keys = {
                    "text": {"type", "text", "cache_control"},
                    "tool_use": {"type", "id", "name", "input", "cache_control"},
                    "tool_result": {
                        "type",
                        "tool_use_id",
                        "content",
                        "is_error",
                        "cache_control",
                    },
                }.get(block_type)
                if allowed_block_keys is None or any(
                    key not in allowed_block_keys for key in block
                ):
                    return True
            return False
        return not isinstance(content, str)
    item_type = message.get("type")
    if item_type in (None, "message") and message.get("role") in {
        "system",
        "developer",
        "user",
        "assistant",
    }:
        return any(
            key not in {"type", "role", "content", "id", "status"}
            for key in message
        )
    if item_type in {"function_call", "function_call_output"}:
        allowed = (
            {"type", "id", "call_id", "name", "arguments", "status"}
            if item_type == "function_call"
            else {"type", "id", "call_id", "output", "status"}
        )
        return any(key not in allowed for key in message)
    return True


def _is_fcc_summary(protocol: str, message: object) -> bool:
    if not isinstance(message, Mapping):
        return False
    content = message.get("content")
    if protocol == "responses":
        if not isinstance(content, (list, tuple)):
            return False
        texts = [
            block.get("text", "")
            for block in content
            if isinstance(block, Mapping) and isinstance(block.get("text"), str)
        ]
    else:
        texts = [content] if isinstance(content, str) else []
    marker = "FCC compacted context state (prior completed turns):"
    return any(text.startswith(marker) for text in texts)


def _requested_output_tokens(protocol: str, body: Mapping[str, Any]) -> int | None:
    key = "max_output_tokens" if protocol == "responses" else "max_tokens"
    value = body.get(key)
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _summary_message(
    protocol: str, role: str, summary: StructuredSummary
) -> dict[str, Any]:
    serialized = json.dumps(asdict(summary), ensure_ascii=False, separators=(",", ":"))
    text = f"FCC compacted context state (prior completed turns):\n{serialized}"
    if protocol == "responses":
        return {
            "type": "message",
            "role": role,
            "content": [{"type": "output_text", "text": text}],
        }
    return {"role": role, "content": text}


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    dump = getattr(value, "model_dump", None)
    if callable(dump):
        return _jsonable(dump(mode="json", exclude_none=True))
    return value
