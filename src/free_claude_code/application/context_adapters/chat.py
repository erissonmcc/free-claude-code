"""Context adaptation for OpenAI Chat Completions payloads."""

from collections.abc import Mapping
from typing import Any

from ._shared import apply_decision as _apply_decision
from ._shared import to_context_request as _to_context_request
from free_claude_code.application.context_manager import ContextCompactionResult, ContextRequest


def to_context_request(
    body: Mapping[str, Any],
    provider: str,
    model: str,
    *,
    input_window_tokens: int | None = None,
    scope_key: str = "",
) -> ContextRequest:
    stable_fields = tuple(
        key
        for key in body
        if key not in {"messages", "max_tokens", "max_completion_tokens", "stream"}
    )
    return _to_context_request(
        body,
        provider=provider,
        model=model,
        protocol="chat",
        history_key="messages",
        stable_fields=stable_fields,
        input_window_tokens=input_window_tokens,
        scope_key=scope_key,
    )


def apply_decision(
    body: Mapping[str, Any], decision: ContextCompactionResult
) -> dict[str, Any]:
    return _apply_decision(
        body,
        decision,
        protocol="chat",
        history_key="messages",
        summary_role="assistant",
    )
