"""Request-scoped hook for last-moment outbound context preparation."""

from collections.abc import Awaitable, Callable, Mapping
from contextvars import ContextVar, Token
from typing import Any
from loguru import logger

ContextPreparer = Callable[..., Awaitable[dict[str, Any]]]
UsageRecorder = Callable[[Mapping[str, Any]], Awaitable[None]]

_PREPARER: ContextVar[ContextPreparer | None] = ContextVar(
    "fcc_context_preparer", default=None
)
_BYPASS: ContextVar[bool] = ContextVar("fcc_context_bypass", default=False)
_INPUT_ESTIMATE: ContextVar[int | None] = ContextVar("fcc_context_input_estimate", default=None)
_CONTEXT_USAGE: ContextVar[dict[str, Any] | None] = ContextVar("fcc_context_usage", default=None)
_USAGE_RECORDER: ContextVar[UsageRecorder | None] = ContextVar("fcc_usage_recorder", default=None)
_REQUEST_KIND: ContextVar[str] = ContextVar("fcc_usage_request_kind", default="generation")
_CACHE_OBSERVATION: ContextVar[dict[str, Any] | None] = ContextVar(
    "fcc_cache_observation", default=None
)


def install_context_preparer(preparer: ContextPreparer) -> Token:
    return _PREPARER.set(preparer)


def reset_context_preparer(token: Token) -> None:
    _PREPARER.reset(token)


def bypass_context_preparation() -> Token:
    return _BYPASS.set(True)


def reset_context_bypass(token: Token) -> None:
    _BYPASS.reset(token)


async def prepare_outbound_context(
    protocol: str,
    body: Mapping[str, Any],
    provider: str,
    *,
    endpoint_scope: str | None = None,
) -> dict[str, Any]:
    """Prepare one actual attempt; absent a request hook, preserve it verbatim."""
    _CACHE_OBSERVATION.set(None)
    if _BYPASS.get():
        return dict(body)
    preparer = _PREPARER.get()
    if preparer is None:
        return dict(body)
    if endpoint_scope is None:
        return await preparer(protocol, body, provider)
    return await preparer(protocol, body, provider, endpoint_scope=endpoint_scope)


def record_context_input_estimate(tokens: int) -> None:
    _INPUT_ESTIMATE.set(tokens)


def current_context_input_estimate() -> int | None:
    return _INPUT_ESTIMATE.get()


def set_context_usage_estimate(
    before: int,
    after: int,
    saved: int,
    *,
    messages_before: int | None = None,
    messages_after: int | None = None,
    internal_summary_calls: int | None = None,
) -> None:
    _CACHE_OBSERVATION.set(None)
    _INPUT_ESTIMATE.set(after)
    _CONTEXT_USAGE.set({
        "estimated_input_before": before,
        "estimated_input_after": after,
        "estimated_tokens_saved": saved,
        "estimated_messages_before": messages_before,
        "estimated_messages_after": messages_after,
        "internal_summary_calls": internal_summary_calls,
    })


def set_context_cache_observation(
    prefix_key: str | None,
    prefix_tokens: int | None,
    *,
    endpoint_key: str | None = None,
    observed: bool = False,
) -> None:
    _CACHE_OBSERVATION.set(
        {
            "cache_prefix_key": prefix_key,
            "cache_prefix_tokens": prefix_tokens,
            "cache_endpoint_key": endpoint_key,
            "cache_observation_version": 2 if endpoint_key is not None else 1,
        }
        if observed or (prefix_key is not None and prefix_tokens is not None) else None
    )


def current_context_cache_observation() -> dict[str, Any] | None:
    """Return this attempt's private cache observation for provider policy."""
    value = _CACHE_OBSERVATION.get()
    return dict(value) if value is not None else None


def install_usage_recorder(recorder: UsageRecorder) -> Token:
    return _USAGE_RECORDER.set(recorder)


def reset_usage_recorder(token: Token) -> None:
    _USAGE_RECORDER.reset(token)


def set_usage_request_kind(kind: str) -> Token:
    return _REQUEST_KIND.set(kind)


def reset_usage_request_kind(token: Token) -> None:
    _REQUEST_KIND.reset(token)


async def record_provider_usage(**usage: Any) -> None:
    recorder = _USAGE_RECORDER.get()
    if recorder is None:
        return
    recorder_data = {
        "request_kind": _REQUEST_KIND.get(),
        **(_CONTEXT_USAGE.get() or {}),
        **(_CACHE_OBSERVATION.get() or {}),
        **usage,
    }
    try:
        await recorder(recorder_data)
    except Exception as exc:
        logger.warning("Token usage recording failed: exc_type={}", type(exc).__name__)
