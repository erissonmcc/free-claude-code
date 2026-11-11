"""Build explicit OpenAI Responses cache breakpoints for stable prefixes."""

from collections.abc import Mapping
from copy import deepcopy
import re
from typing import Any

from free_claude_code.core.token_estimation import estimate_text_tokens

_MODEL_VERSION = re.compile(r"^gpt-(\d+)(?:\.(\d+))?(?:[-.]|$)", re.IGNORECASE)
_MINIMUM_CACHEABLE_TOKENS = 1024


def add_explicit_prompt_cache_breakpoint(body: dict[str, Any]) -> dict[str, Any]:
    """Mark a sufficiently large top-level instruction prefix for Responses.

    Eligible models use explicit breakpoints. The stable instructions become a
    developer content block; the changing input remains after that breakpoint.
    Existing caller cache settings and explicitly selected mode are respected.
    """
    result = deepcopy(body)
    model = result.get("model")
    instructions = result.get("instructions")
    input_items = result.get("input")
    options = result.get("prompt_cache_options")
    if (
        not isinstance(model, str)
        or not _supports_explicit_breakpoints(model)
        or not isinstance(instructions, str)
        or not instructions.strip()
        or not isinstance(input_items, (list, str))
        or not isinstance(options, (dict, type(None)))
        or (isinstance(options, dict) and options.get("mode") == "explicit")
        or _MINIMUM_CACHEABLE_TOKENS > estimate_text_tokens(instructions)
    ):
        return result

    if _contains_breakpoint(input_items):
        return result

    if isinstance(input_items, str):
        input_items = [{"role": "user", "content": input_items}]

    options = dict(options or {})
    options.setdefault("mode", "implicit")
    options.setdefault("ttl", "30m")
    ttl = options.get("ttl")
    block: dict[str, Any] = {
        "type": "input_text",
        "text": instructions,
        "prompt_cache_breakpoint": {"mode": "explicit"},
    }
    if isinstance(ttl, str):
        block["prompt_cache_breakpoint"]["ttl"] = ttl

    result["input"] = [{"role": "developer", "content": [block]}, *input_items]
    result["prompt_cache_options"] = options
    del result["instructions"]
    return result


def _supports_explicit_breakpoints(model: str) -> bool:
    if model.strip().lower() in {"gpt-5.6-luna", "gpt-6-luna"}:
        return False
    match = _MODEL_VERSION.match(model.strip())
    if match is None:
        return False
    version = (int(match.group(1)), int(match.group(2) or 0))
    return version >= (5, 6)


def _contains_breakpoint(value: Any) -> bool:
    if isinstance(value, Mapping):
        if "prompt_cache_breakpoint" in value:
            return True
        return any(_contains_breakpoint(item) for item in value.values())
    if isinstance(value, list):
        return any(_contains_breakpoint(item) for item in value)
    return False
