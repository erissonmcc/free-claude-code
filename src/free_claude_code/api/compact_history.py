"""Restore compacted Claude turns as native Messages API conversation entries."""

import base64
import binascii
import json
import re
from collections.abc import Mapping
from typing import Any

_MARKER = re.compile(r"\[\[FCC_COMPACT_HISTORY_V1:([A-Za-z0-9_=-]+)\]\]")
_MAX_PAYLOAD_BYTES = 1_000_000
_MAX_ENTRIES = 15


def _decode_snapshot(encoded: str) -> list[dict[str, str]]:
    if len(encoded) > (_MAX_PAYLOAD_BYTES * 4 // 3 + 8):
        return []
    try:
        raw = base64.urlsafe_b64decode(encoded + "=" * (-len(encoded) % 4))
        if len(raw) > _MAX_PAYLOAD_BYTES:
            return []
        value = json.loads(raw)
    except (ValueError, binascii.Error, UnicodeError):
        return []
    if not isinstance(value, list) or not value or len(value) > _MAX_ENTRIES:
        return []
    entries: list[dict[str, str]] = []
    for item in value:
        if not isinstance(item, dict):
            return []
        role, content = item.get("role"), item.get("content")
        if role not in {"user", "assistant"} or not isinstance(content, str):
            return []
        if len(content) > 200_000:
            return []
        entries.append({"role": role, "content": content})
    return entries


def _strip_marker(value: Any, snapshots: list[list[dict[str, str]]]) -> Any:
    if isinstance(value, str):
        def replace(match: re.Match[str]) -> str:
            if snapshots:
                return ""
            decoded = _decode_snapshot(match.group(1))
            if not decoded:
                return match.group(0)
            snapshots.append(decoded)
            return ""

        return _MARKER.sub(replace, value)
    if isinstance(value, list):
        return [_strip_marker(item, snapshots) for item in value]
    if isinstance(value, dict):
        return {key: _strip_marker(item, snapshots) for key, item in value.items()}
    return value


def restore_compact_history(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Extract the private hook marker and prepend its turns to the native request.

    The marker can be placed in system context or in a message text block. It is
    removed before normal routing, so neither the marker nor its JSON reaches
    the upstream provider as a text blob.
    """
    copied = dict(raw)
    snapshots: list[list[dict[str, str]]] = []
    copied = _strip_marker(copied, snapshots)
    if not snapshots:
        return copied

    replay = snapshots[0]
    existing = copied.get("messages")
    if not isinstance(existing, list):
        return copied
    remaining = [
        message
        for message in existing
        if not (
            isinstance(message, dict)
            and message.get("role") == "user"
            and message.get("content") == ""
        )
    ]
    copied["messages"] = [*replay, *remaining]
    return copied
