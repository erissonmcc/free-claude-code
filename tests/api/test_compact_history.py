import base64
import json

from free_claude_code.api.compact_history import restore_compact_history


def _marker(entries: list[dict[str, str]]) -> str:
    raw = json.dumps(entries, separators=(",", ":")).encode()
    encoded = base64.urlsafe_b64encode(raw).decode().rstrip("=")
    return f"[[FCC_COMPACT_HISTORY_V1:{encoded}]]"


def test_restore_compact_history_injects_native_turns_from_system_context() -> None:
    history = [
        {"role": "user", "content": "Primeiro pedido"},
        {"role": "assistant", "content": "Primeira resposta"},
    ]
    body = {
        "model": "test-model",
        "system": [{"type": "text", "text": "contexto " + _marker(history)}],
        "messages": [{"role": "user", "content": "Novo pedido"}],
    }

    restored = restore_compact_history(body)

    assert restored["messages"] == [
        *history,
        {"role": "user", "content": "Novo pedido"},
    ]
    assert "FCC_COMPACT_HISTORY_V1" not in json.dumps(restored)
    assert restored["system"] == [{"type": "text", "text": "contexto "}]


def test_restore_compact_history_strips_marker_from_message_content() -> None:
    history = [{"role": "assistant", "content": "Resposta anterior"}]
    body = {
        "model": "test-model",
        "messages": [
            {"role": "user", "content": _marker(history)},
            {"role": "user", "content": "Pedido atual"},
        ],
    }

    restored = restore_compact_history(body)

    assert restored["messages"] == [
        *history,
        {"role": "user", "content": "Pedido atual"},
    ]


def test_restore_compact_history_ignores_invalid_marker() -> None:
    body = {
        "model": "test-model",
        "system": "[[FCC_COMPACT_HISTORY_V1:not-valid-base64!!!]]",
        "messages": [{"role": "user", "content": "Pedido atual"}],
    }

    restored = restore_compact_history(body)

    assert restored == body
