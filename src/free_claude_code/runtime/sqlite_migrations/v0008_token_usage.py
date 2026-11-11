"""Persist sanitized provider token usage for the Admin dashboard."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE token_usage_attempts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            occurred_at TEXT NOT NULL,
            provider TEXT NOT NULL,
            model TEXT NOT NULL,
            protocol TEXT NOT NULL,
            request_kind TEXT NOT NULL,
            estimated_input_before INTEGER,
            estimated_input_after INTEGER,
            estimated_tokens_saved INTEGER,
            estimated_messages_before INTEGER,
            estimated_messages_after INTEGER,
            internal_summary_calls INTEGER,
            actual_input_tokens INTEGER,
            normal_input_tokens INTEGER,
            cached_input_tokens INTEGER,
            cache_creation_input_tokens INTEGER,
            output_tokens INTEGER,
            total_tokens INTEGER
        )"""
    )
    connection.execute(
        "CREATE INDEX token_usage_occurred_at ON token_usage_attempts(occurred_at DESC)"
    )
    connection.execute(
        "CREATE INDEX token_usage_provider_time ON token_usage_attempts(provider, occurred_at DESC)"
    )
