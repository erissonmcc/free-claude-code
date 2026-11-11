"""Persist bounded Vertex cache resource references without prompt content."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute(
        """CREATE TABLE vertex_cached_contents (
            cache_key TEXT PRIMARY KEY CHECK(length(cache_key) = 64),
            resource_name TEXT NOT NULL,
            model TEXT NOT NULL,
            endpoint_key TEXT NOT NULL CHECK(length(endpoint_key) = 64),
            prefix_tokens INTEGER NOT NULL CHECK(prefix_tokens > 0),
            created_at TEXT NOT NULL,
            expires_at TEXT NOT NULL,
            ttl_seconds INTEGER NOT NULL CHECK(ttl_seconds > 0)
        )"""
    )
    connection.execute(
        "CREATE INDEX vertex_cached_contents_expiry ON vertex_cached_contents(expires_at)"
    )
