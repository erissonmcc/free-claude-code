"""Retain private cache-prefix observations alongside terminal usage."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE token_usage_attempts ADD COLUMN cache_prefix_key TEXT")
    connection.execute("ALTER TABLE token_usage_attempts ADD COLUMN cache_prefix_tokens INTEGER")
    connection.execute("ALTER TABLE token_usage_attempts ADD COLUMN cache_observation_version INTEGER")
