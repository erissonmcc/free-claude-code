"""Persist per-run token usage."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute("ALTER TABLE code_runs ADD COLUMN consumed_tokens INTEGER")
