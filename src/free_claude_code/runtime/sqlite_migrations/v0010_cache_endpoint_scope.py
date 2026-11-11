"""Add a private endpoint/project scope to cache baseline observations."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    connection.execute(
        "ALTER TABLE token_usage_attempts ADD COLUMN cache_endpoint_key TEXT"
    )
