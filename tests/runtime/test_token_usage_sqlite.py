import sqlite3
from contextlib import closing

import pytest

from free_claude_code.runtime.sqlite_database import initialize_database


def test_database_migrates_token_usage_table(tmp_path):
    path = tmp_path / "fcc.db"

    initialize_database(path)

    connection = sqlite3.connect(path)
    try:
        version = connection.execute("PRAGMA user_version").fetchone()[0]
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_schema WHERE type='table'"
            )
        }
    finally:
        connection.close()

    assert version == 11
    assert "token_usage_attempts" in tables
    assert "vertex_cached_contents" in tables


@pytest.mark.asyncio
async def test_vertex_cache_store_persists_only_scoped_resource_metadata(database_factory, tmp_path):
    from free_claude_code.runtime.vertex_cache_sqlite import SQLiteVertexCacheStore

    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    store = SQLiteVertexCacheStore(database)
    record = {
        "resource_name": "projects/project/locations/us-central1/cachedContents/cache1",
        "model": "google/gemini-3.8-flash",
        "endpoint_key": "b" * 64,
        "prefix_tokens": 2000,
        "created_at": "2026-11-08T12:00:00+00:00",
        "expires_at": "2026-11-08T12:05:00+00:00",
        "ttl_seconds": 300,
    }

    await store.put("a" * 64, record)

    assert await store.get("a" * 64) == record
    assert await store.get("z" * 64) is None
    await store.delete("a" * 64)
    assert await store.get("a" * 64) is None


def _usage_store_types():
    try:
        from free_claude_code.runtime.token_usage_sqlite import (
            SQLiteTokenUsageStore,
            TokenUsageRecord,
        )
    except ModuleNotFoundError as error:
        pytest.fail(f"token usage SQLite store is not implemented: {error.name}")
    return SQLiteTokenUsageStore, TokenUsageRecord


@pytest.mark.asyncio
async def test_store_preserves_unknown_cache_and_aggregates_usage(database_factory, tmp_path):
    store_type, record_type = _usage_store_types()
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    store = store_type(database)
    await store.record(
        record_type(
            occurred_at="2026-11-08T12:00:00+00:00",
            provider="gemini",
            model="gemini-test",
            protocol="chat",
            request_kind="generation",
            estimated_input_before=1000,
            estimated_input_after=600,
            estimated_tokens_saved=400,
            actual_input_tokens=650,
            normal_input_tokens=500,
            cached_input_tokens=150,
            output_tokens=40,
            total_tokens=690,
        )
    )
    await store.record(
        record_type(
            occurred_at="2026-11-08T12:01:00+00:00",
            provider="gemini",
            model="gemini-test",
            protocol="chat",
            request_kind="generation",
            estimated_input_before=1200,
            estimated_input_after=800,
            estimated_tokens_saved=400,
            actual_input_tokens=820,
            normal_input_tokens=None,
            cached_input_tokens=None,
            output_tokens=20,
            total_tokens=840,
        )
    )

    snapshot = await store.query()

    assert snapshot.totals["calls"] == 2
    assert snapshot.totals["estimated_tokens_saved"] == 800
    assert snapshot.totals["actual_input_tokens"] == 1470
    assert snapshot.totals["cached_input_tokens"] == 150
    assert snapshot.totals["cache_usage_calls"] == 1
    assert snapshot.recent[0].cached_input_tokens is None
    assert snapshot.series[0]["calls"] == 2


@pytest.mark.asyncio
async def test_store_filters_provider_and_time_and_retains_newest_10000(
    database_factory, tmp_path
):
    store_type, record_type = _usage_store_types()
    database = database_factory(tmp_path / "fcc.db", tmp_path / "fcc.lock")
    await database.start()
    store = store_type(database)

    def seed(connection):
        connection.executemany(
            "INSERT INTO token_usage_attempts "
            "(occurred_at, provider, model, protocol, request_kind) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                (
                    f"2026-11-08T{12 + index // 3600:02d}:"
                    f"{(index // 60) % 60:02d}:{index % 60:02d}+00:00",
                    "test",
                    "m",
                    "chat",
                    "generation",
                )
                for index in range(10000)
            ),
        )

    await database.run(seed)
    await store.record(
        record_type(
            occurred_at="2026-11-08T15:00:00+00:00",
            provider="newest",
            model="m",
            protocol="chat",
            request_kind="generation",
        )
    )

    filtered = await store.query(
        since="2026-11-08T14:00:00+00:00", provider="newest", limit=5
    )
    retained = await store.query()

    assert filtered.totals["calls"] == 1
    assert retained.totals["calls"] == 10000
    assert retained.oldest_retained_at == "2026-11-08T12:00:01+00:00"
    assert len(retained.recent) == 100


@pytest.mark.asyncio
async def test_prefix_observations_persist_but_never_appear_in_public_snapshot(database_factory, tmp_path):
    store_type, record_type = _usage_store_types()
    database = database_factory(tmp_path / "cache.db", tmp_path / "cache.lock")
    await database.start()
    store = store_type(database)
    await store.record(record_type(
        occurred_at="2026-11-01T12:00:00+00:00", provider="vertex", model="m",
        protocol="chat", request_kind="generation", cache_prefix_key="a" * 64,
        cache_prefix_tokens=1000, cache_endpoint_key="b" * 64,
        cache_observation_version=2, actual_input_tokens=2000, cached_input_tokens=100,
    ))
    snapshot = await store.query(limit=1)
    assert snapshot.cache_observations[0].cache_prefix_key == "a" * 64
    assert snapshot.cache_observations[0].cache_prefix_tokens == 1000
    assert snapshot.cache_observations[0].cache_endpoint_key == "b" * 64
    assert "cache_prefix_key" not in str(snapshot.as_dict())
    assert "cache_endpoint_key" not in str(snapshot.as_dict())
    assert "a" * 64 not in str(snapshot.as_dict())


def test_cache_baseline_migration_preserves_existing_v8_usage(tmp_path):
    from free_claude_code.runtime.sqlite_migrations import MIGRATIONS
    path = tmp_path / "upgrade.db"
    with closing(sqlite3.connect(path)) as connection, connection:
        for version, upgrade in MIGRATIONS:
            if version > 8:
                break
            upgrade(connection)
        connection.execute("PRAGMA user_version = 8")
        connection.execute(
            "INSERT INTO token_usage_attempts (occurred_at, provider, model, protocol, request_kind, actual_input_tokens) "
            "VALUES ('2026-11-01T12:00:00+00:00', 'vertex', 'm', 'chat', 'generation', 123)"
        )
    initialize_database(path)
    with closing(sqlite3.connect(path)) as connection:
        row = connection.execute("SELECT actual_input_tokens, cache_prefix_key, cache_endpoint_key FROM token_usage_attempts").fetchone()
    assert row == (123, None, None)


def test_incomplete_v9_schema_is_rejected_instead_of_losing_observations(tmp_path):
    path = tmp_path / "broken.db"
    initialize_database(path)
    with closing(sqlite3.connect(path)) as connection, connection:
        connection.execute("ALTER TABLE token_usage_attempts DROP COLUMN cache_observation_version")
    with pytest.raises(sqlite3.DatabaseError, match="Incomplete"):
        initialize_database(path)
