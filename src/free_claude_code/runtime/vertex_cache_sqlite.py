"""Private storage for Vertex CachedContent resource references."""

from .sqlite_database import SQLiteDatabase

_MAX_ENTRIES = 512


class SQLiteVertexCacheStore:
    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def get(self, cache_key: str) -> dict[str, object] | None:
        def select(connection):
            row = connection.execute(
                "SELECT resource_name, model, endpoint_key, prefix_tokens, created_at, "
                "expires_at, ttl_seconds FROM vertex_cached_contents WHERE cache_key = ?",
                (cache_key,),
            ).fetchone()
            return dict(row) if row is not None else None

        return await self._database.run(select, write=False)

    async def put(self, cache_key: str, record: dict[str, object]) -> None:
        def upsert(connection):
            connection.execute(
                "INSERT INTO vertex_cached_contents "
                "(cache_key, resource_name, model, endpoint_key, prefix_tokens, created_at, expires_at, ttl_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(cache_key) DO UPDATE SET resource_name=excluded.resource_name, "
                "model=excluded.model, endpoint_key=excluded.endpoint_key, "
                "prefix_tokens=excluded.prefix_tokens, created_at=excluded.created_at, "
                "expires_at=excluded.expires_at, ttl_seconds=excluded.ttl_seconds",
                (
                    cache_key, record["resource_name"], record["model"],
                    record["endpoint_key"], record["prefix_tokens"],
                    record["created_at"], record["expires_at"], record["ttl_seconds"],
                ),
            )
            connection.execute(
                "DELETE FROM vertex_cached_contents WHERE cache_key NOT IN "
                "(SELECT cache_key FROM vertex_cached_contents ORDER BY created_at DESC LIMIT ?)",
                (_MAX_ENTRIES,),
            )

        await self._database.run(upsert)

    async def delete(self, cache_key: str) -> None:
        await self._database.run(
            lambda connection: connection.execute(
                "DELETE FROM vertex_cached_contents WHERE cache_key = ?", (cache_key,)
            ),
        )
