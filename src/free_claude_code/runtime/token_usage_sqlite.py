"""Persist and aggregate provider-reported token usage for Admin."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .sqlite_database import SQLiteDatabase

_RETAINED_ATTEMPTS = 10_000
_TOKEN_FIELDS = (
    "estimated_input_before",
    "estimated_input_after",
    "estimated_tokens_saved",
    "estimated_messages_before",
    "estimated_messages_after",
    "internal_summary_calls",
    "actual_input_tokens",
    "normal_input_tokens",
    "cached_input_tokens",
    "cache_creation_input_tokens",
    "output_tokens",
    "total_tokens",
    "cache_prefix_tokens",
)


@dataclass(frozen=True, slots=True)
class TokenUsageRecord:
    occurred_at: str
    provider: str
    model: str
    protocol: str
    request_kind: str
    estimated_input_before: int | None = None
    estimated_input_after: int | None = None
    estimated_tokens_saved: int | None = None
    estimated_messages_before: int | None = None
    estimated_messages_after: int | None = None
    internal_summary_calls: int | None = None
    actual_input_tokens: int | None = None
    normal_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    cache_creation_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cache_prefix_tokens: int | None = None
    cache_prefix_key: str | None = None
    cache_observation_version: int | None = None
    cache_endpoint_key: str | None = None

    def __post_init__(self) -> None:
        if self.cache_observation_version is not None and (
            type(self.cache_observation_version) is not int
            or self.cache_observation_version not in {1, 2}
        ):
            raise ValueError("unsupported cache observation version")
        if self.cache_prefix_key is not None and not re.fullmatch(r"[0-9a-f]{64}", self.cache_prefix_key):
            raise ValueError("cache prefix key must be a SHA-256 HMAC")
        if self.cache_endpoint_key is not None and not re.fullmatch(r"[0-9a-f]{64}", self.cache_endpoint_key):
            raise ValueError("cache endpoint key must be a SHA-256 HMAC")
        for name in ("occurred_at", "provider", "model", "protocol", "request_kind"):
            if not getattr(self, name):
                raise ValueError(f"{name} must not be empty")
        for name in _TOKEN_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int) or value < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer or None")


@dataclass(frozen=True, slots=True)
class TokenUsageSnapshot:
    totals: dict[str, int | None]
    series: tuple[dict[str, int | str | None], ...]
    oldest_retained_at: str | None
    recent: tuple[TokenUsageRecord, ...]
    cache_observations: tuple[TokenUsageRecord, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "totals": self.totals,
            "series": list(self.series),
            "oldest_retained_at": self.oldest_retained_at,
            "recent": [
                {
                    key: value
                    for key, value in asdict(record).items()
                    if key not in {
                        "cache_prefix_key",
                        "cache_endpoint_key",
                        "cache_observation_version",
                    }
                }
                for record in self.recent
            ],
        }


class SQLiteTokenUsageStore:
    """Store sanitized completed usage records in FCC's owned database."""

    def __init__(self, database: SQLiteDatabase) -> None:
        self._database = database

    async def record(self, record: TokenUsageRecord) -> None:
        values = asdict(record)

        def insert(connection):
            connection.execute(
                "INSERT INTO token_usage_attempts "
                "(occurred_at, provider, model, protocol, request_kind, "
                "estimated_input_before, estimated_input_after, estimated_tokens_saved, "
                "estimated_messages_before, estimated_messages_after, internal_summary_calls, "
                "actual_input_tokens, normal_input_tokens, cached_input_tokens, "
                "cache_creation_input_tokens, output_tokens, total_tokens, cache_prefix_key, cache_prefix_tokens, cache_observation_version, cache_endpoint_key) "
                "VALUES (:occurred_at, :provider, :model, :protocol, :request_kind, "
                ":estimated_input_before, :estimated_input_after, :estimated_tokens_saved, "
                ":estimated_messages_before, :estimated_messages_after, :internal_summary_calls, "
                ":actual_input_tokens, :normal_input_tokens, :cached_input_tokens, "
                ":cache_creation_input_tokens, :output_tokens, :total_tokens, :cache_prefix_key, :cache_prefix_tokens, :cache_observation_version, :cache_endpoint_key)",
                values,
            )
            connection.execute(
                "DELETE FROM token_usage_attempts WHERE id NOT IN "
                "(SELECT id FROM token_usage_attempts ORDER BY id DESC LIMIT ?)",
                (_RETAINED_ATTEMPTS,),
            )

        await self._database.run(insert)

    async def query(
        self,
        *,
        since: str | None = None,
        provider: str | None = None,
        limit: int = 100,
    ) -> TokenUsageSnapshot:
        if not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")

        def select(connection):
            oldest = connection.execute(
                "SELECT MIN(occurred_at) FROM token_usage_attempts"
            ).fetchone()[0]
            clauses: list[str] = []
            parameters: list[str] = []
            if since is not None:
                clauses.append("occurred_at >= ?")
                parameters.append(since)
            if provider is not None:
                clauses.append("provider = ?")
                parameters.append(provider)
            where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
            rows = connection.execute(
                f"SELECT * FROM token_usage_attempts {where} ORDER BY id DESC LIMIT ?",
                (*parameters, _RETAINED_ATTEMPTS),
            ).fetchall()
            return oldest, rows

        oldest, rows = await self._database.run(select, write=False)
        records = tuple(_record_from_row(row) for row in rows)
        totals = _aggregate(records)
        buckets: dict[str, list[TokenUsageRecord]] = {}
        for record in records:
            buckets.setdefault(record.occurred_at[:10], []).append(record)
        series = tuple(
            _aggregate(bucket, label=day)
            for day, bucket in sorted(buckets.items())
        )
        return TokenUsageSnapshot(
            totals=totals,
            series=series,
            oldest_retained_at=oldest,
            recent=records[:limit],
            cache_observations=records,
        )


def _record_from_row(row) -> TokenUsageRecord:
    return TokenUsageRecord(
        occurred_at=row["occurred_at"],
        provider=row["provider"],
        model=row["model"],
        protocol=row["protocol"],
        request_kind=row["request_kind"],
        cache_prefix_key=row["cache_prefix_key"],
        cache_observation_version=row["cache_observation_version"],
        cache_endpoint_key=row["cache_endpoint_key"],
        **{field: row[field] for field in _TOKEN_FIELDS},
    )


def _aggregate(
    records: tuple[TokenUsageRecord, ...] | list[TokenUsageRecord],
    *,
    label: str | None = None,
) -> dict[str, int | str | None]:
    result: dict[str, int | str | None] = {
        "calls": len(records),
        **{
            field: _sum_known(records, field)
            for field in _TOKEN_FIELDS
        },
        "cache_usage_calls": sum(
            record.cached_input_tokens is not None for record in records
        ),
        "actual_usage_calls": sum(
            record.actual_input_tokens is not None for record in records
        ),
    }
    if label is not None:
        result["date"] = label
    return result


def _sum_known(records: list[TokenUsageRecord] | tuple[TokenUsageRecord, ...], field: str):
    values = [getattr(record, field) for record in records]
    known = [value for value in values if value is not None]
    return sum(known) if known else None
