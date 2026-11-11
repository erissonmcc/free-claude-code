"""Private durable storage for derived context summaries only."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import sqlite3

from free_claude_code.config.paths import config_dir_path

from .context_manager import (
    CheckpointStore,
    StructuredSummary,
    SummaryCheckpoint,
)


class SQLiteCheckpointStore(CheckpointStore):
    """Persist summary checkpoints without storing any source conversation text."""

    def __init__(self, path: Path | None = None, *, max_entries: int = 512) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._path = path or (config_dir_path() / "context-checkpoints.sqlite3")
        self._max_entries = max_entries

    def get(self, key: tuple[str, ...]) -> SummaryCheckpoint | None:
        digest = _key_digest(key)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT summary_json FROM context_checkpoints WHERE key_digest = ?",
                (digest,),
            ).fetchone()
        if row is None:
            return None
        summary = StructuredSummary.model_validate(json.loads(row[0]))
        return SummaryCheckpoint(
            source_span_digest=key[0],
            stable_prefix_digest=key[1],
            provider=key[2],
            model=key[3],
            protocol=key[4],
            policy_version=key[5],
            scope_key=key[6],
            summary=summary,
        )

    def put(self, checkpoint: SummaryCheckpoint) -> None:
        payload = {
            field: getattr(checkpoint.summary, field)
            for field in checkpoint.summary.__dataclass_fields__
        }
        summary_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO context_checkpoints(key_digest, summary_json) VALUES (?, ?) "
                "ON CONFLICT(key_digest) DO NOTHING",
                (_key_digest(checkpoint.key), summary_json),
            )
            connection.execute(
                "DELETE FROM context_checkpoints WHERE key_digest IN ("
                "SELECT key_digest FROM context_checkpoints ORDER BY rowid DESC "
                "LIMIT -1 OFFSET ?)" ,
                (self._max_entries,),
            )
            connection.commit()

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if self._path.is_symlink():
            raise sqlite3.DatabaseError("Context checkpoint database path is a symlink.")
        if not self._path.exists():
            descriptor = os.open(
                self._path,
                os.O_CREAT | os.O_EXCL | os.O_RDWR,
                0o600,
            )
            os.close(descriptor)
        os.chmod(self._path, 0o600)
        connection = sqlite3.connect(self._path, timeout=10)
        try:
            connection.execute(
                "CREATE TABLE IF NOT EXISTS context_checkpoints ("
                "key_digest TEXT PRIMARY KEY, summary_json TEXT NOT NULL)"
            )
            connection.commit()
            return connection
        except BaseException:
            connection.close()
            raise


def _key_digest(key: Sequence[str]) -> str:
    encoded = json.dumps(list(key), separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()
