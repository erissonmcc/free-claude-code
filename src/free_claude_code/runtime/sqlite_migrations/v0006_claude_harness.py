"""Version 6: allow both Codex and Claude Code in the shared session store."""

import sqlite3


def upgrade(connection: sqlite3.Connection) -> None:
    if connection.execute("PRAGMA foreign_keys").fetchone()[0]:
        raise sqlite3.DatabaseError("Migration 6 requires foreign keys disabled.")
    connection.execute(
        """CREATE TABLE code_sessions_new(
        id TEXT PRIMARY KEY NOT NULL, cwd TEXT NOT NULL, model TEXT NOT NULL, reasoning_effort TEXT,
        harness TEXT NOT NULL CHECK(harness IN ('codex','claude')), title TEXT NOT NULL,
        title_search TEXT NOT NULL, cwd_search TEXT NOT NULL,
        auto_title INTEGER NOT NULL CHECK(auto_title IN (0,1)), native_thread_id TEXT,
        native_may_have_input INTEGER NOT NULL CHECK(native_may_have_input IN (0,1)),
        revision INTEGER NOT NULL CHECK(revision > 0),
        status TEXT NOT NULL CHECK(status IN ('ready','deleting','delete_uncertain')),
        error TEXT, created_at INTEGER NOT NULL, updated_at INTEGER NOT NULL,
        mode TEXT NOT NULL DEFAULT 'config' CHECK(mode IN ('config','ask','auto_review','full_access')),
        native_permission_defaults TEXT CHECK(native_permission_defaults IS NULL OR
            (json_valid(native_permission_defaults) AND json_type(native_permission_defaults)='object')),
        context_used_tokens INTEGER CHECK(context_used_tokens IS NULL OR context_used_tokens >= 0)
        )"""
    )
    connection.execute(
        """INSERT INTO code_sessions_new(
        id,cwd,model,reasoning_effort,harness,title,title_search,cwd_search,auto_title,
        native_thread_id,native_may_have_input,revision,status,error,created_at,updated_at,
        mode,native_permission_defaults,context_used_tokens)
        SELECT id,cwd,model,reasoning_effort,harness,title,title_search,cwd_search,auto_title,
        native_thread_id,native_may_have_input,revision,status,error,created_at,updated_at,
        mode,native_permission_defaults,context_used_tokens FROM code_sessions"""
    )
    connection.execute("DROP TABLE code_sessions")
    connection.execute("ALTER TABLE code_sessions_new RENAME TO code_sessions")
    connection.execute(
        "CREATE INDEX code_sessions_recent ON code_sessions(updated_at DESC, id DESC)"
    )
