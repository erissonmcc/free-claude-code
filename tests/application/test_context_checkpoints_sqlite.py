import pytest

from free_claude_code.application.context_checkpoints_sqlite import SQLiteCheckpointStore
from free_claude_code.application.context_manager import (
    ContextCompactor,
    ContextPolicy,
    ContextRequest,
    MessageGroup,
    StructuredSummary,
    SummaryCheckpoint,
)


def test_checkpoint_is_reused_by_a_new_store_instance(tmp_path):
    database = tmp_path / "private" / "context-checkpoints.sqlite3"
    original_source = "ORIGINAL PRIVATE SOURCE SHOULD NEVER BE STORED"
    first_server_store = SQLiteCheckpointStore(database)
    checkpoint = SummaryCheckpoint(
        source_span_digest="source-digest",
        stable_prefix_digest="prefix-digest",
        provider="vertex",
        model="gemini-model",
        protocol="messages",
        policy_version="1",
        scope_key="private-scope-digest",
        summary=StructuredSummary(
            current_task="Finish context management",
            decisions=("Keep original transcript unchanged",),
            files=("src/free_claude_code/application/context_manager.py",),
        ),
    )

    first_server_store.put(checkpoint)
    restarted_server_store = SQLiteCheckpointStore(database)

    assert restarted_server_store.get(checkpoint.key) == checkpoint
    database_bytes = database.read_bytes()
    assert b"Finish context management" in database_bytes
    assert original_source.encode() not in database_bytes
    assert database.stat().st_mode & 0o777 == 0o600


@pytest.mark.asyncio
async def test_compactor_does_not_resummarize_after_store_recreation(tmp_path):
    database = tmp_path / "context-checkpoints.sqlite3"
    groups = (
        MessageGroup(messages=({"role": "user", "content": "old decision"},)),
        MessageGroup(messages=({"role": "user", "content": "current task"},)),
    )
    request = ContextRequest(
        provider="gemini",
        model="gemini-model",
        protocol="chat",
        messages=tuple(message for group in groups for message in group.messages),
        input_window_tokens=100,
        requested_output_tokens=10,
    )

    class Generator:
        calls = 0

        async def summarize(self, _source_groups, *, target_tokens):
            self.calls += 1
            return StructuredSummary(
                current_task="current task",
                decisions=("old decision",),
            )

    generator = Generator()
    compactor = ContextCompactor(
        ContextPolicy(
            max_input_tokens=60,
            response_reserve_tokens=10,
            minimum_recent_messages=1,
        )
    )

    async def estimate(_groups, summary):
        return 100 if summary is None else 30

    first = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=SQLiteCheckpointStore(database),
        stable_prefix_digest="stable-prefix",
        scope_key="session-digest",
        policy_version="1",
    )
    after_restart = await compactor.compact(
        request,
        groups,
        estimate_context=estimate,
        summary_generator=generator,
        checkpoint_store=SQLiteCheckpointStore(database),
        stable_prefix_digest="stable-prefix",
        scope_key="session-digest",
        policy_version="1",
    )

    assert first.summary == after_restart.summary
    assert after_restart.internal_summary_calls == 0
    assert generator.calls == 1


def test_sqlite_checkpoint_store_evicts_old_entries_at_its_limit(tmp_path):
    store = SQLiteCheckpointStore(tmp_path / "bounded.sqlite3", max_entries=2)
    checkpoints = tuple(
        SummaryCheckpoint(
            source_span_digest=f"source-{index}",
            stable_prefix_digest="prefix",
            provider="vertex",
            model="gemini-model",
            protocol="chat",
            policy_version="1",
            scope_key="scope",
            summary=StructuredSummary(current_task=f"task-{index}"),
        )
        for index in range(3)
    )

    for checkpoint in checkpoints:
        store.put(checkpoint)

    assert store.get(checkpoints[0].key) is None
    assert store.get(checkpoints[1].key) == checkpoints[1]
    assert store.get(checkpoints[2].key) == checkpoints[2]
