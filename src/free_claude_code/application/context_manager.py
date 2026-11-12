"""Protocol-neutral context budget decisions."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
import hashlib
import inspect
import json
from typing import Protocol

from free_claude_code.core.token_estimation import estimate_text_tokens


@dataclass(frozen=True, slots=True)
class ContextPolicy:
    """Validated limits used while preparing an outbound model request."""

    enabled: bool = True
    max_input_tokens: int | None = None
    response_reserve_tokens: int = 8192
    minimum_recent_messages: int = 6
    max_tool_result_tokens: int = 20000
    summary_target_tokens: int = 2048
    max_compaction_passes: int = 3
    growth_limit_percent: int = 25

    def __post_init__(self) -> None:
        if self.max_input_tokens is not None and self.max_input_tokens <= 0:
            raise ValueError("max_input_tokens must be positive when set")
        for name in (
            "response_reserve_tokens",
            "minimum_recent_messages",
            "max_tool_result_tokens",
            "summary_target_tokens",
            "max_compaction_passes",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if not 0 <= self.growth_limit_percent <= 100:
            raise ValueError("growth_limit_percent must be between 0 and 100")


@dataclass(frozen=True, slots=True)
class ContextRequest:
    """A provider request's visible messages and known model limits."""

    provider: str
    model: str
    protocol: str
    messages: tuple[object, ...]
    input_window_tokens: int | None
    requested_output_tokens: int | None
    groups: tuple[MessageGroup, ...] = ()
    stable_prefix_digest: str = ""
    scope_key: str = ""
    stable_message_count: int = 0
    opaque_remote_context: bool = False

    def __post_init__(self) -> None:
        for name in ("input_window_tokens", "requested_output_tokens"):
            value = getattr(self, name)
            if value is not None and value < 0:
                raise ValueError(f"{name} cannot be negative")


@dataclass(frozen=True, slots=True)
class ContextMeasurement:
    """Estimated and provider-reported token usage, kept distinct."""

    estimated_input_tokens: int
    actual_input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    estimate_incomplete: bool = False


@dataclass(frozen=True, slots=True)
class ContextDecision:
    """The visible request state and its effective input budget."""

    messages: tuple[object, ...]
    input_budget_tokens: int | None
    over_budget: bool
    measurement: ContextMeasurement


InputEstimator = Callable[[Sequence[object]], int]


@dataclass(frozen=True, slots=True)
class MessageGroup:
    """A removable unit that never splits a provider tool transaction."""

    messages: tuple[object, ...]
    kind: str = "ordinary"
    complete: bool = True
    pinned: bool = False


@dataclass(frozen=True, slots=True)
class StructuredSummary:
    """Task state retained when completed older context is compacted."""

    current_task: str = ""
    requirements: tuple[str, ...] = ()
    preferences: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    decisions: tuple[str, ...] = ()
    files: tuple[str, ...] = ()
    changes: tuple[str, ...] = ()
    errors_and_fixes: tuple[str, ...] = ()
    pending_work: tuple[str, ...] = ()
    tool_result_references: tuple[str, ...] = ()

    @classmethod
    def model_validate(cls, value: object) -> StructuredSummary:
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise ValueError("summary must be a structured mapping")
        current_task = value.get("current_task", "")
        if not isinstance(current_task, str):
            raise ValueError("summary current_task must be text")
        fields = (
            "requirements",
            "preferences",
            "constraints",
            "decisions",
            "files",
            "changes",
            "errors_and_fixes",
            "pending_work",
            "tool_result_references",
        )
        normalized: dict[str, tuple[str, ...]] = {}
        for name in fields:
            entries = value.get(name, ())
            if not isinstance(entries, (list, tuple)) or any(
                not isinstance(entry, str) for entry in entries
            ):
                raise ValueError(f"summary {name} must be a list of text")
            normalized[name] = tuple(entries)
        return cls(current_task=current_task, **normalized)


@dataclass(frozen=True, slots=True)
class SummaryCheckpoint:
    source_span_digest: str
    stable_prefix_digest: str
    provider: str
    model: str
    protocol: str
    policy_version: str
    scope_key: str
    summary: StructuredSummary

    @property
    def key(self) -> tuple[str, ...]:
        return (
            self.source_span_digest,
            self.stable_prefix_digest,
            self.provider,
            self.model,
            self.protocol,
            self.policy_version,
            self.scope_key,
        )


class CheckpointStore(Protocol):
    def get(self, key: tuple[str, ...]) -> SummaryCheckpoint | None: ...

    def put(self, checkpoint: SummaryCheckpoint) -> None: ...


class InMemoryCheckpointStore:
    """Bounded process-local checkpoint cache; it never stores source messages."""

    def __init__(self, *, max_entries: int = 128) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be positive")
        self._max_entries = max_entries
        self._entries: OrderedDict[tuple[str, ...], SummaryCheckpoint] = OrderedDict()

    def get(self, key: tuple[str, ...]) -> SummaryCheckpoint | None:
        checkpoint = self._entries.get(key)
        if checkpoint is not None:
            self._entries.move_to_end(key)
        return checkpoint

    def put(self, checkpoint: SummaryCheckpoint) -> None:
        self._entries[checkpoint.key] = checkpoint
        self._entries.move_to_end(checkpoint.key)
        while len(self._entries) > self._max_entries:
            self._entries.popitem(last=False)


class SummaryGenerator(Protocol):
    async def summarize(
        self,
        source_groups: tuple[MessageGroup, ...],
        *,
        target_tokens: int,
    ) -> StructuredSummary: ...


ContextEstimate = Callable[
    [tuple[MessageGroup, ...], StructuredSummary | None], int
]


@dataclass(frozen=True, slots=True)
class ContextCompactionResult:
    groups: tuple[MessageGroup, ...]
    summary: StructuredSummary | None
    estimated_before_tokens: int
    estimated_after_tokens: int
    over_budget: bool
    stop_reason: str | None = None
    internal_summary_calls: int = 0
    duplicate_tool_outputs_removed: int = 0
    tool_results_limited: int = 0


def group_message_transactions(
    protocol: str, messages: Sequence[object]
) -> tuple[MessageGroup, ...]:
    """Keep provider tool calls and their results in a single atomic group."""
    groups: list[MessageGroup] = []
    pending_messages: list[object] = []
    pending_ids: set[str] = set()

    for message in messages:
        calls, results = _tool_ids(protocol, message)
        if pending_messages:
            pending_messages.append(message)
            pending_ids.update(calls)
            pending_ids.difference_update(results)
            if not pending_ids:
                groups.append(
                    MessageGroup(tuple(pending_messages), kind="tool", complete=True)
                )
                pending_messages = []
            continue

        if calls:
            pending_messages = [message]
            pending_ids = set(calls).difference(results)
            if not pending_ids:
                groups.append(
                    MessageGroup((message,), kind="tool", complete=True)
                )
                pending_messages = []
            continue

        if results:
            groups.append(
                MessageGroup((message,), kind="tool", complete=False, pinned=True)
            )
        else:
            groups.append(MessageGroup((message,)))

    if pending_messages:
        groups.append(
            MessageGroup(
                tuple(pending_messages), kind="tool", complete=False, pinned=True
            )
        )
    return tuple(groups)


def deduplicate_tool_outputs(
    protocol: str, messages: Sequence[object]
) -> tuple[tuple[object, ...], int]:
    """Replace exact repeated outputs with references while retaining call IDs."""
    seen: dict[str, int] = {}
    result: list[object] = []
    removed = 0
    for index, message in enumerate(messages):
        if protocol.lower() == "messages" and isinstance(message, Mapping):
            content = message.get("content")
            if isinstance(content, (list, tuple)):
                changed_blocks: list[object] = []
                changed_message = False
                for block in content:
                    if (
                        not isinstance(block, Mapping)
                        or block.get("type") != "tool_result"
                    ):
                        changed_blocks.append(block)
                        continue
                    digest = _digest(block.get("content"))
                    first_index = seen.get(digest)
                    if first_index is None:
                        seen[digest] = index
                        changed_blocks.append(block)
                        continue
                    updated = dict(block)
                    updated["content"] = (
                        f"[duplicate tool result; see message {first_index + 1}]"
                    )
                    changed_blocks.append(updated)
                    changed_message = True
                    removed += 1
                if changed_message:
                    changed = dict(message)
                    changed["content"] = (
                        changed_blocks if isinstance(content, list) else tuple(changed_blocks)
                    )
                    result.append(changed)
                else:
                    result.append(message)
                continue
        output = _tool_output(protocol, message)
        if output is None:
            result.append(message)
            continue
        digest = _digest(output)
        first_index = seen.get(digest)
        if first_index is None:
            seen[digest] = index
            result.append(message)
            continue
        replacement = _replace_tool_output(protocol, message, first_index + 1)
        if replacement is message:
            result.append(message)
            continue
        result.append(replacement)
        removed += 1
    return tuple(result), removed


def limit_tool_result_content(
    protocol: str, groups: tuple[MessageGroup, ...], max_tokens: int
) -> tuple[tuple[MessageGroup, ...], int]:
    """Bound individual plain-text tool results while retaining both ends."""
    changed_count = 0
    output_groups: list[MessageGroup] = []
    # Leave half of the summary input budget for call metadata, JSON framing,
    # and the summary instructions themselves.
    char_budget = max_tokens * 2
    for group in groups:
        changed_messages: list[object] = []
        for message in group.messages:
            changed, count = _limit_message_tool_result(protocol, message, char_budget)
            changed_messages.append(changed)
            changed_count += count
        output_groups.append(
            MessageGroup(
                tuple(changed_messages), group.kind, group.complete, group.pinned
            )
        )
    return tuple(output_groups), changed_count


def _limit_message_tool_result(
    protocol: str, message: object, char_budget: int
) -> tuple[object, int]:
    if not isinstance(message, Mapping):
        return message, 0
    protocol = protocol.lower()
    key: str | None = None
    if protocol == "chat" and message.get("role") == "tool":
        key = "content"
    elif protocol == "responses" and message.get("type") == "function_call_output":
        key = "output"
    if key is not None:
        value = message.get(key)
        if not isinstance(value, str) or estimate_text_tokens(value) <= char_budget // 4:
            return message, 0
        head = char_budget // 2
        tail = char_budget - head
        limited = (
            value[:head]
            + f"\n[tool result limited; original ~{estimate_text_tokens(value)} tokens]\n"
            + value[-tail:]
        )
        changed = dict(message)
        changed[key] = limited
        return changed, 1
    if protocol == "messages":
        content = message.get("content")
        if not isinstance(content, (list, tuple)):
            return message, 0
        changed = False
        limited_blocks: list[object] = []
        count = 0
        for block in content:
            if not isinstance(block, Mapping) or block.get("type") != "tool_result":
                limited_blocks.append(block)
                continue
            value = block.get("content")
            if not isinstance(value, str) or estimate_text_tokens(value) <= char_budget // 4:
                limited_blocks.append(block)
                continue
            head = char_budget * 3 // 4
            tail = char_budget - head
            updated = dict(block)
            updated["content"] = (
                value[:head]
                + f"\n[tool result limited; original ~{estimate_text_tokens(value)} tokens]\n"
                + value[-tail:]
            )
            limited_blocks.append(updated)
            changed = True
            count += 1
        if not changed:
            return message, 0
        updated_message = dict(message)
        updated_message["content"] = (
            limited_blocks if isinstance(content, list) else tuple(limited_blocks)
        )
        return updated_message, count
    return message, 0


def _tool_ids(protocol: str, message: object) -> tuple[set[str], set[str]]:
    if not isinstance(message, Mapping):
        return set(), set()
    protocol = protocol.lower()
    calls: set[str] = set()
    results: set[str] = set()
    if protocol == "chat":
        tool_calls = message.get("tool_calls", ())
        if isinstance(tool_calls, (list, tuple)):
            for call in tool_calls:
                if isinstance(call, Mapping) and isinstance(call.get("id"), str):
                    calls.add(call["id"])
        if message.get("role") == "tool" and isinstance(
            message.get("tool_call_id"), str
        ):
            results.add(message["tool_call_id"])
    elif protocol == "messages":
        content = message.get("content", ())
        if isinstance(content, (list, tuple)):
            for block in content:
                if not isinstance(block, Mapping):
                    continue
                if block.get("type") == "tool_use" and isinstance(
                    block.get("id"), str
                ):
                    calls.add(block["id"])
                if block.get("type") == "tool_result" and isinstance(
                    block.get("tool_use_id"), str
                ):
                    results.add(block["tool_use_id"])
    elif protocol == "responses":
        if message.get("type") == "function_call":
            call_id = message.get("call_id", message.get("id"))
            if isinstance(call_id, str):
                calls.add(call_id)
        elif message.get("type") == "function_call_output" and isinstance(
            message.get("call_id"), str
        ):
            results.add(message["call_id"])
    return calls, results


def _tool_output(protocol: str, message: object) -> object | None:
    if not isinstance(message, Mapping):
        return None
    if protocol.lower() == "chat" and message.get("role") == "tool":
        return message.get("content")
    if protocol.lower() == "responses" and message.get("type") == "function_call_output":
        return message.get("output")
    if protocol.lower() == "messages":
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            return tuple(
                block.get("content")
                for block in content
                if isinstance(block, Mapping) and block.get("type") == "tool_result"
            ) or None
    return None


def _replace_tool_output(protocol: str, message: object, first_index: int) -> object:
    if not isinstance(message, Mapping):
        return message
    reference = f"[duplicate tool result; see message {first_index}]"
    changed = dict(message)
    if protocol.lower() == "chat" and message.get("role") == "tool":
        changed["content"] = reference
        return changed
    if protocol.lower() == "responses" and message.get("type") == "function_call_output":
        changed["output"] = reference
        return changed
    if protocol.lower() == "messages":
        content = message.get("content")
        if isinstance(content, (list, tuple)):
            blocks = []
            for block in content:
                if isinstance(block, Mapping) and block.get("type") == "tool_result":
                    updated = dict(block)
                    updated["content"] = reference
                    blocks.append(updated)
                else:
                    blocks.append(block)
            changed["content"] = blocks if isinstance(content, list) else tuple(blocks)
            return changed
    return message


def _canonical(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        return _canonical(model_dump(mode="json"))
    raise ValueError(f"cannot create a stable context digest for {type(value).__name__}")


def _digest(value: object) -> str:
    encoded = json.dumps(
        _canonical(value), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def context_digest(value: object) -> str:
    """Return a stable digest for JSON-compatible context data."""
    return _digest(value)


def _merge_summaries(*summaries: StructuredSummary) -> StructuredSummary:
    sequence_fields = (
        "requirements",
        "preferences",
        "constraints",
        "decisions",
        "files",
        "changes",
        "errors_and_fixes",
        "pending_work",
        "tool_result_references",
    )
    merged = {
        name: tuple(
            dict.fromkeys(
                entry
                for summary in summaries
                for entry in getattr(summary, name)
            )
        )
        for name in sequence_fields
    }
    current_task = next(
        (summary.current_task for summary in reversed(summaries) if summary.current_task),
        "",
    )
    return StructuredSummary(current_task=current_task, **merged)


async def _summarize_in_bounded_batches(
    groups: tuple[MessageGroup, ...],
    *,
    summary_generator: SummaryGenerator,
    target_tokens: int,
    max_source_chars: int,
    max_levels: int,
) -> tuple[StructuredSummary, int]:
    summaries: list[StructuredSummary] = []
    calls = 0
    # Reserve 25% of the summarizer's source allowance for prompt framing,
    # instructions, and token-estimation variance.
    batch_limit = max(1024, max_source_chars * 3 // 4)
    batches: list[list[MessageGroup]] = []
    current: list[MessageGroup] = []
    current_chars = 2
    for original_group in groups:
        # Keep provider transactions atomic in retained history, but allow the
        # summary view to batch large transactions in bounded fragments.
        for group in _summary_fragments(original_group, batch_limit):
            group_chars = len(json.dumps(group.messages, ensure_ascii=False, default=str))
            if current and current_chars + group_chars > batch_limit:
                batches.append(current)
                current = []
                current_chars = 2
            current.append(group)
            current_chars += group_chars + 1
    if current:
        batches.append(current)

    for batch in batches:
        generated = await summary_generator.summarize(
            tuple(batch), target_tokens=target_tokens
        )
        summaries.append(StructuredSummary.model_validate(generated))
        calls += 1
    if not summaries:
        return StructuredSummary(), calls

    combined = _merge_summaries(*summaries)
    if max_levels == 0:
        return combined, calls
    return await _consolidate_summary(
        combined,
        summary_generator=summary_generator,
        target_tokens=target_tokens,
        max_source_chars=max_source_chars,
        max_levels=max_levels,
        calls=calls,
    )



def _summary_fragments(group: MessageGroup, max_source_chars: int) -> tuple[MessageGroup, ...]:
    """Bound the summary representation without splitting retained history."""
    serialized = json.dumps(group.messages, ensure_ascii=False, default=str)
    if len(serialized) <= max_source_chars:
        return (group,)
    # Reserve room for JSON framing and the synthetic fragment label.
    chunk_size = max(256, max_source_chars - 1024)
    overlap = min(256, chunk_size // 8)
    step = max(1, chunk_size - overlap)
    chunks = [serialized[start : start + chunk_size] for start in range(0, len(serialized), step)]
    total = len(chunks)
    return tuple(
        MessageGroup(
            messages=({
                "role": "assistant",
                "content": (
                    f"[Summary-only fragment {index + 1}/{total} of oversized "
                    f"{group.kind} group; original transaction remains intact. "
                    "Serialized excerpt, not a standalone message.]\n" + chunk
                ),
            },),
            kind="summary_fragment",
            complete=True,
            pinned=False,
        )
        for index, chunk in enumerate(chunks)
    )

async def _consolidate_summary(
    summary: StructuredSummary,
    *,
    summary_generator: SummaryGenerator,
    target_tokens: int,
    max_source_chars: int,
    max_levels: int,
    calls: int = 0,
) -> tuple[StructuredSummary, int]:
    combined = summary
    for _ in range(max_levels):
        serialized = json.dumps(
            asdict(combined), ensure_ascii=False, separators=(",", ":")
        )
        if estimate_text_tokens(serialized) <= target_tokens:
            return combined, calls
        summary_groups = _summary_groups(combined)
        previous_size = len(serialized)
        combined, nested_calls = await _summarize_in_bounded_batches(
            summary_groups,
            summary_generator=summary_generator,
            target_tokens=target_tokens,
            max_source_chars=max_source_chars,
            max_levels=0,
        )
        calls += nested_calls
        next_serialized = json.dumps(
            asdict(combined), ensure_ascii=False, separators=(",", ":")
        )
        if len(next_serialized) >= previous_size:
            # The un-consolidated structured state may still fit the request's
            # actual input budget. Let the caller measure the whole context
            # instead of discarding a useful summary for missing this target.
            return combined, calls
    return combined, calls


def _summary_groups(summary: StructuredSummary) -> tuple[MessageGroup, ...]:
    values: list[object] = []
    if summary.current_task:
        values.append({"role": "assistant", "content": f"current_task: {summary.current_task}"})
    for name in (
        "requirements",
        "preferences",
        "constraints",
        "decisions",
        "files",
        "changes",
        "errors_and_fixes",
        "pending_work",
        "tool_result_references",
    ):
        values.extend(
            {"role": "assistant", "content": f"{name}: {entry}"}
            for entry in getattr(summary, name)
        )
    return tuple(MessageGroup((value,)) for value in values)


class ContextCompactor:
    """Compact completed old groups and reuse exact source-span summaries."""

    def __init__(self, policy: ContextPolicy) -> None:
        self._policy = policy

    async def compact(
        self,
        request: ContextRequest,
        groups: tuple[MessageGroup, ...],
        *,
        estimate_context: ContextEstimate,
        summary_generator: SummaryGenerator,
        checkpoint_store: CheckpointStore,
        stable_prefix_digest: str,
        scope_key: str,
        policy_version: str,
    ) -> ContextCompactionResult:
        budget_decision = ContextManager(self._policy, estimator=lambda _messages: 0).prepare(
            request
        )
        current_groups, limited_count = limit_tool_result_content(
            request.protocol, groups, self._policy.max_tool_result_tokens
        )
        before = await _resolve(estimate_context(current_groups, None))
        if request.opaque_remote_context:
            return ContextCompactionResult(
                current_groups,
                None,
                before,
                before,
                budget_decision.over_budget,
                stop_reason="opaque_remote_context",
                tool_results_limited=limited_count,
            )
        if budget_decision.input_budget_tokens is None:
            return ContextCompactionResult(
                current_groups, None, before, before, False,
                tool_results_limited=limited_count,
            )
        if before <= budget_decision.input_budget_tokens:
            return ContextCompactionResult(
                current_groups, None, before, before, False,
                tool_results_limited=limited_count,
            )

        recent_start = _recent_group_start(
            groups, self._policy.minimum_recent_messages
        )
        candidate_indices = tuple(
            index
            for index, group in enumerate(groups[:recent_start])
            if group.complete
            and not group.pinned
            and group.kind in {"ordinary", "tool", "reasoning_metadata"}
        )
        if not candidate_indices:
            return ContextCompactionResult(
                current_groups, None, before, before, True,
                stop_reason="pinned_context", tool_results_limited=limited_count,
            )

        current_estimate = before
        passes = 0
        deduplicated_count = 0
        summary_candidate_indices = tuple(
            index
            for index in candidate_indices
            if current_groups[index].kind != "reasoning_metadata"
        )
        if not summary_candidate_indices:
            retained = tuple(
                group
                for index, group in enumerate(current_groups)
                if index not in candidate_indices
            )
            after = await _resolve(estimate_context(retained, None))
            return ContextCompactionResult(
                retained if after < before else current_groups,
                None,
                before,
                after if after < before else before,
                after >= before or after > budget_decision.input_budget_tokens,
                stop_reason="no_progress" if after >= before else (
                    "over_budget" if after > budget_decision.input_budget_tokens else None
                ),
                tool_results_limited=limited_count,
            )
        candidate_messages = tuple(
            message
            for index in summary_candidate_indices
            for message in current_groups[index].messages
        )
        deduplicated_messages, deduplicated_count = deduplicate_tool_outputs(
            request.protocol, candidate_messages
        )
        if deduplicated_count:
            current_groups = _replace_group_messages(
                current_groups, summary_candidate_indices, deduplicated_messages
            )
            current_estimate = await _resolve(estimate_context(current_groups, None))
            passes += 1
            if current_estimate >= before:
                return ContextCompactionResult(
                    current_groups,
                    None,
                    before,
                    before,
                    True,
                    stop_reason="no_progress",
                    duplicate_tool_outputs_removed=deduplicated_count,
                    tool_results_limited=limited_count,
                )
            if current_estimate <= budget_decision.input_budget_tokens:
                return ContextCompactionResult(
                    current_groups,
                    None,
                    before,
                    current_estimate,
                    False,
                    duplicate_tool_outputs_removed=deduplicated_count,
                    tool_results_limited=limited_count,
                )
        if passes >= self._policy.max_compaction_passes:
            return ContextCompactionResult(
                current_groups,
                None,
                before,
                current_estimate,
                True,
                stop_reason="max_passes",
                duplicate_tool_outputs_removed=deduplicated_count,
                tool_results_limited=limited_count,
            )

        candidates = tuple(current_groups[index] for index in summary_candidate_indices)
        internal_calls = 0
        summary: StructuredSummary | None = None
        checkpoint: SummaryCheckpoint | None = None
        source_digest = _digest([group.messages for group in candidates])
        full_key = _checkpoint_key(
            source_digest,
            stable_prefix_digest,
            request,
            policy_version,
            scope_key,
        )
        checkpoint = checkpoint_store.get(full_key)
        if checkpoint is not None:
            summary = checkpoint.summary
        else:
            prior_summary: StructuredSummary | None = None
            prior_count = 0
            for count in range(len(candidates) - 1, 0, -1):
                prefix_digest = _digest([group.messages for group in candidates[:count]])
                prefix_key = _checkpoint_key(
                    prefix_digest,
                    stable_prefix_digest,
                    request,
                    policy_version,
                    scope_key,
                )
                cached_prefix = checkpoint_store.get(prefix_key)
                if cached_prefix is not None:
                    prior_summary = cached_prefix.summary
                    prior_count = count
                    break
            unseen = candidates[prior_count:]
            try:
                generated_summary, internal_calls = await _summarize_in_bounded_batches(
                    unseen,
                    summary_generator=summary_generator,
                    target_tokens=self._policy.summary_target_tokens,
                    max_source_chars=max(
                        4096, self._policy.max_tool_result_tokens * 4
                    ),
                    max_levels=self._policy.max_compaction_passes,
                )
                summary = (
                    _merge_summaries(prior_summary, generated_summary)
                    if prior_summary is not None
                    else generated_summary
                )
                summary, consolidation_calls = await _consolidate_summary(
                    summary,
                    summary_generator=summary_generator,
                    target_tokens=self._policy.summary_target_tokens,
                    max_source_chars=max(
                        4096, self._policy.max_tool_result_tokens * 4
                    ),
                    max_levels=self._policy.max_compaction_passes,
                )
                internal_calls += consolidation_calls
            except Exception as error:
                from loguru import logger

                logger.opt(exception=error).warning(
                    "FCC context summary failed; refusing unsafe compaction"
                )
                return ContextCompactionResult(
                    current_groups,
                    None,
                    before,
                    before,
                    True,
                    stop_reason="summary_failed",
                    tool_results_limited=limited_count,
                )
            checkpoint = SummaryCheckpoint(
                source_span_digest=source_digest,
                stable_prefix_digest=stable_prefix_digest,
                provider=request.provider,
                model=request.model,
                protocol=request.protocol,
                policy_version=policy_version,
                scope_key=scope_key,
                summary=summary,
            )
            checkpoint_store.put(checkpoint)

        retained = tuple(
            group
            for index, group in enumerate(current_groups)
            if index not in candidate_indices
        )
        after = await _resolve(estimate_context(retained, summary))
        if after >= current_estimate:
            return ContextCompactionResult(
                current_groups,
                None,
                before,
                current_estimate,
                True,
                stop_reason="no_progress",
                internal_summary_calls=internal_calls,
                duplicate_tool_outputs_removed=deduplicated_count,
                tool_results_limited=limited_count,
            )
        return ContextCompactionResult(
            retained,
            summary,
            before,
            after,
            after > budget_decision.input_budget_tokens,
            stop_reason=(
                "over_budget" if after > budget_decision.input_budget_tokens else None
            ),
            internal_summary_calls=internal_calls,
            duplicate_tool_outputs_removed=deduplicated_count,
            tool_results_limited=limited_count,
        )


def _checkpoint_key(
    source_digest: str,
    stable_prefix_digest: str,
    request: ContextRequest,
    policy_version: str,
    scope_key: str,
) -> tuple[str, ...]:
    return (
        source_digest,
        stable_prefix_digest,
        request.provider,
        request.model,
        request.protocol,
        policy_version,
        scope_key,
    )


def _recent_group_start(
    groups: tuple[MessageGroup, ...], minimum_recent_messages: int
) -> int:
    remaining = minimum_recent_messages
    for index in range(len(groups) - 1, -1, -1):
        remaining -= len(groups[index].messages)
        if remaining <= 0:
            return index
    return 0


def _replace_group_messages(
    groups: tuple[MessageGroup, ...],
    group_indices: tuple[int, ...],
    replacements: tuple[object, ...],
) -> tuple[MessageGroup, ...]:
    replacement_index = 0
    output: list[MessageGroup] = []
    selected = set(group_indices)
    for index, group in enumerate(groups):
        if index not in selected:
            output.append(group)
            continue
        length = len(group.messages)
        output.append(
            MessageGroup(
                messages=replacements[replacement_index : replacement_index + length],
                kind=group.kind,
                complete=group.complete,
                pinned=group.pinned,
            )
        )
        replacement_index += length
    return tuple(output)


async def _resolve(value: object) -> int:
    result = await value if inspect.isawaitable(value) else value
    if isinstance(result, bool) or not isinstance(result, int) or result < 0:
        raise ValueError("context estimator must return a non-negative integer")
    return result


class ContextManager:
    """Measure a request and calculate its effective input budget."""

    def __init__(self, policy: ContextPolicy, *, estimator: InputEstimator) -> None:
        self._policy = policy
        self._estimator = estimator

    def prepare(self, request: ContextRequest) -> ContextDecision:
        estimated_tokens = self._estimator(request.messages)
        if (
            isinstance(estimated_tokens, bool)
            or not isinstance(estimated_tokens, int)
            or estimated_tokens < 0
        ):
            raise ValueError("input estimator must return a non-negative integer")

        budget = self._input_budget(request)
        over_budget = (
            self._policy.enabled
            and budget is not None
            and estimated_tokens > budget
        )
        return ContextDecision(
            messages=request.messages,
            input_budget_tokens=budget,
            over_budget=over_budget,
            measurement=ContextMeasurement(
                estimated_input_tokens=estimated_tokens,
                estimate_incomplete=request.opaque_remote_context,
            ),
        )

    def _input_budget(self, request: ContextRequest) -> int | None:
        if not self._policy.enabled:
            return None

        candidates: list[int] = []
        if self._policy.max_input_tokens is not None:
            candidates.append(self._policy.max_input_tokens)
        if request.input_window_tokens is not None:
            output_reserve = max(
                self._policy.response_reserve_tokens,
                request.requested_output_tokens or 0,
            )
            candidates.append(max(0, request.input_window_tokens - output_reserve))
        return min(candidates) if candidates else None
