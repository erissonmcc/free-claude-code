# Intelligent Context Management Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce redundant FCC input and improve provider prompt-cache reuse while preserving task continuity, wire validity, original history, and real usage accounting.

**Architecture:** Add a protocol-neutral context manager that plans trimming/compaction over complete message transactions and protocol adapters that apply the plan at the last safe request boundary. Keep structured summary checkpoints separate from source history, then integrate actual provider cache usage into per-attempt diagnostics.

**Tech Stack:** Python 3.12+, Pydantic settings, existing Anthropic Messages/OpenAI Chat/OpenAI Responses transports, pytest and existing stream contract tests.

**Spec:** `docs/superpowers/specs/2026-11-08-intelligent-context-management-design.md`

## Global Constraints

- Do not alter the source request, client transcript, or persisted code-session history.
- Preserve chronological order, recent messages, pinned user instructions, tool call/result pairs, and provider-owned signatures.
- Keep stable serialized prefixes unchanged whenever possible; keep variable conversation content at the suffix.
- Use actual provider usage when available; retain “unknown” when cache usage is omitted instead of reporting a false zero.
- Reuse source-digest summary checkpoints; never summarize already summarized spans again.
- Stop compaction after a bounded number of attempts or immediately when a pass does not reduce the estimate.
- Never log raw prompt or tool-result content in context diagnostics.
- Preserve all pre-existing work in the dirty checkout; do not commit unrelated changes.

## Review Focus

- Tool calls with missing, duplicate, or out-of-order results must not produce an invalid provider history; cover each protocol adapter's atomic transaction grouping.
- A summary or provider repair must not mutate stable system/tool prefixes; test exact serialized prefix equality across repeated turns.
- A cache usage field omitted by a provider must remain unknown in diagnostics, while explicit `0` remains a measured zero.
- Repeated over-budget input must terminate on no progress or the configured pass cap without resummarizing an existing checkpoint.
- Opaque Responses items and reasoning/signature extensions must survive unchanged if an adapter cannot safely interpret them.

---

## File Structure

- Create `src/free_claude_code/application/context_manager.py` for budget decisions, safe candidate selection, message grouping, structured summary/checkpoint contracts, progress detection, and request measurements.
- Create protocol adapter modules under `src/free_claude_code/application/context_adapters/` for Messages, Chat Completions, and Responses serialization; keep format-specific transformations out of the manager.
- Modify `src/free_claude_code/config/settings.py` for validated context controls and defaults.
- Modify final-send paths in `src/free_claude_code/providers/anthropic_messages/`, `src/free_claude_code/providers/openai_chat/transport.py`, `src/free_claude_code/providers/openai_responses/transport.py`, and `src/free_claude_code/core/anthropic/passthrough.py` so every actual attempt is budget-checked after transport repairs.
- Modify `src/free_claude_code/providers/openai_chat/behavior.py` and `transport.py` to retain “cache usage omitted” separately from a measured zero; extend trace/usage structures only where needed.
- Add focused tests in `tests/application/test_context_manager.py`, `tests/application/test_context_adapters.py`, provider transport tests, and API/stream tests for execution paths.
- Add a deterministic benchmark script or test fixture under `tests/benchmarks/` that runs identical histories through existing estimators before and after optimization.

## Task 1: Context budget model and settings

**Files:**
- Create: `src/free_claude_code/application/context_manager.py`
- Modify: `src/free_claude_code/config/settings.py`
- Test: `tests/application/test_context_manager.py`
- Test: settings tests under `tests/config/`

**Interfaces:**
- Produce `ContextPolicy`, `ContextRequest`, `ContextMeasurement`, and `ContextDecision` dataclasses.
- Produce `ContextManager.prepare(request: ContextRequest) -> ContextDecision` and a pure protocol-neutral estimator callback interface.
- Add validated `FCC_CONTEXT_*` settings for enablement, maximum input budget, response reserve, minimum recent messages, maximum tool-result contribution, summary target, maximum compaction passes, and per-call growth cap.
- Use a conservative default maximum equal to the known model input window minus requested output and reserve; if the window is unknown, do not silently invent one and only apply an explicit configured cap.

- [x] Write failing tests: `test_short_history_is_returned_unchanged_and_keeps_estimate_separate_from_usage`, `test_budget_reserves_the_larger_of_requested_output_and_configured_reserve`, `test_unknown_model_window_uses_only_explicit_input_budget`, and `test_context_management_settings_have_safe_defaults_and_process_aliases`; invalid values are covered by `test_context_management_settings_reject_invalid_limits`.
- [x] Run the focused tests and confirm missing context types/settings fail before implementation.
- [x] Implement only policy parsing and a pure measurement/decision model; no provider calls or persistence in this task.
- [x] Verify short histories are unchanged and estimated counts are labeled separately from provider usage.

## Task 2: Safe grouping, deduplication, summaries, and checkpoint reuse

**Files:**
- Modify: `src/free_claude_code/application/context_manager.py`
- Create if justified by size: `src/free_claude_code/application/context_checkpoints.py`
- Test: `tests/application/test_context_manager.py`

**Interfaces:**
- Produce `MessageGroup` boundaries for ordinary messages and complete tool call/result transactions.
- Produce `SummaryCheckpoint` with source-span digest, stable-prefix digest, policy/provider/model keys, structured state, and source references.
- Add `CheckpointStore` protocol and a private sidecar SQLite implementation with bounded retention for summaries, without persisting source conversation text.
- Produce an injectable `SummaryGenerator` protocol returning validated structured summary fields.

- [x] Test exact tool-output deduplication and transaction grouping with `test_chat_tool_call_and_results_form_one_complete_group`, `test_incomplete_chat_tool_transaction_is_pinned_as_a_unit`, `test_message_and_responses_protocols_group_call_result_pairs`, `test_deduplicating_tool_outputs_keeps_both_call_result_pairs_valid`, `test_anthropic_deduplication_keeps_unrelated_parallel_result_blocks`, and `test_compactor_deduplicates_old_tool_results_before_summarizing`.
- [x] Test structured preservation with `test_structured_summary_preserves_required_task_state_categories`.
- [x] Test checkpoint reuse with `test_compactor_reuses_the_same_checkpoint_without_resummarizing`.
- [x] Verify idempotency across store recreation with `test_compactor_does_not_resummarize_after_store_recreation`.
- [x] Test no-progress, invalid summary schema, and pass cap with `test_compactor_stops_when_summary_makes_no_progress`, `test_invalid_summary_schema_keeps_all_source_groups`, and `test_compaction_pass_cap_prevents_summary_after_a_tool_dedup_pass`.
- [x] Implemented duplicate-result handling before summarizing completed old groups; pinned groups and the configured recent tail remain intact.
- [x] Verify original input objects remain unmodified and retained groups remain chronological.

## Task 3: Protocol adapters and prefix stability

**Files:**
- Create: `src/free_claude_code/application/context_adapters/messages.py`
- Create: `src/free_claude_code/application/context_adapters/chat.py`
- Create: `src/free_claude_code/application/context_adapters/responses.py`
- Create: `src/free_claude_code/application/context_adapters/_shared.py` for serialization, digest, pinning, and summary insertion shared by adapters
- Create: `src/free_claude_code/application/context_adapters/__init__.py`
- Test: `tests/application/test_context_adapters.py`

**Interfaces:**
- Each adapter implements `to_context_request(body, provider, model) -> ContextRequest` and `apply_decision(body, decision) -> body`.
- Each adapter must identify stable prefix elements, chronological content, atomic tool groups, pinned elements, and opaque items without changing original provider extension fields.

- [x] Test exact prefix stability with `test_chat_stable_prefix_is_identical_when_only_the_last_turn_changes` and `test_chat_tool_schema_change_invalidates_stable_prefix_digest`.
- [x] Test unknown provider extensions, encrypted reasoning, hidden `previous_response_id`, and string input with `test_chat_adapter_pins_messages_with_unknown_provider_extensions`, `test_responses_adapter_pins_unknown_and_encrypted_reasoning_items`, `test_responses_previous_response_id_marks_hidden_history_as_opaque`, and `test_responses_string_input_is_kept_opaque_and_unchanged`.
- [x] Test atomic chronological histories and summary insertion with `test_protocol_adapters_keep_tool_transactions_atomic_and_ordered`, `test_chat_summary_is_inserted_after_stable_prefix_without_mutating_body`, and `test_non_chat_adapters_insert_summary_after_stable_instructions`.
- [x] Implemented adapters without mutating the original body; unchanged stable fields remain the same objects and compacted summaries follow invariant instructions.
- [x] Run adapter tests and compare stable prefix digests across repeated-call fixtures.

## Task 4: Final-send integration and every internal attempt

**Files:**
- Modify: `src/free_claude_code/providers/anthropic_messages/transport.py` final request send
- Modify: `src/free_claude_code/providers/openai_chat/transport.py`
- Modify: `src/free_claude_code/providers/openai_responses/transport.py`
- Modify: `src/free_claude_code/core/anthropic/passthrough.py`
- Modify: `src/free_claude_code/application/execution.py` and `src/free_claude_code/application/web_tools/service.py` only for wiring policy/summary executor
- Test: corresponding provider transport, API, recovery, and web-tool tests

**Interfaces:**
- Inject a `ContextManager` and policy into each final-send transport boundary.
- Prepare context after provider retry corrections, history repair, and continuation changes, immediately before each network request.
- Ensure summary-generation requests use a bounded internal executor and cannot recursively trigger summaries.

- [x] Add transport tests proving the three final protocol boundaries are checked; corrected Responses retries are checked after repair. Continuations, native Messages, and web-tool requests share the same executor context scope and final transport hooks.
- [x] Test that a corrected Responses retry is re-estimated after its body changes.
- [x] Integrate adapters immediately before each shared transport's final network call, preserving fields outside history.
- [x] Return explicit budget failures before upstream I/O when pinned content cannot fit.
- [x] Focused tests pass; opaque `previous_response_id` histories retain the payload and emit `estimate_incomplete=true`.

## Task 5: Actual usage, cache accounting, and diagnostics

**Files:**
- Modify: `src/free_claude_code/providers/openai_chat/behavior.py`
- Modify: `src/free_claude_code/providers/openai_chat/transport.py`
- Modify: Anthropic and Responses usage parsers only where missing fields currently collapse to zero
- Modify: `src/free_claude_code/core/trace.py` or existing diagnostics structures
- Test: provider usage/parser tests and diagnostics tests

**Interfaces:**
- Record per outbound attempt: estimated before/after/savings, message and tool-result counts, stable-prefix digest/change reason, internal summary count, provider-reported input/cached/output/total, and estimator deltas.
- Cache counts use `int | None`: `None` means provider omitted measurement; `0` means measured zero.
- Expose a cache-hit ratio only when actual input and cached counts are available and valid.

- [x] Existing Chat usage tests cover actual cache counts, explicit zero, omitted cache fields, and malformed values.
- [x] Provider traces read final completed usage snapshots; Anthropic usage keys merge by latest value and Chat retains the final usage chunk, avoiding cumulative summation.
- [x] Anthropic traces parse nested `message_start.message.usage`; actual input uses uncached plus cache input when available, with uncached input reported separately.
- [x] Add structured diagnostics without prompt/tool text, separating provider cache usage from local summary checkpoints.
- [x] Diagnostics derive normal input only when actual and cached counts are both valid; missing values stay null.
- [x] Focused usage/stream tests pass for all three protocols.

## Task 6: End-to-end regression suite and savings benchmark

**Files:**
- Add/update tests across `tests/application/`, `tests/providers/`, `tests/api/`, and `tests/contracts/`
- Create: `tests/benchmarks/context_manager_benchmark.py` or deterministic benchmark fixture
- Update: design/plan docs with verified limitations and results

- [x] Cover the ten required scenarios across manager, adapter, runtime, and transport tests.
- [x] Add a deterministic benchmark using the shared estimator and identical input payload.
- [x] Verify adapters do not mutate original request objects; context management does not write to code-session persistence.
- [x] Focused suite (322 tests), compile checks, and diff whitespace checks pass. Additional API regression suites passed (318 and 145 tests respectively).
- [ ] Broad suite remains incomplete: PyYAML was installed locally to remove three collection errors; four modules still cannot collect on Android because of the cryptography ABI and missing Playwright. A broader run excluding those modules was interrupted at 32% (2,663 passed, 16 pre-existing Android launcher fixture failures due missing `am`).
- [x] Report estimated fixture savings separately from actual provider usage; direct native-CLI providers outside the three shared HTTP transports are not covered by these hooks.

## Execution Notes

- The checkout contains extensive unrelated dirty/untracked work. Inspect `git diff` before edits and stage only files from this plan if committing is later requested; do not create commits during this task because that would risk including the user's pre-existing changes.
- Cache stability tests compare serialized prefixes, not just semantic JSON equality.
- Summary and checkpoint code must not log prompt contents or persist original context elsewhere.
- Durable summary checkpoints are stored in `~/.fcc/context-checkpoints.sqlite3` with mode `0600` and a 512-entry retention cap; the database stores key digests and structured summary JSON, never source groups. Growth-limit state is scoped to each ProviderExecutor and auth/session identity, capped at 512 entries.
- The executor installs and resets its context preparer around each awaited stream advance, avoiding cross-task ContextVar token resets in ASGI streaming.
- Benchmark fixture result: estimated input 9,167 -> 223 tokens (8,944 / 97.6% reduction), 30 -> 7 messages. The canned summary makes this a deterministic estimator comparison, not a real provider quality or billing result.
