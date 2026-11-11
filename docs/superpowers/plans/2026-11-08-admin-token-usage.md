# FCC Admin token usage Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an Admin tab with token usage history, provider-reported cache metrics, time-series charts, and clearly labeled estimated context-compaction savings.

**Architecture:** Persist one sanitized usage record per completed upstream call in the existing SQLite database. Capture compaction estimates at request preparation and provider-reported usage at the shared transport completion boundary, then expose read-only aggregates to the loopback Admin and render them in the existing static navigation.

**Tech Stack:** Python 3.14, FastAPI, SQLite migrations, existing vanilla JavaScript/CSS Admin.

**Spec:** `docs/superpowers/specs/2026-11-08-admin-token-usage-design.md`

## Global Constraints

- Do not persist prompts, message bodies, credentials, authorization headers, or session identifiers.
- Preserve missing provider usage fields as `null`; never replace actual values with estimates.
- Keep the latest 10,000 completed attempts and identify the covered window in API/UI results.
- Poll every five seconds only while the Token usage tab is active.
- Mark compaction savings as estimated and do not convert token counts to money.
- Maintain existing loopback-only Admin security and SQLite migration ownership.
- Only shared Anthropic Messages, OpenAI Chat, and OpenAI Responses transports contribute in this change.

## Review Focus

- A provider reports input usage but omits cache usage; show cache as unavailable rather than zero.
- A request fails or is cancelled before terminal usage; do not persist it as a completed call.
- A prepared context is compacted more than once because of retries; report savings per completed upstream attempt and avoid counting one attempt twice.
- The database reaches 10,001 rows; retain the newest 10,000 and set the aggregate window from the oldest retained record.
- A concurrent stream is completing while Admin polls; return a consistent aggregate and recent-call page without blocking provider streaming for long.

---

### Task 1: Persist usage and expose Admin aggregates

**Files:**
- Create: `src/free_claude_code/runtime/sqlite_migrations/v0008_token_usage.py`
- Create: `src/free_claude_code/runtime/token_usage_sqlite.py`
- Modify: `src/free_claude_code/runtime/sqlite_migrations/__init__.py`
- Modify: `src/free_claude_code/runtime/sqlite_database.py`
- Modify: `src/free_claude_code/runtime/application.py`
- Modify: `src/free_claude_code/api/ports.py`
- Modify: `src/free_claude_code/api/admin_routes.py`
- Test: `tests/runtime/test_token_usage_sqlite.py`
- Test: `tests/api/test_admin_token_usage.py`

**Interfaces:**
- Produces `TokenUsageRecord` fields for `occurred_at`, `provider`, `model`, `protocol`, `estimated_input_before`, `estimated_input_after`, `estimated_tokens_saved`, `actual_input_tokens`, `normal_input_tokens`, `cached_input_tokens`, `cache_creation_input_tokens`, `output_tokens`, `total_tokens`, and `request_kind`.
- Produces `SQLiteTokenUsageStore.record(record) -> None` and `SQLiteTokenUsageStore.query(*, since, provider, limit) -> TokenUsageSnapshot`.
- `TokenUsageSnapshot` contains aggregate totals, time-bucket series for call count and token types, `oldest_retained_at`, and recent records.
- Produces `AdminRuntimePort.token_usage(*, since, provider, limit) -> JsonObject` and `AdminRuntimePort.record_token_usage(record) -> None`.

- [x] **Step 1: Write failing SQLite tests** for insert/read of nullable usage, provider/time filtering and aggregates, 10,001-record retention, and oldest retained timestamp.
- [x] **Step 2: Run those tests** with `./.venv/bin/pytest -o addopts='' tests/runtime/test_token_usage_sqlite.py -q`; expect failures because the store and migration do not exist.
- [x] **Step 3: Implement migration v0008 and the SQLite store** using the existing `SQLiteDatabase.run` transaction boundary and immutable numbered-migration convention.
- [x] **Step 4: Run the SQLite tests** and require all cases to pass.
- [x] **Step 5: Write failing Admin API tests** for loopback authorization and serialized aggregates/recent records.
- [x] **Step 6: Run the API tests** and confirm the new endpoint is missing or incomplete.
- [x] **Step 7: Add `GET /admin/api/token-usage` and runtime ownership** of the store; return only persisted metric metadata.
- [x] **Step 8: Run both Task 1 test files** and require all cases to pass.

### Task 2: Record context estimates and provider usage per completed attempt

**Files:**
- Modify: `src/free_claude_code/core/context_hook.py`
- Modify: `src/free_claude_code/application/context_runtime.py`
- Modify: `src/free_claude_code/application/execution.py`
- Modify: `src/free_claude_code/api/routes.py`
- Modify: `src/free_claude_code/api/handlers/messages.py`
- Modify: `src/free_claude_code/api/handlers/responses.py`
- Modify: `src/free_claude_code/providers/anthropic_messages/transport.py`
- Modify: `src/free_claude_code/providers/openai_chat/transport.py`
- Modify: `src/free_claude_code/providers/openai_responses/transport.py`
- Test: `tests/application/test_context_runtime.py`
- Test: `tests/application/test_execution.py`
- Test: `tests/providers/test_anthropic_messages_transport.py`
- Test: `tests/providers/test_openai_chat_usage.py`
- Test: `tests/providers/test_openai_responses_transport.py`

**Interfaces:**
- Consumes Task 1 `TokenUsageRecord` and `AdminRuntimePort.record_token_usage`.
- Produces a request-scoped usage recorder installed by `ProviderExecutor`; context preparation attaches before/after/saved estimates, and a completed transport attaches actual provider usage and persists one record.
- Records provider summary requests as `request_kind="context_summary"`; normal upstream attempts use `request_kind="generation"`.

- [x] **Step 1: Write failing unit tests** for estimated savings attached to an attempt, provider actual/cache fields attached at stream completion, unknown cache remaining `None`, and no record on failed/cancelled streams.
- [x] **Step 2: Run the focused tests** and confirm the recorder hook is absent.
- [x] **Step 3: Implement the request-scoped recorder hook** so each prepared attempt carries its context metrics until a provider terminal usage event; do not change user-visible SSE payloads.
- [x] **Step 4: Wire route/runtime recording into `ProviderExecutor`** and call it from the three shared provider transports only after terminal provider usage is available.
- [x] **Step 5: Run the focused tests** and require each successful terminal event to write exactly one record and failures to write none.

### Task 3: Add the Token usage Admin tab and charts

**Files:**
- Modify: `src/free_claude_code/api/admin_static/index.html`
- Modify: `src/free_claude_code/api/admin_static/admin.js`
- Modify: `src/free_claude_code/api/admin_static/admin.css`
- Modify: `src/free_claude_code/api/admin_routes.py`
- Test: `tests/api/test_admin.py`
- Test: `tests/api/test_admin_token_usage.py`

**Interfaces:**
- Consumes `GET /admin/api/token-usage` response from Task 1.
- Produces the `usage` view with refresh-on-activation and five-second polling while active.

- [x] **Step 1: Write failing static/Admin tests** for the navigation entry, section, asset registration, and displayed unknown-cache/estimated-savings labels.
- [x] **Step 2: Run those tests** and confirm the view is absent.
- [x] **Step 3: Add the Admin view and chart rendering** using native SVG or canvas without new dependencies: call-count series; actual normal-input, cached-input, output series; estimated compaction-savings series; recent calls and aggregate totals.
- [x] **Step 4: Refresh only while the usage view is active** and stop polling after navigation away or page teardown.
- [x] **Step 5: Run focused Admin tests plus `git diff --check`** and verify responsive empty, loading, unavailable-cache, and populated states.

## Completion verification

- Run the three focused test groups, compile the changed Python modules, and run `git diff --check`.
- Run the broader available test suite once; report all failures by name, including failures outside this feature.
- Inspect the final diff and verify no prompt text, credentials, or session IDs are written to the usage table.

## Verification result

- Focused token usage, Admin, context hook/runtime, and three provider transport tests: 315 passed.
- `node --check`, Python `compileall`, and `git diff --check` passed.
- Full broad suite was not rerun for this Admin feature. The prior FCC context-manager broad run recorded 7,740 passed, 10 skipped, 35 unrelated failures; see project memory and prior run notes.
- Admin usage rows contain metric fields only. No prompt, credential, authorization value, request body, or session identifier is persisted.

## Execution decision

The tasks share a new record format and a request-scoped callback, so implement them inline in sequence. Do not create a commit because this checkout contains unrelated uncommitted user work.
