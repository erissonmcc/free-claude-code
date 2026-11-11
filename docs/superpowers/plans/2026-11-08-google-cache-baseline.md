# Google Cache Baseline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure repeated Google prefixes and actual cache usage, and expose explicit-cache cost projections before creating billable caches.

**Architecture:** Attach privacy-safe prefix observations to each final outbound attempt, retain them with existing usage records, and evaluate observed requests over candidate TTL intervals against configured published prices. Admin shows actual token usage separately from hypothetical costs. Native Google cache creation remains a subsequent phase gated on collected evidence.

**Tech Stack:** Python, existing ContextVar transport hooks, Pydantic settings, SQLite migrations, Admin JavaScript, pytest.

**Spec:** docs/superpowers/specs/2026-11-08-intelligent-context-management-design.md

## Global Constraints

- Preserve serialized prompts and original histories; observation never changes payloads.
- Never persist raw prompts, credentials, headers, or session IDs. Persist only an HMAC prefix key scoped to the authenticated session.
- Missing cache counts stay unknown. Cost projections use USD and configured exact provider/model rates with a source/date; unknown prices prevent a positive readiness verdict.
- Baseline requires seven elapsed days and 100 eligible requests per provider/model. No explicit cache resources are created in this phase.
- Continue in the existing feature checkout because this work depends on its uncommitted context manager and Admin; do not stage or commit unrelated changes.

## Review Focus

- Internal summaries must not inherit the main call's prefix observation.
- Changing system instructions must change the prefix key; changing the newest user message must not.
- Missing session/auth, unreported cache usage, stale pricing, or insufficient sample must prevent readiness.
- High existing implicit reuse can make explicit caching more expensive; retain negative projections.
- Retention/filtering and opaque Responses history must not produce a falsely sufficient baseline.

### Task 1: Capture prefix observations

**Files:** application/context_runtime.py; core/context_hook.py; tests/application/test_context_runtime.py; tests/core/test_context_hook.py.

**Interfaces:** `set_context_cache_observation(prefix_key: str | None, prefix_tokens: int | None)` attaches observation to terminal usage; `RequestContextRuntime` builds it from the final prepared payload with system/tools plus leading system/developer messages.

- [x] Add tests for unchanged payload, identical prefix across tail changes, invalidation on instruction/schema changes, auth/session isolation, and summary clearing.
- [x] Run focused tests and observe missing behavior fail.
- [x] Implement observation after final context preparation, including when compaction is disabled. Clear observation before each attempt and internal summary.
- [x] Run runtime/hook tests; expect all pass.

### Task 2: Persist and evaluate baseline economics

**Files:** runtime/sqlite_migrations/v0009_cache_baseline.py; runtime/token_usage_sqlite.py; application/google_cache_baseline.py; config/settings.py; runtime/application.py; focused runtime/application/config tests.

**Interfaces:** `GoogleCachePrice` validates exact model pricing in USD per million tokens and per million token-hours, minimum cache tokens, price validity date/source, and TTL. `evaluate_cache_baseline(records, prices, now)` returns per-provider/model readiness and a hypothetical TTL-based comparison from observed calls.

- [x] Test migration preserves old records, persistence/private serialization, exact break-even examples, negative savings with implicit hits, unknown usage/pricing, expiry, sample boundaries and internal-call exclusion.
- [x] Observe tests fail, then implement additive migration, sanitized observations, projection and configuration (`FCC_GOOGLE_CACHE_PRICES`).
- [x] Cost simulation groups immutable prefixes by authenticated scope and TTL. Each hypothetical creation pays full input and the full TTL storage allocation; each generation pays the cached input rate. Baseline conservatively credits existing implicit hits against the candidate prefix before calling explicit caching economical.
- [x] Use seven elapsed days and at least 100 model-eligible requests; require all eligible requests to report valid input/cache usage, trustworthy scope and current prices. Keep native creation unavailable in this phase even when projection is favorable.
- [x] Run evaluator/store/settings tests; expect all pass.

### Task 3: Admin baseline display and verification

**Files:** api/admin_static/index.html; api/admin_static/admin.js; tests/api/test_admin_token_usage.py; README.md.

**Interfaces:** Existing `GET /admin/api/token-usage` additionally returns `cache_baseline` groups without private prefix keys; existing calls/charts continue to work.

- [x] Add API tests proving private keys are omitted and real tokens/unknown cost fields survive serialization.
- [x] Add provider/model rows showing sample duration, cache-hit rate, repeated prefix rate, actual tokens, hypothetical write/read/storage cost, estimated net savings and readiness reason.
- [x] Document configuration units, price source/date, retention and absence of native cache creation in `docs/google-cache-baseline.md`.
- [x] Finish final focused review and record results and pending real baseline collection in project memory.

## Subsequent native Google phase

Only after real baseline evidence satisfies the approved gate, plan native GenerateContent/CachedContent adapters, nonduplicating prefix splitting, session-scoped lifecycle, streaming parity, explicit fallback and actual storage lifecycle metering. This plan delivers the required measurement phase without creating billable cache resources.
