# FCC Provider Explicit Cache Implementation Plan

> **For agentic workers:** Execute inline in the current authorized checkout. Use TDD for each behavior. Do not commit or stage unrelated dirty files.

**Goal:** Add explicit prompt caching for OpenAI Responses and native explicit context caching for Vertex only.

**Architecture:** OpenAI Responses marks a stable instruction breakpoint in the existing Responses transport and keeps dynamic input after it. Vertex uses a native cached-content lifecycle and generation adapter only when the existing Vertex economic gate passes; unsupported calls retain the current OpenAI-compatible transport.

**Tech Stack:** Python, Pydantic, HTTPX2, SQLite, existing provider transports and usage store.

**Spec:** `docs/superpowers/specs/2026-11-08-provider-explicit-cache-design.md`

## Global Constraints

- Gemini is excluded from explicit Google cache baselines and cache-resource creation.
- OpenAI explicit breakpoints apply only to supported Responses models; Chat Completions remains implicit.
- Vertex resource creation remains behind the existing seven-day, 100-call, known-usage/current-pricing, positive-savings gate.
- Never resend cached Vertex contents inline or persist raw prompts, credentials, or raw session IDs.
- No live cache resource creation in tests.

## Review Focus

- OpenAI instructions are transformed once, preserving the uncached suffix and any caller cache settings.
- Unsupported OpenAI model/protocol paths leave their requests unchanged.
- Vertex cache identity changes across project, model, auth, prefix, or session scopes.
- Vertex create/use/delete failures retain a correct non-cache fallback path.
- Vertex streaming tools, usage, cancellation, and retry behavior remain compatible.

### Task 1: OpenAI Responses explicit breakpoint and usage

**Files:** `providers/openai_responses/prompt_cache.py`, `providers/openai_responses/transport.py`, and their tests.

- [x] Add failing tests for stable-prefix split, unsupported/small-prefix no-op, caller option preservation, and provider transport wire behavior.
- [x] Mark eligible top-level instructions for OpenAI Responses using an explicit breakpoint with implicit lookup retained.
- [x] Record provider `cache_write_tokens` as actual cache-creation input usage.
- [x] Run the final focused provider/runtime tests and validate the complete diff (92 passed; compileall, `node --check`, and `git diff --check` clean).

### Task 2: Vertex native CachedContent lifecycle on its OpenAI-compatible transport

**Files:** Vertex endpoint/client/transport modules, cache registry migration/store, runtime wiring, and Vertex tests.

- [x] Use Vertex's documented OpenAI-compatible `extra_body.google.cached_content` extension without replacing the existing stream transport.
- [x] Persist bounded cache resource metadata keyed by private scoped digests; never persist prefix contents, credentials, or raw session identifiers.
- [x] Create/reuse Vertex `CachedContent` only after the runtime baseline/economics gate passes; omit the cached prefix and inline tool declarations from the generation body.
- [x] Fall back to the original inline request on unsupported layouts, a closed economic gate, or cache lifecycle/storage failures.
- [ ] Verify streamed text/tool calls, provider-reported cache usage, cancellation, and retries using mocked Vertex HTTP responses only; no live Vertex request was made.

### Task 3: Admin/docs and final verification

- [x] Present Vertex baseline and OpenAI cache-read/write usage separately from estimated savings/costs.
- [x] Update configuration and operator docs for Vertex-only Google cache prices and OpenAI Responses cache controls.
- [x] Run the focused suite, JS syntax check, Python compile check, and `git diff --check` (138 focused tests passed before the final transport-hook test was added; rerun pending).
