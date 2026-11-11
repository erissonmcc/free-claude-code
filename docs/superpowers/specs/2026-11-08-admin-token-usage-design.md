# FCC Admin token usage tab design

## Intent

Show FCC token consumption in its local admin so the operator can see actual input, cached input, output, and estimates per upstream request without inspecting DEBUG logs.

## Current state

- The admin is a static single page. `admin.js` builds the navigation from `VIEW_GROUPS` and switches sections by `data-view`.
- Shared Anthropic Messages, OpenAI Chat, and OpenAI Responses transports emit `provider.response.completed` trace events after usage is received. They already distinguish actual usage from the pre-send input estimate, and keep missing cache counts unknown.
- These values are only sent to logs today. They are not retained for querying by the admin.
- FCC already owns a SQLite database and local admin APIs are restricted to loopback.

## Approaches

1. Parse log files in the admin API. Rejected because logs are not a stable data interface and may omit structured fields or rotate.
2. Keep counters only in process memory. Rejected because restart loses history and it cannot reliably associate in-flight calls with their final usage.
3. Record usage at the provider response boundary in SQLite and expose a local read-only admin API. Selected because it preserves the actual provider-returned accounting and survives restarts.

## Proposed design

- Add a token-usage repository using the existing FCC SQLite database and migration system. Persist one record per completed upstream attempt, with timestamp, provider, model, protocol, estimated input, actual total input, normal input, cached input, cache creation input, output, total usage, and estimate delta when available.
- Do not persist prompts, message bodies, credentials, authorization headers, or session identifiers. Store only usage metadata and provider/model labels.
- Write a record only when the provider stream reaches its terminal usage event. Keep missing usage fields as `null`; never substitute estimates for actual counts.
- Add a loopback-protected read-only endpoint returning aggregate totals and recent records, with optional time range/provider filters. Totals must identify the covered time window and report unavailable cache usage separately from zero.
- Add a `Token usage` admin tab with totals for actual input, normal input, cached input, output, cache hit rate when calculable, and estimated-versus-actual delta; include a recent-call table with provider, model, time, and the same usage breakdown.
- Include charts for completed call count over time and token volume over time, with normal input, cached input, output, and estimated compaction savings visible as distinct series. The primary optimization metric is estimated tokens saved (`estimated_input_tokens_before - estimated_input_tokens_after`) and must be labeled as an estimate, not confirmed billing savings.
- Poll the endpoint every five seconds while the tab is active. A call's actual metrics appear after the provider returns usage at completion; before that, only a request estimate is available. Show this distinction in the UI.
- Bound retained detail to the latest 10,000 completed attempts; show the oldest retained timestamp so aggregates are not mistaken for lifetime totals. No destructive history cleanup or prompt logging is added.

## Scope and constraints

- Covers requests using the shared Anthropic Messages, OpenAI Chat, and OpenAI Responses transports. Native provider CLIs or providers that bypass them do not contribute unless they emit compatible usage through those boundaries.
- No cost conversion is included because pricing varies by provider/model and cached-token rates; this tab reports token counts only.
- Preserve the existing static admin patterns, loopback security check, and SQLite database ownership/migration boundaries.

## Verification

- Unit tests cover usage normalization, unknown cache counts, aggregation and retention boundaries.
- API tests cover admin authorization, filters, and serialized totals.
- UI tests verify navigation, refresh, empty state, and distinction between estimated and actual usage.
- Transport tests verify records are written only on completed responses and include actual returned usage.
