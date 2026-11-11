# FCC explicit prompt caching: OpenAI Responses and Vertex

## Goal

Use provider-native explicit caching for OpenAI Responses and Vertex while preserving normal behavior for unsupported protocols and models. Gemini is intentionally excluded from this explicit-cache phase. Stable prefix ordering remains enabled across the shared context manager.

## Approaches

1. Keep implicit caching only. This requires little protocol work but cannot select the cache write boundary.
2. Use provider-native mechanisms. OpenAI Responses uses explicit prompt-cache breakpoints, while Vertex uses native `CachedContent` references and `generateContent`; each retains its existing fallback when ineligible. This is the selected approach.

## OpenAI Responses

- On supported GPT-5.6+ models, mark a stable, sufficiently large top-level instruction prefix with an explicit breakpoint and keep the dynamic input after it.
- Use the Responses API's implicit mode alongside the explicit breakpoint so automatic matching remains available. Default to the documented 30-minute lifetime; preserve caller-provided supported cache options.
- Do not mark changing user content. Do not duplicate instructions when moving them into the developer content block required for a breakpoint.
- Do not inject these fields into OpenAI Chat Completions or non-OpenAI-compatible providers. Unsupported models and too-small prefixes keep the current request unchanged.
- Record provider-reported cached input and cache-write token counts separately from estimates.

## Vertex

- Use only Vertex for Google explicit cache resources. Gemini remains a provider but is not included in Vertex cache baseline or native cache resource creation.
- Create/reuse native Vertex cached content only when the existing baseline has at least seven elapsed days, 100 eligible calls for the same model/endpoint scope, known actual usage and current official prices, and a positive net savings projection.
- Scope resources by project/endpoint, model, authorization identity, stable-prefix digest, and trusted session. Never persist raw prompts or credentials. The generation request must reference the cached resource and omit cached fields inline.
- Unsupported requests, absent session scope, insufficient/unknown economics, expired resources, or cache errors use the existing Vertex OpenAI-compatible route without dropping conversation context.

## Verification

Unit and transport tests prove the OpenAI breakpoint split, model/size eligibility, settings preservation, provider isolation, and real cache-write usage reporting. Vertex tests prove native resource reuse/expiry, cache reference without inline duplication, baseline gating, safe fallback, and stream/tool/usage compatibility. No test may create a live billable Google cache resource.
