# Google cache baseline

FCC currently observes Vertex input and cache usage and shows a hypothetical explicit-cache cost comparison in Admin. Gemini remains a supported provider, but is excluded from this explicit-cache integration. FCC does not create Vertex cache resources until the baseline gate passes and exact model prices are configured and verified.

Set `FCC_GOOGLE_CACHE_PRICES` to a JSON array. Prices are USD per one million normal input tokens, USD per one million cached input tokens, and USD per one million token-hours of stored cache. `source` must be an official Google documentation URL, and `verified_at` / `valid_until` must cover the current date.

```json
[
  {
    "provider": "vertex",
    "model": "EXACT_UPSTREAM_MODEL_ID",
    "input_per_million": 0,
    "cached_input_per_million": 0,
    "storage_per_million_hour": 0,
    "minimum_tokens": 1,
    "ttl_seconds": 300,
    "verified_at": "2026-11-08",
    "valid_until": "2026-11-08",
    "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing"
  }
]
```

The zeroes and model ID above are placeholders, not Google prices; replace every value with the current rate and minimum for that exact Vertex model and API. Use an official `cloud.google.com` or `docs.cloud.google.com` source. Unknown or expired prices leave projections unavailable.

The Vertex baseline requires at least seven elapsed days and 100 eligible generation calls per model and resolved endpoint/project. Calls need provider-reported input and cached-input usage, plus a stable prefix scoped to authentication, endpoint/project, and a session identifier. Calls without a trusted session or resolved endpoint scope still count toward token usage, but not explicit-cache readiness.

Admin keeps actual input, cached input and output usage separate from estimated prefix size and hypothetical USD costs. Endpoint/project scope is represented only by a private HMAC digest; no URL or project identifier is stored. The projection charges the full configured TTL for each candidate cache creation and credits existing implicit cache hits conservatively. Negative or insufficient projections keep `explicit_cache_enabled` false. Usage history retains at most 10,000 calls; prefix and endpoint digests are HMAC-derived and private, and are omitted from the Admin response.
