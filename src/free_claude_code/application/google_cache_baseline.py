"""Compare observed implicit reuse with hypothetical explicit Google caches.

No network calls or cache creation occur here. Prefix token sizes and monetary
amounts are projections; provider input/cache/output counts remain actual usage.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from datetime import datetime, timedelta, timezone
from typing import Any

from free_claude_code.config.google_cache import GoogleCachePrice


def _count(value):
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def _time(value):
    try:
        stamp = datetime.fromisoformat(value)
        return stamp.astimezone(timezone.utc) if stamp.tzinfo is not None else None
    except (TypeError, ValueError):
        return None


def evaluate_cache_baseline(
    records: Iterable[Mapping[str, Any]],
    prices: Sequence[GoogleCachePrice],
    *,
    now: datetime | None = None,
) -> list[dict[str, Any]]:
    now = now or datetime.now(timezone.utc)
    price_map = {(item.provider, item.model): item for item in prices}
    groups = defaultdict(list)
    for record in records:
        provider = str(record.get("provider", "")).lower()
        if provider == "vertex" and record.get("request_kind") == "generation":
            groups[(provider, record["model"])].append(record)
    results = []
    for key, rows in sorted(groups.items()):
        observed = [row for row in rows if row.get("cache_observation_version") in {1, 2}]
        endpoint_groups = defaultdict(list)
        for row in observed:
            endpoint_key = row.get("cache_endpoint_key")
            endpoint_groups[endpoint_key if isinstance(endpoint_key, str) else None].append(row)
        ordered_scopes = sorted(endpoint_groups, key=lambda value: value or "")
        for index, endpoint_key in enumerate(ordered_scopes):
            result = _evaluate(
                endpoint_groups[endpoint_key],
                price_map.get(key),
                now,
                key,
                endpoint_scope_available=endpoint_key is not None,
            )
            result["legacy_calls"] = len(rows) - len(observed) if index == 0 else 0
            if endpoint_key is not None:
                result["endpoint_scope"] = f"scope {index + 1}"
            results.append(result)
    return results


def _evaluate(rows, price, now, key, *, endpoint_scope_available):
    paired = [row for row in rows if _count(row.get("actual_input_tokens"))
              and _count(row.get("cached_input_tokens"))
              and row["cached_input_tokens"] <= row["actual_input_tokens"]]
    scoped = [row for row in rows if isinstance(row.get("cache_prefix_key"), str)
              and len(row["cache_prefix_key"]) == 64
              and _count(row.get("cache_prefix_tokens")) and row["cache_prefix_tokens"] > 0]
    stamps = [_time(row.get("occurred_at")) for row in rows]
    valid_stamps = [stamp for stamp in stamps if stamp is not None and stamp <= now]
    elapsed = (now - min(valid_stamps)).total_seconds() / 86400 if valid_stamps else 0
    result = {
        "provider": key[0], "model": key[1], "calls": len(rows),
        "collected_days": elapsed, "required_days": 7, "required_calls": 100,
        "usage_known_calls": len(paired), "scoped_calls": len(scoped),
        "cache_hit_rate": sum(row["cached_input_tokens"] > 0 for row in paired) / len(paired) if paired else None,
        "prefix_reuse_rate": (len(scoped) - len({row["cache_prefix_key"] for row in scoped})) / len(scoped) if scoped else None,
        "normal_input_tokens": _sum_known(rows, "normal_input_tokens"),
        "cached_input_tokens": _sum_known(rows, "cached_input_tokens"),
        "output_tokens": _sum_known(rows, "output_tokens"),
        "eligible_calls": None, "candidate_cache_creations": None,
        "estimated_baseline_prefix_usd": None, "estimated_cache_creation_usd": None,
        "estimated_cache_reads_usd": None, "estimated_cache_storage_usd": None,
        "estimated_net_savings_usd": None, "currency": "USD",
        "price_source": str(price.source) if price else None,
        "price_verified_at": price.verified_at.isoformat() if price else None,
        "price_valid_until": price.valid_until.isoformat() if price else None,
        "candidate_ttl_seconds": price.ttl_seconds if price else None,
        "baseline_ready": False, "explicit_cache_enabled": False,
        "status": "pricing_unavailable",
    }
    if price is None:
        return result
    if not price.verified_at <= now.date() <= price.valid_until:
        result["status"] = "pricing_expired"
        return result
    if not rows:
        result["status"] = "collecting"
        return result
    if not endpoint_scope_available:
        result["status"] = "endpoint_scope_unavailable"
        return result
    eligible = [row for row in scoped if row["cache_prefix_tokens"] >= price.minimum_tokens
                and (not _count(row.get("actual_input_tokens")) or row["actual_input_tokens"] >= price.minimum_tokens)]
    result["eligible_calls"] = len(eligible)
    eligible_stamps = [_time(row.get("occurred_at")) for row in eligible]
    eligible_stamps = [stamp for stamp in eligible_stamps if stamp is not None and stamp <= now]
    elapsed = (now - min(eligible_stamps)).total_seconds() / 86400 if eligible_stamps else 0
    result["collected_days"] = elapsed
    if len(scoped) != len(rows):
        result["status"] = "scope_unavailable"
        return result
    if not eligible:
        result["status"] = "no_eligible_prefix"
        return result
    if len(paired) != len(rows) or len(valid_stamps) != len(rows):
        result["status"] = "usage_unavailable"
        return result
    creation = reads = storage = baseline = 0.0
    caches = {}
    creations = 0
    for row in sorted(eligible, key=lambda row: _time(row["occurred_at"])):
        stamp = _time(row["occurred_at"])
        tokens = min(row["cache_prefix_tokens"], row["actual_input_tokens"])
        cache_key = (row["cache_prefix_key"], tokens)
        if cache_key not in caches or stamp >= caches[cache_key]:
            caches[cache_key] = stamp + timedelta(seconds=price.ttl_seconds)
            creations += 1
            creation += tokens * price.input_per_million / 1_000_000
            storage += tokens * price.ttl_seconds / 3600 * price.storage_per_million_hour / 1_000_000
        reads += tokens * price.cached_input_per_million / 1_000_000
        # Credit existing implicit hits to the candidate prefix first. This is
        # conservative: do not claim explicit savings from already cached input.
        cached = min(tokens, row["cached_input_tokens"])
        baseline += ((tokens - cached) * price.input_per_million + cached * price.cached_input_per_million) / 1_000_000
    savings = baseline - creation - reads - storage
    result.update({
        "candidate_cache_creations": creations,
        "estimated_baseline_prefix_usd": baseline,
        "estimated_cache_creation_usd": creation,
        "estimated_cache_reads_usd": reads,
        "estimated_cache_storage_usd": storage,
        "estimated_net_savings_usd": savings,
    })
    if elapsed < 7 or len(eligible) < 100:
        result["status"] = "collecting"
    elif savings <= 0:
        result["status"] = "not_economical"
    else:
        result["status"] = "projection_favorable"
        result["baseline_ready"] = True
    return result


def _sum_known(rows, field):
    values = [row[field] for row in rows if _count(row.get(field))]
    return sum(values) if values else None


def estimate_usage_costs(records, prices, *, now=None):
    """Estimate reported Vertex generation usage cost, independent of cache scope."""
    now = now or datetime.now(timezone.utc)
    price_map = {(item.provider, item.model): item for item in prices}
    groups = defaultdict(list)
    for record in records:
        if str(record.get("provider", "")).lower() == "vertex" and record.get("request_kind") == "generation":
            groups[("vertex", str(record.get("model", "")))].append(record)

    results = []
    for (provider, model), rows in sorted(groups.items()):
        price = price_map.get((provider, model))
        complete = [row for row in rows if all(_count(row.get(field)) for field in (
            "normal_input_tokens", "cached_input_tokens", "output_tokens"
        ))]
        result = {
            "provider": provider, "model": model, "calls": len(rows),
            "reported_calls": len(complete), "normal_input_tokens": _sum_known(complete, "normal_input_tokens"),
            "cached_input_tokens": _sum_known(complete, "cached_input_tokens"),
            "output_tokens": _sum_known(complete, "output_tokens"),
            "estimated_cost_usd": None, "estimated_cost_brl": None,
            "usd_to_brl": price.usd_to_brl if price else None,
            "usd_to_brl_verified_at": price.usd_to_brl_verified_at.isoformat() if price and price.usd_to_brl_verified_at else None,
            "fx_source": str(price.fx_source) if price and price.fx_source else None,
            "price_region": price.price_region if price else None,
            "price_source": str(price.source) if price else None,
            "price_verified_at": price.verified_at.isoformat() if price else None,
            "price_valid_until": price.valid_until.isoformat() if price else None,
            "status": "pricing_unavailable",
        }
        if price is None or price.output_per_million is None:
            results.append(result)
            continue
        if not price.verified_at <= now.date() <= price.valid_until:
            result["status"] = "pricing_expired"
            results.append(result)
            continue
        if not complete:
            result["status"] = "usage_unavailable"
            results.append(result)
            continue
        usd = (
            result["normal_input_tokens"] * price.input_per_million
            + result["cached_input_tokens"] * price.cached_input_per_million
            + result["output_tokens"] * price.output_per_million
        ) / 1_000_000
        result["estimated_cost_usd"] = usd
        if price.usd_to_brl is not None:
            result["estimated_cost_brl"] = usd * price.usd_to_brl
            result["estimated_cost_brl_non_global"] = usd * 1.1 * price.usd_to_brl
        result["status"] = "estimated" if len(complete) == len(rows) else "partial_estimate"
        results.append(result)
    return results
