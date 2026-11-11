from datetime import datetime, timezone

import pytest

from free_claude_code.application.google_cache_baseline import (
    estimate_usage_costs,
    evaluate_cache_baseline,
)
from free_claude_code.config.google_cache import GoogleCachePrice


NOW = datetime(2026, 11, 8, 12, tzinfo=timezone.utc)


def price(**overrides):
    return GoogleCachePrice(**{
        "provider": "vertex", "model": "m", "input_per_million": 10,
        "cached_input_per_million": 1, "storage_per_million_hour": 2,
        "minimum_tokens": 1000, "ttl_seconds": 300,
        "verified_at": "2026-11-01", "valid_until": "2026-11-30",
        "source": "https://cloud.google.com/vertex-ai/generative-ai/pricing",
        **overrides,
    })


def record(at="2026-11-01T12:00:00+00:00", **overrides):
    return {
        "occurred_at": at, "provider": "vertex", "model": "m",
        "protocol": "chat", "request_kind": "generation",
        "cache_prefix_key": "a" * 64, "cache_prefix_tokens": 1000,
        "cache_endpoint_key": "e" * 64,
        "cache_observation_version": 2,
        "actual_input_tokens": 2000, "cached_input_tokens": 0,
        "normal_input_tokens": 2000, "output_tokens": 100,
        **overrides,
    }


def test_projection_accounts_for_creation_read_and_full_ttl_storage():
    records = [record(), record("2026-11-01T12:01:00+00:00")]
    result = evaluate_cache_baseline(records, [price()], now=NOW)[0]
    # 2 normal prefixes cost .02; explicit: .01 write + .002 reads + .0001667 storage.
    assert result["estimated_baseline_prefix_usd"] == pytest.approx(.02)
    assert result["estimated_cache_creation_usd"] == pytest.approx(.01)
    assert result["estimated_cache_reads_usd"] == pytest.approx(.002)
    assert result["estimated_cache_storage_usd"] == pytest.approx(1 / 6000)
    assert result["estimated_net_savings_usd"] == pytest.approx(.007833333333)
    assert result["candidate_cache_creations"] == 1
    assert result["status"] == "collecting"
    assert result["explicit_cache_enabled"] is False


def test_expiry_and_session_changes_pay_new_creation_and_storage():
    rows = [record(), record("2026-11-01T12:05:00+00:00"), record(cache_prefix_key="b" * 64)]
    result = evaluate_cache_baseline(rows, [price()], now=NOW)[0]
    assert result["candidate_cache_creations"] == 3
    assert result["estimated_net_savings_usd"] < 0


def test_existing_implicit_hits_can_make_explicit_cache_more_expensive():
    rows = [record(cached_input_tokens=1000, normal_input_tokens=1000) for _ in range(100)]
    result = evaluate_cache_baseline(rows, [price()], now=NOW)[0]
    assert result["cache_hit_rate"] == 1
    assert result["status"] == "not_economical"
    assert result["estimated_net_savings_usd"] < 0


def test_positive_projection_needs_seven_days_and_100_calls_but_never_creates_resources():
    rows = [record() for _ in range(100)]
    result = evaluate_cache_baseline(rows, [price()], now=NOW)[0]
    assert result["status"] == "projection_favorable"
    assert result["baseline_ready"] is True
    assert result["explicit_cache_enabled"] is False
    assert result["prefix_reuse_rate"] == .99
    assert result["cached_input_tokens"] == 0
    assert result["normal_input_tokens"] == 200000
    assert result["output_tokens"] == 10000
    assert "cache_prefix_key" not in str(result)
    recent = evaluate_cache_baseline(rows, [price()], now=datetime(2026, 11, 2, tzinfo=timezone.utc))[0]
    assert recent["baseline_ready"] is False


@pytest.mark.parametrize("change,status", [
    ({"cached_input_tokens": None}, "usage_unavailable"),
    ({"cached_input_tokens": 2001}, "usage_unavailable"),
    ({"cache_prefix_key": None}, "scope_unavailable"),
    ({"cache_endpoint_key": None}, "endpoint_scope_unavailable"),
    ({"cache_prefix_tokens": 999}, "no_eligible_prefix"),
])
def test_incomplete_baselines_cannot_be_recommended(change, status):
    result = evaluate_cache_baseline([record(**change) for _ in range(100)], [price()], now=NOW)[0]
    assert result["baseline_ready"] is False
    assert result["status"] == status


def test_missing_or_expired_prices_and_internal_calls():
    rows = [record() for _ in range(100)]
    assert evaluate_cache_baseline(rows, [], now=NOW)[0]["status"] == "pricing_unavailable"
    assert evaluate_cache_baseline(rows, [price(valid_until="2026-11-02")], now=NOW)[0]["status"] == "pricing_expired"
    assert evaluate_cache_baseline([record(request_kind="context_summary"), record(provider="openai")], [price()], now=NOW) == []


def test_google_explicit_cache_baseline_ignores_gemini_provider():
    assert evaluate_cache_baseline(
        [record(provider="gemini") for _ in range(100)], [price()], now=NOW
    ) == []


def test_legacy_records_do_not_poison_or_backdate_new_baseline():
    rows = [record() for _ in range(100)] + [record(
        "2025-01-01T00:00:00+00:00", cache_observation_version=None,
        cache_prefix_key=None, cache_prefix_tokens=None,
    )]
    result = evaluate_cache_baseline(rows, [price()], now=NOW)[0]
    assert result["baseline_ready"] is True
    assert result["collected_days"] == 7
    assert result["calls"] == 100
    assert result["legacy_calls"] == 1


def test_ineligible_old_prefix_cannot_backdate_eligible_baseline():
    rows = [record("2026-11-08T12:00:00+00:00") for _ in range(100)]
    rows.append(record(cache_prefix_tokens=999))
    result = evaluate_cache_baseline(rows, [price()], now=NOW)[0]
    assert result["status"] == "collecting"
    assert result["collected_days"] == 0


def test_estimated_prefix_cannot_exceed_actual_input_to_meet_model_minimum():
    rows = [record(actual_input_tokens=500, normal_input_tokens=500) for _ in range(100)]
    result = evaluate_cache_baseline(rows, [price()], now=NOW)[0]
    assert result["status"] == "no_eligible_prefix"
    assert result["baseline_ready"] is False


def test_endpoint_scopes_do_not_pool_calls_to_reach_the_baseline_gate():
    rows = [record(cache_endpoint_key="a" * 64) for _ in range(50)]
    rows.extend(record(cache_endpoint_key="b" * 64) for _ in range(50))

    results = evaluate_cache_baseline(rows, [price()], now=NOW)

    assert len(results) == 2
    assert {item["calls"] for item in results} == {50}
    assert all(item["status"] == "collecting" for item in results)
    assert {item["endpoint_scope"] for item in results} == {"scope 1", "scope 2"}


def test_google_price_configuration_decodes_json_and_rejects_invalid_rates():
    import json
    from free_claude_code.config.settings import Settings
    value = price().model_dump(mode="json")
    settings = Settings(FCC_GOOGLE_CACHE_PRICES=json.dumps([value]))
    assert settings.google_cache_prices[0].model == "m"
    with pytest.raises(ValueError):
        Settings(FCC_GOOGLE_CACHE_PRICES=json.dumps([{**value, "provider": "gemini"}]))
    with pytest.raises(ValueError):
        price(storage_per_million_hour=float("nan"))
    with pytest.raises(ValueError):
        Settings(FCC_GOOGLE_CACHE_PRICES=json.dumps([value, value]))


def test_usage_cost_estimate_does_not_require_cache_endpoint_scope():
    rows = [
        record(cache_endpoint_key=None, cache_prefix_key=None, cache_observation_version=None),
        record(cache_endpoint_key=None, cache_prefix_key=None, cache_observation_version=None,
               normal_input_tokens=1000, cached_input_tokens=1000, output_tokens=200),
    ]
    configured = price(
        input_per_million=0.30, cached_input_per_million=0.03,
        output_per_million=2.50, usd_to_brl=5.50,
        usd_to_brl_verified_at="2026-11-08",
        fx_source="https://www.oeb.org.br/indicadores/dolar",
    )
    result = estimate_usage_costs(rows, [configured], now=NOW)[0]
    expected_usd = (3000 * 0.30 + 1000 * 0.03 + 300 * 2.50) / 1_000_000
    assert result["reported_calls"] == 2
    assert result["estimated_cost_usd"] == pytest.approx(expected_usd)
    assert result["estimated_cost_brl"] == pytest.approx(expected_usd * 5.50)
    assert result["estimated_cost_brl_non_global"] == pytest.approx(expected_usd * 5.50 * 1.1)
    assert result["status"] == "estimated"


def test_usage_cost_estimate_marks_partial_provider_usage():
    rows = [record(), record(normal_input_tokens=None)]
    configured = price(output_per_million=2.50, usd_to_brl=5.50)
    result = estimate_usage_costs(rows, [configured], now=NOW)[0]
    assert result["reported_calls"] == 1
    assert result["calls"] == 2
    assert result["status"] == "partial_estimate"
