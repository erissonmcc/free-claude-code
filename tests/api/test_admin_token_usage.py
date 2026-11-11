from fastapi.testclient import TestClient
import pytest

from free_claude_code.core.version import package_version
from free_claude_code.runtime.token_usage_sqlite import TokenUsageRecord, TokenUsageSnapshot
from tests.api.support import create_test_app, runtime_for_app


def _local_client(app):
    return TestClient(app, base_url="http://127.0.0.1", client=("127.0.0.1", 50000))


def test_token_usage_is_loopback_only():
    app = create_test_app()

    assert _local_client(app).get("/admin/api/token-usage").status_code == 200
    response = TestClient(app, client=("203.0.113.10", 50000)).get(
        "/admin/api/token-usage"
    )
    assert response.status_code == 403


def test_token_usage_returns_no_store_empty_snapshot():
    response = _local_client(create_test_app()).get(
        "/admin/api/token-usage?since=2026-11-01&provider=anthropic&limit=20"
    )

    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json() == {
        "totals": {},
        "series": [],
        "oldest_retained_at": None,
        "recent": [],
        "cache_baseline": [],
        "usage_costs": [],
    }


def test_token_usage_rejects_invalid_page_limit():
    response = _local_client(create_test_app()).get(
        "/admin/api/token-usage?limit=1001"
    )

    assert response.status_code == 422


def test_token_usage_admin_tab_is_registered_and_documents_estimate():
    client = _local_client(create_test_app())
    page = client.get("/admin").text
    script = client.get(f"/admin/assets/{package_version()}/admin.js").text

    assert 'data-view="usage"' in page
    assert "Estimated context savings" in page
    assert "Normal input" in script and "Cached input" in script
    assert "setInterval(() =>" in script and "}, 5000)" in script
    assert client.get("/admin/usage").status_code == 200


def test_token_usage_serializes_real_values_without_filling_unknown_cache():
    app = create_test_app()

    class Store:
        async def query(self, *, since, provider, limit):
            assert (since, provider, limit) == (None, None, 100)
            record = TokenUsageRecord(
                occurred_at="2026-11-08T12:00:00+00:00",
                provider="openai",
                model="model-a",
                protocol="responses",
                request_kind="generation",
                actual_input_tokens=500,
                normal_input_tokens=None,
                cached_input_tokens=None,
                output_tokens=50,
            )
            return TokenUsageSnapshot(
                totals={"calls": 1, "actual_input_tokens": 500, "cached_input_tokens": None},
                series=({"date": "2026-11-08", "calls": 1, "cached_input_tokens": None},),
                oldest_retained_at=record.occurred_at,
                recent=(record,),
            )

    runtime_for_app(app)._token_usage = Store()
    response = _local_client(app).get("/admin/api/token-usage")

    assert response.status_code == 200
    assert response.json()["totals"]["cached_input_tokens"] is None
    assert response.json()["recent"][0]["actual_input_tokens"] == 500
    assert response.json()["recent"][0]["cached_input_tokens"] is None


@pytest.mark.asyncio
async def test_cache_baseline_survives_runtime_recording_without_disclosing_private_key(database_factory, tmp_path):
    from free_claude_code.runtime.token_usage_sqlite import SQLiteTokenUsageStore

    app = create_test_app()
    runtime = runtime_for_app(app)
    database = database_factory(tmp_path / "admin.db", tmp_path / "admin.lock")
    await database.start()
    runtime._token_usage = SQLiteTokenUsageStore(database)
    await runtime.record_token_usage({
        "provider": "VERTEX", "model": "m", "protocol": "chat",
        "cache_prefix_key": "a" * 64, "cache_prefix_tokens": 2000,
        "cache_observation_version": 1,
        "actual_input_tokens": 3000, "normal_input_tokens": 2000,
        "cached_input_tokens": 1000, "output_tokens": 50,
    })
    response = _local_client(app).get("/admin/api/token-usage")
    assert response.status_code == 200
    assert "a" * 64 not in response.text
    assert "cache_prefix_key" not in response.text
    group = response.json()["cache_baseline"][0]
    assert group["scoped_calls"] == 1
    assert group["normal_input_tokens"] == 2000
    assert group["cache_hit_rate"] == 1
    assert group["estimated_cache_storage_usd"] is None
    assert group["status"] == "pricing_unavailable"
