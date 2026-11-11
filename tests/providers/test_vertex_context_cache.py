import asyncio
import json

import httpx2
import pytest

from free_claude_code.config.google_cache import GoogleCachePrice
from free_claude_code.providers.vertex.context_cache import VertexContextCacheAdapter


def _price():
    return GoogleCachePrice(
        provider="vertex", model="google/gemini-3.8-flash",
        input_per_million=1, cached_input_per_million=.1,
        storage_per_million_hour=.01, minimum_tokens=100,
        ttl_seconds=300, verified_at="2026-11-08", valid_until="2026-11-30",
        source="https://cloud.google.com/vertex-ai/generative-ai/pricing",
    )


def _observation():
    return {
        "cache_prefix_key": "a" * 64,
        "cache_prefix_tokens": 200,
        "cache_endpoint_key": "b" * 64,
        "cache_observation_version": 2,
    }


class _Store:
    def __init__(self):
        self.values = {}

    async def get(self, key):
        return self.values.get(key)

    async def put(self, key, record):
        self.values[key] = record

    async def delete(self, key):
        self.values.pop(key, None)


@pytest.mark.asyncio
async def test_vertex_cache_requires_positive_baseline_and_trusted_scope():
    calls = []
    client = httpx2.AsyncClient(transport=httpx2.MockTransport(
        lambda request: calls.append(request) or httpx2.Response(500)
    ))
    adapter = VertexContextCacheAdapter(
        project_id="project", location="us-central1", access_token_provider=lambda: _token(),
        client=client, store=_Store(), readiness=lambda *_args: _not_ready(),
        prices=(_price(),),
    )
    body = {
        "model": "google/gemini-3.8-flash",
        "messages": [{"role": "system", "content": "instructions"}, {"role": "user", "content": "question"}],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
    }
    try:
        result = await adapter.prepare(body, observation=_observation())
    finally:
        await client.aclose()
    assert result == body
    assert calls == []


@pytest.mark.asyncio
async def test_vertex_cache_reference_removes_cached_fields_and_reuses_resource():
    calls = []

    def handler(request):
        calls.append((request.url.path, json.loads(request.content)))
        return httpx2.Response(200, json={
            "name": "projects/project/locations/us-central1/cachedContents/cache1"
        })

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    store = _Store()
    adapter = VertexContextCacheAdapter(
        project_id="project", location="us-central1", access_token_provider=lambda: _token(),
        client=client, store=store, readiness=lambda *_args: _ready(), prices=(_price(),),
    )
    original = {
        "model": "google/gemini-3.8-flash",
        "messages": [
            {"role": "system", "content": "stable instructions"},
            {"role": "user", "content": "question"},
        ],
        "tools": [{"type": "function", "function": {"name": "lookup", "parameters": {"type": "object"}}}],
        "stream": True,
    }
    try:
        first = await adapter.prepare(original, observation=_observation())
        second = await adapter.prepare(original, observation=_observation())
    finally:
        await client.aclose()

    assert len(calls) == 1
    assert calls[0][1]["ttl"] == "300s"
    assert calls[0][1]["systemInstruction"]["parts"][0]["text"] == "stable instructions"
    assert first["messages"] == [{"role": "user", "content": "question"}]
    assert "tools" not in first
    assert first["extra_body"]["google"]["cached_content"].endswith("cachedContents/cache1")
    assert first == second
    assert original["messages"][0]["role"] == "system"


@pytest.mark.asyncio
async def test_vertex_cache_creation_is_deduplicated_for_parallel_requests():
    calls = []

    def handler(request):
        calls.append(request)
        return httpx2.Response(200, json={
            "name": "projects/project/locations/us-central1/cachedContents/cache1"
        })

    client = httpx2.AsyncClient(transport=httpx2.MockTransport(handler))
    adapter = VertexContextCacheAdapter(
        project_id="project", location="us-central1", access_token_provider=lambda: _token(),
        client=client, store=_Store(), readiness=lambda *_args: _ready(), prices=(_price(),),
    )
    body = {
        "model": "google/gemini-3.8-flash",
        "messages": [{"role": "system", "content": "stable instructions"}, {"role": "user", "content": "question"}],
    }
    try:
        results = await asyncio.gather(*(
            adapter.prepare(body, observation=_observation()) for _ in range(4)
        ))
    finally:
        await client.aclose()

    assert len(calls) == 1
    assert all(
        result["extra_body"]["google"]["cached_content"].endswith("cache1")
        for result in results
    )


async def _token():
    return "test-token"


async def _ready(*_args):
    return True


async def _not_ready(*_args):
    return False
