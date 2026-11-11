"""Optional Vertex CachedContent reuse on the OpenAI-compatible Chat API."""

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import re
from typing import Any
from urllib.parse import quote

import httpx2
from loguru import logger

from free_claude_code.config.google_cache import GoogleCachePrice
from free_claude_code.providers.vertex.endpoint import vertex_service_endpoint

CacheReadiness = Callable[[str, str, str, int], Awaitable[bool]]


class VertexContextCacheAdapter:
    """Create/reuse Vertex cache resources only after the runtime cost gate."""

    def __init__(
        self,
        *,
        project_id: str,
        location: str,
        access_token_provider: Callable[[], Awaitable[str]],
        client: httpx2.AsyncClient,
        store: Any,
        readiness: CacheReadiness,
        prices: tuple[GoogleCachePrice, ...],
    ) -> None:
        self._project_id = project_id
        self._location = location
        self._access_token_provider = access_token_provider
        self._client = client
        self._store = store
        self._readiness = readiness
        self._prices = {price.model: price for price in prices if price.provider == "vertex"}
        self._creation_lock = asyncio.Lock()

    async def prepare(
        self, body: Mapping[str, Any], *, observation: Mapping[str, Any] | None
    ) -> dict[str, Any]:
        original = deepcopy(dict(body))
        model = original.get("model")
        if not isinstance(model, str):
            return original
        price = self._prices.get(model)
        prefix_key = observation.get("cache_prefix_key") if observation else None
        prefix_tokens = observation.get("cache_prefix_tokens") if observation else None
        endpoint_key = observation.get("cache_endpoint_key") if observation else None
        if (
            price is None
            or not isinstance(prefix_key, str)
            or not isinstance(endpoint_key, str)
            or observation.get("cache_observation_version") != 2
            or not _is_digest(prefix_key)
            or not _is_digest(endpoint_key)
            or not isinstance(prefix_tokens, int)
            or isinstance(prefix_tokens, bool)
            or prefix_tokens < price.minimum_tokens
            or _caller_cache_is_present(original)
        ):
            return original
        try:
            ready = await self._readiness(model, endpoint_key, prefix_key, prefix_tokens)
        except Exception as error:
            logger.info(
                "Vertex explicit cache gate unavailable; using inline request: {}",
                type(error).__name__,
            )
            return original
        if not ready:
            return original

        cache_parts = _stable_cache_parts(original)
        if cache_parts is None:
            return original
        cache_name = await self._get_or_create_cache(
            prefix_key, endpoint_key, model, prefix_tokens, price, cache_parts
        )
        if cache_name is None:
            return original

        prepared = deepcopy(original)
        prepared["messages"] = [
            item for item in prepared.get("messages", [])
            if not isinstance(item, Mapping) or item.get("role") not in {"system", "developer"}
        ]
        prepared.pop("tools", None)
        prepared.pop("functions", None)
        extra = prepared.setdefault("extra_body", {})
        google = extra.setdefault("google", {})
        google["cached_content"] = cache_name
        return prepared

    async def _get_or_create_cache(
        self, prefix_key, endpoint_key, model, prefix_tokens, price, cache_parts
    ):
        async with self._creation_lock:
            try:
                cached = await self._store.get(prefix_key)
                now = datetime.now(timezone.utc)
                if _valid_cache_record(
                    cached,
                    now,
                    model=model,
                    endpoint_key=endpoint_key,
                    resource_prefix=(
                        f"projects/{self._project_id}/locations/{self._location}/cachedContents/"
                    ),
                ):
                    return cached["resource_name"]
                if cached is not None:
                    await self._store.delete(prefix_key)
                cache_name = await self._create_cache(model, prefix_key, prefix_tokens, price, cache_parts)
                await self._store.put(
                    prefix_key,
                    {
                        "resource_name": cache_name,
                        "model": model,
                        "endpoint_key": endpoint_key,
                        "prefix_tokens": prefix_tokens,
                        "created_at": now.isoformat(),
                        "expires_at": (
                            now + timedelta(seconds=max(1, price.ttl_seconds - 30))
                        ).isoformat(),
                        "ttl_seconds": price.ttl_seconds,
                    },
                )
                return cache_name
            except Exception as error:
                logger.info(
                    "Vertex explicit cache unavailable; using inline request: {}",
                    type(error).__name__,
                )
                return None

    async def _create_cache(self, model, prefix_key, prefix_tokens, price, parts):
        del prefix_key
        model_id = model.removeprefix("google/")
        model_name = (
            f"projects/{self._project_id}/locations/{self._location}/"
            f"publishers/google/models/{quote(model_id, safe='') }"
        )
        payload: dict[str, Any] = {
            "model": model_name,
            "ttl": f"{price.ttl_seconds}s",
            "displayName": f"fcc-{model_id[:32]}-{prefix_tokens}",
        }
        if parts["system_instruction"]:
            payload["systemInstruction"] = {
                "role": "system",
                "parts": [{"text": parts["system_instruction"]}],
            }
        if parts["tools"]:
            payload["tools"] = [{"functionDeclarations": parts["tools"]}]
        token = await self._access_token_provider()
        response = await self._client.post(
            self._cache_url(),
            headers={"Authorization": f"Bearer {token}"},
            json=payload,
        )
        response.raise_for_status()
        data = response.json()
        name = data.get("name") if isinstance(data, Mapping) else None
        expected = f"projects/{self._project_id}/locations/{self._location}/cachedContents/"
        if not isinstance(name, str) or not name.startswith(expected):
            raise ValueError("Vertex cache create response did not contain a scoped resource name")
        return name

    def _cache_url(self) -> str:
        return (
            f"{vertex_service_endpoint(self._location)}/v1/projects/"
            f"{quote(self._project_id, safe='')}/locations/{self._location}/cachedContents"
        )


def _caller_cache_is_present(body: Mapping[str, Any]) -> bool:
    extra = body.get("extra_body")
    google = extra.get("google") if isinstance(extra, Mapping) else None
    return isinstance(google, Mapping) and bool(google.get("cached_content"))


def _stable_cache_parts(body: Mapping[str, Any]) -> dict[str, Any] | None:
    messages = body.get("messages")
    if not isinstance(messages, list):
        return None
    instructions = []
    dynamic_messages = False
    for item in messages:
        if not isinstance(item, Mapping):
            return None
        if item.get("role") in {"system", "developer"} and not dynamic_messages:
            text = _message_text(item.get("content"))
            if text is None:
                return None
            if text:
                instructions.append(text)
        else:
            dynamic_messages = True
            if item.get("role") in {"system", "developer"}:
                return None
    tools = _tool_declarations(body.get("tools"))
    if body.get("tools") and tools is None:
        return None
    if not instructions and not tools:
        return None
    return {
        "system_instruction": "\n\n".join(instructions),
        "tools": tools or [],
    }


def _message_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if not isinstance(item, Mapping) or item.get("type") not in {"text", "input_text"}:
                return None
            text = item.get("text")
            if not isinstance(text, str):
                return None
            pieces.append(text)
        return "\n".join(pieces)
    return None


def _tool_declarations(value: Any) -> list[dict[str, Any]] | None:
    if value is None:
        return []
    if not isinstance(value, list):
        return None
    declarations = []
    for tool in value:
        if not isinstance(tool, Mapping) or tool.get("type") != "function":
            return None
        function = tool.get("function")
        if not isinstance(function, Mapping) or not isinstance(function.get("name"), str):
            return None
        declaration = {key: function[key] for key in ("name", "description", "parameters") if key in function}
        declarations.append(declaration)
    return declarations


def _valid_cache_record(
    record: Any,
    now: datetime,
    *,
    model: str,
    endpoint_key: str,
    resource_prefix: str,
) -> bool:
    if (
        not isinstance(record, Mapping)
        or record.get("model") != model
        or record.get("endpoint_key") != endpoint_key
        or not isinstance(record.get("resource_name"), str)
        or not record["resource_name"].startswith(resource_prefix)
    ):
        return False
    try:
        expiry = datetime.fromisoformat(record["expires_at"])
        return expiry > now
    except (KeyError, TypeError, ValueError):
        return False


def _is_digest(value: str) -> bool:
    return re.fullmatch(r"[0-9a-f]{64}", value) is not None
