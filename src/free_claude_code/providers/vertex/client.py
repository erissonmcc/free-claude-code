"""Google Vertex AI provider using the OpenAI-compatible Chat Completions API."""

from dataclasses import replace
from typing import Any

import httpx

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.anthropic import ReasoningReplayMode
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.providers.admission import (
    ProviderAdmissionController,
    ProviderOperationKind,
)
from free_claude_code.providers.base import ProviderConfig
from free_claude_code.providers.google_openai import (
    GoogleOpenAIProvider,
    VertexReasoningEncoder,
    validate_google_extra_body,
)
from free_claude_code.providers.http import maybe_await_aclose
from free_claude_code.core.context_hook import current_context_cache_observation
from free_claude_code.providers.model_listing import (
    ModelListResponseError,
)
from free_claude_code.providers.openai_chat import (
    OpenAIChatProfile,
    OpenAIChatRequestPolicy,
)

from .auth import GoogleAccessTokenProvider
from .context_cache import VertexContextCacheAdapter
from .endpoint import vertex_openai_base_url, vertex_publisher_models_url
from .models import extract_vertex_model_page

_REQUEST_POLICY = OpenAIChatRequestPolicy(
    provider_name="VERTEX",
    reasoning_replay=ReasoningReplayMode.REASONING_CONTENT,
    include_extra_body=True,
    extra_body_validator=validate_google_extra_body,
)
_PROFILE = OpenAIChatProfile(_REQUEST_POLICY, VertexReasoningEncoder())


class VertexProvider(GoogleOpenAIProvider):
    """Vertex AI Gemini models with renewable ADC and native model discovery."""

    def __init__(
        self,
        config: ProviderConfig,
        *,
        project_id: str,
        location: str,
        admission: ProviderAdmissionController,
        access_token_provider: GoogleAccessTokenProvider | None = None,
        cache_store=None,
        cache_readiness=None,
        cache_prices=(),
    ) -> None:
        self._project_id = project_id.strip()
        self._location = location.strip().lower()
        base_url = vertex_openai_base_url(self._project_id, self._location)
        self._models_url = vertex_publisher_models_url(self._location)
        self._access_token_provider = (
            access_token_provider or GoogleAccessTokenProvider(proxy=config.proxy)
        )
        self._model_list_client = httpx.AsyncClient(
            proxy=config.proxy,
            timeout=httpx.Timeout(
                config.http_read_timeout,
                connect=config.http_connect_timeout,
                read=config.http_read_timeout,
                write=config.http_write_timeout,
            ),
        )
        self._cache_client = None
        request_body_preparer = None
        if cache_store is not None and cache_readiness is not None and cache_prices:
            self._cache_client = httpx.AsyncClient(
                proxy=config.proxy,
                timeout=httpx.Timeout(
                    config.http_read_timeout,
                    connect=config.http_connect_timeout,
                    read=config.http_read_timeout,
                    write=config.http_write_timeout,
                ),
            )
            self._context_cache = VertexContextCacheAdapter(
                project_id=self._project_id,
                location=self._location,
                access_token_provider=self._access_token_provider,
                client=self._cache_client,
                store=cache_store,
                readiness=cache_readiness,
                prices=cache_prices,
            )

            async def prepare_cached_body(body):
                return await self._context_cache.prepare(
                    body, observation=current_context_cache_observation()
                )

            request_body_preparer = prepare_cached_body
        super().__init__(
            replace(config, base_url=base_url),
            profile=_PROFILE,
            admission=admission,
            api_key_provider=self._access_token_provider,
            default_headers={"x-goog-user-project": self._project_id},
            request_body_preparer=request_body_preparer,
            context_endpoint_scope=base_url + "|" + self._project_id,
        )

    async def cleanup(self) -> None:
        """Release both OpenAI-compatible and native model-list clients."""
        try:
            await super().cleanup()
        finally:
            await self._model_list_client.aclose()
            if self._cache_client is not None:
                await self._cache_client.aclose()

    async def list_model_infos(self) -> frozenset[ProviderModelInfo]:
        """List Vertex publisher models and translate their resource names."""
        model_ids: set[str] = set()
        page_token: str | None = None
        seen_page_tokens: set[str] = set()
        while True:
            payload = await self._list_model_page(page_token)
            page_ids, page_token = extract_vertex_model_page(payload)
            model_ids.update(page_ids)
            if page_token is None:
                break
            if page_token in seen_page_tokens:
                raise ModelListResponseError(
                    "VERTEX model-list response is malformed: repeated nextPageToken"
                )
            seen_page_tokens.add(page_token)
        if not model_ids:
            raise ModelListResponseError(
                "VERTEX model-list response is malformed: response did not include "
                "any model ids"
            )
        # Vertex publisher-model discovery returns IDs only. Codex uses the
        # advertised input modalities to decide whether its native view_image
        # tool is available, so declare image input for Gemini generation models.
        return frozenset(
            ProviderModelInfo(
                model_id=model_id,
                input_modalities=(
                    frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE})
                    if model_id.startswith("google/gemini-")
                    and "embedding" not in model_id.lower()
                    else None
                ),
            )
            for model_id in model_ids
        )

    async def _list_model_page(self, page_token: str | None) -> Any:
        async def request() -> httpx.Response:
            token = await self._access_token_provider()
            response = await self._model_list_client.get(
                self._models_url,
                params={"pageToken": page_token} if page_token else None,
                headers={
                    "Authorization": f"Bearer {token}",
                    "x-goog-user-project": self._project_id,
                },
            )
            try:
                response.raise_for_status()
            except Exception:
                await maybe_await_aclose(response)
                raise
            return response

        execution = self._admission.start_execution()
        response = await execution.run_call(
            request,
            operation_kind=ProviderOperationKind.MODEL_DISCOVERY,
        )
        try:
            try:
                return response.json()
            except ValueError as exc:
                raise ModelListResponseError(
                    "VERTEX model-list response is malformed: invalid JSON"
                ) from exc
        finally:
            await maybe_await_aclose(response)
