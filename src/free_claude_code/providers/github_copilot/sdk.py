"""Official Copilot SDK boundary; never run SDK agent turns."""

import asyncio
import re
import tempfile
import uuid
from pathlib import Path
from types import MappingProxyType
from typing import Literal
from urllib.parse import urlsplit

import httpx
from copilot import (
    CapiSessionOptions,
    CopilotClient,
    CopilotSession,
    PermissionRequest,
    RuntimeConnection,
)
from copilot.rpc import (
    AdaptiveThinkingSupport,
    Model,
    ModelPolicyState,
    ModelsListRequest,
    PermissionDecisionReject,
    ProviderEndpoint,
    ProviderTransport,
    ProviderType,
    ProviderWireAPI,
    SessionProviderGetEndpointRequest,
)
from copilot.session import PermissionInvocation

from free_claude_code.application.model_metadata import ProviderModelInfo
from free_claude_code.core.model_capabilities import ModelInputModality
from free_claude_code.core.reasoning import ReasoningCapability
from free_claude_code.providers.anthropic_messages.request_policy import (
    MessagesModelCapabilities,
)
from free_claude_code.providers.endpoint_types import HttpEndpoint
from free_claude_code.providers.model_listing import reasoning_capability_from_options

from .lifecycle import drain_owned
from .native_cli import COPILOT_CLI_VERSION, profile_environment, verified_cli_path
from .types import (
    CopilotEgress,
    CopilotEndpoint,
    CopilotIdentity,
    CopilotModel,
    CopilotUnavailable,
)

SDK_RPC_TIMEOUT = 30.0
_HEADER_NAME = re.compile(r"[!#$%&'*+.^_`|~0-9A-Za-z-]+")


def _deny_permission(
    _request: PermissionRequest, _invocation: PermissionInvocation
) -> PermissionDecisionReject:
    return PermissionDecisionReject(feedback="FCC owns tool execution.")


async def _delete_session_record(
    client: CopilotClient, session_id: str, *, allow_absent: bool = True
) -> None:
    try:
        async with asyncio.timeout(SDK_RPC_TIMEOUT):
            await client.delete_session(session_id)
    except Exception:
        # Endpoint-only sessions need not have a persisted record. Check this
        # exact owned UUID; neither error prose nor a failed lookup proves absence.
        if not allow_absent:
            raise
        async with asyncio.timeout(SDK_RPC_TIMEOUT):
            metadata = await client.get_session_metadata(session_id)
        if metadata is not None:
            raise



class _DirectSdkSession:
    def __init__(self, endpoint: CopilotEndpoint) -> None:
        self._endpoint = endpoint

    async def endpoint(self) -> CopilotEndpoint:
        return self._endpoint

    async def close(self) -> None:
        return None


class SdkRuntime:
    """Own a hidden official runtime with native authentication and inert sessions."""

    def __init__(self) -> None:
        self._client: CopilotClient | None = None
        self._directory: tempfile.TemporaryDirectory[str] | None = None
        self._sessions: list[SdkSession] = []
        self._pending_session_ids: set[str] = set()
        self._direct_copilot_token: str | None = None
        self._direct_copilot_expiry: float = 0.0
        self._direct_api_url: str = "https://api.githubcopilot.com"
        self._direct_models: dict[str, CopilotModel] = {}
        self._direct_routes: dict[str, CopilotEgress] = {}
        self._direct_models_loaded_at: float = 0.0
        self._lock = asyncio.Lock()
        self._close_task: asyncio.Task[None] | None = None
        self._closing = False
        self._stopped = False

    async def start(self) -> None:
        async with self._lock:
            if self._closing:
                raise CopilotUnavailable("Copilot runtime has closed.")
            if self._client is not None:
                return
            env = profile_environment()
            token_path = Path.home() / ".copilot-termux" / "oauth-token"
            copilot_token = token_path.read_text().strip() if token_path.is_file() else ""
            if copilot_token:
                env["COPILOT_GITHUB_TOKEN"] = copilot_token
            path = await verified_cli_path(env)
            env["COPILOT_ALLOW_GET_PROVIDER_ENDPOINT"] = "true"
            directory = tempfile.TemporaryDirectory(prefix="fcc-copilot-")
            self._directory = directory
            try:
                client = CopilotClient(
                    connection=RuntimeConnection.for_stdio(
                        path=path, args=("--disable-builtin-mcps",)
                    ),
                    env=env,
                    github_token=copilot_token or None,
                    working_directory=directory.name,
                    use_logged_in_user=True,
                    mode="copilot-cli",
                    log_level="error",
                )
                self._client = client
                async with asyncio.timeout(SDK_RPC_TIMEOUT):
                    await client.start()
                    status = await client.get_status()
                if status.version != COPILOT_CLI_VERSION:
                    raise CopilotUnavailable(
                        "Copilot runtime version changed. Reconnect with the supported CLI version."
                    )
            except BaseException:
                self._closing = True
                await drain_owned(asyncio.create_task(self._close_client()))
                raise

    def _running(self) -> CopilotClient:
        if self._client is None:
            raise CopilotUnavailable("Copilot runtime is not connected.")
        return self._client


    def _direct_token(self) -> str:
        token_path = Path.home() / ".copilot-termux" / "oauth-token"
        token = token_path.read_text(encoding="utf-8").strip() if token_path.is_file() else ""
        if not token:
            raise CopilotUnavailable("Copilot OAuth token is missing. Reconnect Copilot in Admin.")
        return token

    async def _direct_refresh(self, *, force: bool = False) -> None:
        import time

        now = time.time()
        if (
            not force
            and self._direct_models
            and self._direct_copilot_token
            and self._direct_models_loaded_at > now - 60
            and self._direct_copilot_expiry > now + 60
        ):
            return

        github_token = self._direct_token()
        headers = {
            "Accept": "application/json",
            "Authorization": f"token {github_token}",
            "User-Agent": f"copilot/{COPILOT_CLI_VERSION}",
            "Copilot-Integration-Id": "copilot-developer-cli",
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            exchange = await client.get(
                "https://api.github.com/copilot_internal/v2/token",
                headers=headers,
            )
            if not exchange.is_success:
                raise CopilotUnavailable(
                    f"Copilot token exchange failed (HTTP {exchange.status_code}). Reconnect Copilot in Admin."
                )
            payload = exchange.json()
            copilot_token = payload.get("token")
            if not isinstance(copilot_token, str) or not copilot_token:
                raise CopilotUnavailable("GitHub did not return a Copilot API token. Reconnect Copilot in Admin.")
            endpoints = payload.get("endpoints")
            api_url = endpoints.get("api") if isinstance(endpoints, dict) else None
            if not isinstance(api_url, str) or not api_url:
                api_url = "https://api.githubcopilot.com"
            api_url = api_url.rstrip("/")
            expiry = payload.get("expires_at")
            try:
                expiry_value = float(expiry)
            except (TypeError, ValueError):
                expiry_value = now + 1500

            api_headers = {
                "Accept": "application/json",
                "Authorization": f"Bearer {copilot_token}",
                "User-Agent": f"copilot/{COPILOT_CLI_VERSION}",
                "Editor-Version": f"copilot/{COPILOT_CLI_VERSION}",
                "Copilot-Integration-Id": "copilot-developer-cli",
                "X-GitHub-Api-Version": "2026-08-01",
            }
            models_response = await client.get(f"{api_url}/models", headers=api_headers)
            if not models_response.is_success:
                raise CopilotUnavailable(
                    f"Copilot model discovery failed (HTTP {models_response.status_code}). Check the selected account and its model policies."
                )
            body = models_response.json()
            items = body.get("data", body.get("models", [])) if isinstance(body, dict) else []
            if not isinstance(items, list):
                items = []

            models: dict[str, CopilotModel] = {}
            routes: dict[str, CopilotEgress] = {}
            for item in items:
                if not isinstance(item, dict):
                    continue
                model_id = item.get("id")
                if not isinstance(model_id, str) or not model_id.strip():
                    continue
                policy = item.get("policy")
                if isinstance(policy, dict) and policy.get("state") not in (None, "enabled"):
                    continue
                caps = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
                limits = caps.get("limits") if isinstance(caps.get("limits"), dict) else {}
                supports = caps.get("supports") if isinstance(caps.get("supports"), dict) else {}
                supported_endpoints = item.get("supported_endpoints")
                endpoints_set = {
                    str(x).strip()
                    for x in supported_endpoints
                    if isinstance(x, str)
                } if isinstance(supported_endpoints, list) else set()

                if "/v1/messages" in endpoints_set and "/chat/completions" not in endpoints_set:
                    egress = CopilotEgress.MESSAGES
                elif "/responses" in endpoints_set and "/chat/completions" not in endpoints_set:
                    egress = CopilotEgress.RESPONSES
                elif "/v1/messages" in endpoints_set and model_id.lower().startswith("claude-"):
                    egress = CopilotEgress.MESSAGES
                elif (
                    "/responses" in endpoints_set
                    and model_id.lower().startswith(("gpt-5", "gpt-6"))
                    and not model_id.lower().startswith("gpt-5-mini")
                ):
                    egress = CopilotEgress.RESPONSES
                else:
                    egress = CopilotEgress.CHAT

                efforts_raw = supports.get("reasoning_effort")
                efforts = (
                    tuple(str(x).strip() for x in efforts_raw if isinstance(x, str) and x.strip())
                    if isinstance(efforts_raw, list)
                    else None
                )
                adaptive_raw = supports.get("adaptive_thinking")
                adaptive = adaptive_raw if adaptive_raw in {"unsupported", "optional", "required"} else None
                max_output = limits.get("max_output_tokens")
                max_output = max_output if isinstance(max_output, int) and max_output > 0 else None
                context_window = limits.get("max_context_window_tokens")
                context_window = context_window if isinstance(context_window, int) and context_window > 0 else None

                reasoning_capability = (
                    ReasoningCapability.REQUIRED
                    if adaptive == "required"
                    else ReasoningCapability.OPTIONAL
                    if adaptive == "optional" or efforts
                    else ReasoningCapability.NONE
                    if adaptive == "unsupported"
                    else ReasoningCapability.UNKNOWN
                )
                info = ProviderModelInfo(
                    model_id=model_id,
                    supports_thinking=adaptive not in (None, "unsupported"),
                    context_window_tokens=context_window,
                    max_output_tokens=max_output,
                    reasoning_capability=reasoning_capability,
                )
                messages = MessagesModelCapabilities(
                    max_output_tokens=max_output,
                    adaptive_thinking=adaptive,
                    supports_output_effort=bool(efforts),
                    supported_efforts=efforts,
                    supports_vision=supports.get("vision") if isinstance(supports.get("vision"), bool) else None,
                )
                models[model_id] = CopilotModel(
                    info=info,
                    messages=messages,
                    supported_efforts=efforts,
                    default_effort=efforts[-1] if efforts else None,
                )
                routes[model_id] = egress

            if not models:
                raise CopilotUnavailable(
                    "This GitHub account exposes no enabled Copilot models. Check its subscription and model policies."
                )

            self._direct_copilot_token = copilot_token
            self._direct_copilot_expiry = expiry_value
            self._direct_api_url = api_url
            self._direct_models = models
            self._direct_routes = routes
            self._direct_models_loaded_at = now

    async def _direct_identity(self) -> CopilotIdentity:
        token = self._direct_token()
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://api.github.com/user",
                headers={
                    "Accept": "application/json",
                    "Authorization": f"Bearer {token}",
                    "User-Agent": "Free-Claude-Code",
                },
            )
        if not response.is_success:
            raise CopilotUnavailable("GitHub OAuth token is invalid or expired. Reconnect Copilot in Admin.")
        profile = response.json()
        login = profile.get("login") if isinstance(profile, dict) else None
        if not isinstance(login, str) or not login:
            raise CopilotUnavailable("GitHub did not return a valid account profile. Reconnect Copilot in Admin.")
        return CopilotIdentity("github.com", login)

    async def _direct_endpoint(self, model_id: str) -> CopilotEndpoint:
        await self._direct_refresh()
        model = self._direct_models.get(model_id)
        if model is None:
            raise CopilotUnavailable("This model is not available from the connected Copilot account. Refresh the model list.")
        token = self._direct_copilot_token
        if not token:
            raise CopilotUnavailable("Copilot API credentials are unavailable. Reconnect Copilot in Admin.")
        route = self._direct_routes.get(model_id, CopilotEgress.CHAT)
        return CopilotEndpoint(
            egress=route,
            http=HttpEndpoint(
                base_url=self._direct_api_url,
                headers={
                    "Authorization": f"Bearer {token}",
                    "Accept": "application/json",
                    "Content-Type": "application/json",
                    "User-Agent": f"copilot/{COPILOT_CLI_VERSION}",
                    "Editor-Version": f"copilot/{COPILOT_CLI_VERSION}",
                    "Copilot-Integration-Id": "copilot-developer-cli",
                    "X-GitHub-Api-Version": "2026-08-01",
                },
                api_key=token,
            ),
                expires_at=self._direct_copilot_expiry,
        )

    async def identity(self) -> CopilotIdentity | None:
        try:
            async with asyncio.timeout(SDK_RPC_TIMEOUT):
                status = await self._running().get_auth_status()
        except Exception:
            raise CopilotUnavailable(
                "Could not read Copilot profile status. Reconnect Copilot."
            ) from None
        if not status.isAuthenticated:
            return await self._direct_identity()
            token_path = Path.home() / ".copilot-termux" / "oauth-token"
            token = token_path.read_text().strip() if token_path.is_file() else ""
            if token:
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(
                            "https://api.github.com/user",
                            headers={
                                "Accept": "application/vnd.github+json",
                                "Authorization": f"Bearer {token}",
                                "User-Agent": "Free-Claude-Code",
                            },
                        )
                    if response.is_success:
                        profile = response.json()
                        login = profile.get("login") if isinstance(profile, dict) else None
                        if isinstance(login, str) and login:
                            return CopilotIdentity("github.com", login)
                except Exception:
                    pass
            return None
        if not status.login or not status.host:
            raise CopilotUnavailable(
                "Copilot did not report the selected GitHub profile."
            )
        host = urlsplit(
            status.host if "://" in status.host else f"https://{status.host}"
        ).hostname
        if not host:
            raise CopilotUnavailable("Copilot reported an invalid GitHub profile host.")
        return CopilotIdentity(host.lower(), status.login)

    async def models(self) -> tuple[CopilotModel, ...]:
        try:
            result = await self._running().rpc.models.list(
                ModelsListRequest(), timeout=SDK_RPC_TIMEOUT
            )
            return tuple(
                _model(model)
                for model in result.models
                if model.id.strip()
                and model.id.strip().casefold() != "auto"
                and (
                    model.policy is None
                    or model.policy.state is ModelPolicyState.ENABLED
                )
            )
        except Exception:
            await self._direct_refresh(force=True)
            return tuple(self._direct_models.values())

    async def session(self, model_id: str) -> SdkSession:
        client = self._running()
        directory = self._directory
        if directory is None:
            raise CopilotUnavailable("Copilot working directory is unavailable.")
        session_id = str(uuid.uuid4())
        config = Path(directory.name) / session_id
        config.mkdir()
        self._pending_session_ids.add(session_id)
        try:
            async with asyncio.timeout(SDK_RPC_TIMEOUT):
                session = await client.create_session(
                    session_id=session_id,
                    model=model_id,
                    working_directory=directory.name,
                    config_directory=str(config),
                    available_tools=[],
                    on_permission_request=_deny_permission,
                    enable_config_discovery=False,
                    skip_custom_instructions=True,
                    custom_agents_local_only=True,
                    enable_file_hooks=False,
                    enable_on_demand_instruction_discovery=False,
                    enable_host_git_operations=False,
                    enable_skills=False,
                    included_builtin_skills=[],
                    request_extensions=False,
                    enable_session_store=False,
                    infinite_sessions={"enabled": False},
                    memory={"enabled": False},
                    skip_embedding_retrieval=True,
                    embedding_cache_storage="in-memory",
                    mcp_oauth_token_storage="in-memory",
                    enable_session_telemetry=False,
                    capi=CapiSessionOptions(enable_web_socket_responses=False),
                )
        except Exception:
            try:
                await self._direct_refresh(force=True)
                endpoint = await self._direct_endpoint(model_id)
                return _DirectSdkSession(endpoint)
            except CopilotUnavailable:
                raise
            except Exception:
                raise CopilotUnavailable(
                    "Could not open a Copilot endpoint session. Reconnect Copilot."
                ) from None
        owned = SdkSession(client, session, model_id, session_id)
        self._sessions.append(owned)
        self._pending_session_ids.discard(session_id)
        return owned

    async def close(self) -> None:
        self._closing = True
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        await drain_owned(self._close_task)

    async def _close(self) -> None:
        async with self._lock:
            await self._close_client()

    async def _close_client(self) -> None:
        client = self._client
        if client is None:
            if self._directory is not None:
                self._directory.cleanup()
                self._directory = None
            return
        records_only = self._stopped
        try:
            # A prior shutdown can stop the process while record deletion fails.
            # Retain that ownership and reconnect only to retry our own deletions.
            if self._stopped:
                self._stopped = False
                try:
                    async with asyncio.timeout(SDK_RPC_TIMEOUT):
                        await client.start()
                except Exception:
                    raise CopilotUnavailable(
                        "Could not reconnect Copilot to remove an FCC session record. Retry disconnect to finish cleanup."
                    ) from None
            failed_sessions: list[SdkSession] = []
            for session in self._sessions:
                try:
                    await session.close(runtime_stopped=records_only)
                except Exception:
                    failed_sessions.append(session)
            self._sessions = failed_sessions
            # A cancelled create RPC can still finish remotely. Do not accept
            # absence until the process that could create its record has stopped.
            for session_id in tuple(self._pending_session_ids) if records_only else ():
                try:
                    await _delete_session_record(client, session_id)
                    self._pending_session_ids.discard(session_id)
                except Exception:
                    pass
        finally:
            # start() may fail after spawning a process. Stop that partial runtime
            # too, without discarding record IDs still owned by this adapter.
            try:
                async with asyncio.timeout(SDK_RPC_TIMEOUT):
                    await client.stop()
            except Exception:
                try:
                    async with asyncio.timeout(SDK_RPC_TIMEOUT):
                        await client.force_stop()
                except Exception:
                    raise CopilotUnavailable(
                        "Could not stop the Copilot runtime. Retry disconnect to finish cleanup."
                    ) from None
            self._stopped = True
        if self._pending_session_ids and not records_only:
            # Reconnect solely to clean known UUIDs after their writer is gone.
            await self._close_client()
            return
        if self._sessions or self._pending_session_ids:
            raise CopilotUnavailable(
                "Copilot stopped, but an FCC session record could not be removed. Retry disconnect to finish cleanup."
            )
        self._client = None
        if self._directory is not None:
            self._directory.cleanup()
            self._directory = None


class SdkSession:
    def __init__(
        self,
        client: CopilotClient,
        session: CopilotSession,
        model_id: str,
        session_id: str,
    ) -> None:
        self._client = client
        self._session = session
        self._model_id = model_id
        self._session_id = session_id
        self._closed = False
        self._disconnected = False
        self._close_task: asyncio.Task[None] | None = None

    async def endpoint(self) -> CopilotEndpoint:
        if self._closed:
            raise CopilotUnavailable("Copilot endpoint session is closed.")
        try:
            endpoint = await self._session.rpc.provider.get_endpoint(
                SessionProviderGetEndpointRequest(model_id=self._model_id),
                timeout=SDK_RPC_TIMEOUT,
            )
            return _endpoint(endpoint, self._model_id)
        except CopilotUnavailable:
            raise
        except Exception:
            raise CopilotUnavailable(
                "Could not resolve this model's Copilot endpoint. Refresh models or reconnect Copilot."
            ) from None

    async def close(self, *, runtime_stopped: bool = False) -> None:
        if runtime_stopped:
            # Only SdkRuntime, after stopping the owning process, supplies this.
            self._disconnected = True
        if self._closed:
            return
        if self._close_task is None or self._close_task.done():
            self._close_task = asyncio.create_task(self._close())
        await drain_owned(self._close_task)

    async def _close(self) -> None:
        if not self._disconnected:
            try:
                async with asyncio.timeout(SDK_RPC_TIMEOUT):
                    await self._session.disconnect()
                self._disconnected = True
            except Exception:
                # Deletion and runtime stop also release server-side ownership.
                pass
        await _delete_session_record(
            self._client, self._session_id, allow_absent=self._disconnected
        )
        self._closed = True


def _model(model: Model) -> CopilotModel:
    limits = model.capabilities.limits
    supports = model.capabilities.supports
    adaptive: Literal["unsupported", "optional", "required"] | None = None
    if supports is not None and supports.adaptive_thinking is not None:
        modes: dict[
            AdaptiveThinkingSupport, Literal["unsupported", "optional", "required"]
        ] = {
            AdaptiveThinkingSupport.UNSUPPORTED: "unsupported",
            AdaptiveThinkingSupport.OPTIONAL: "optional",
            AdaptiveThinkingSupport.REQUIRED: "required",
        }
        adaptive = modes[supports.adaptive_thinking]
    vision = supports.vision if supports is not None else None
    effort_support = supports.reasoning_effort if supports is not None else None
    efforts = (
        tuple(model.supported_reasoning_efforts)
        if model.supported_reasoning_efforts is not None
        else None
    )
    output = limits.max_output_tokens if limits is not None else None
    context = limits.max_context_window_tokens if limits is not None else None
    info = ProviderModelInfo(
        model_id=model.id,
        reasoning_capability=reasoning_capability_from_options(
            {"supported_efforts": efforts}
        ),
        supports_thinking=True
        if effort_support is True or adaptive in {"optional", "required"}
        else None,
        input_modalities=frozenset({ModelInputModality.TEXT, ModelInputModality.IMAGE})
        if vision is True
        else frozenset({ModelInputModality.TEXT})
        if vision is False
        else None,
        context_window_tokens=context if context is not None and context > 0 else None,
        max_output_tokens=output if output is not None and output > 0 else None,
    )
    return CopilotModel(
        info,
        MessagesModelCapabilities(
            max_output_tokens=info.max_output_tokens,
            adaptive_thinking=adaptive,
            supports_output_effort=effort_support,
            supported_efforts=efforts,
            supports_vision=vision,
        ),
        efforts,
        model.default_reasoning_effort,
    )


def _endpoint(endpoint: ProviderEndpoint, model_id: str) -> CopilotEndpoint:
    if endpoint.transport not in {None, ProviderTransport.HTTP}:
        raise CopilotUnavailable(
            "This Copilot model requires an unsupported upstream transport."
        )
    if endpoint.type is ProviderType.ANTHROPIC and endpoint.wire_api is None:
        egress = CopilotEgress.MESSAGES
    elif endpoint.type is ProviderType.OPENAI:
        egress = (
            CopilotEgress.RESPONSES
            if endpoint.wire_api is ProviderWireAPI.RESPONSES
            else CopilotEgress.CHAT
        )
    else:
        raise CopilotUnavailable(
            "This Copilot model requires an unsupported provider endpoint."
        )
    try:
        url = urlsplit(endpoint.base_url)
        valid_url = (
            url.scheme == "https"
            and bool(url.hostname)
            and url.username is None
            and url.password is None
            and not url.query
            and not url.fragment
            and not any(
                character.isspace() or ord(character) < 32
                for character in endpoint.base_url
            )
            and (url.port is None or url.port > 0)
        )
    except ValueError:
        valid_url = False
    if not valid_url:
        raise CopilotUnavailable("Copilot returned an invalid HTTPS endpoint.")
    headers = httpx.Headers(encoding="ascii")
    for key, value in endpoint.headers.items():
        if _HEADER_NAME.fullmatch(key) is None or any(
            character in value for character in "\r\n\0"
        ):
            raise CopilotUnavailable("Copilot returned invalid endpoint headers.")
        try:
            headers[key] = value
        except UnicodeError, ValueError:
            raise CopilotUnavailable(
                "Copilot returned invalid endpoint headers."
            ) from None
    expires = None
    token = endpoint.session_token
    if token is not None:
        if (
            token.model not in {None, model_id}
            or not token.token
            or _HEADER_NAME.fullmatch(token.header) is None
            or any(character in token.token for character in "\r\n\0")
        ):
            raise CopilotUnavailable(
                "Copilot returned an invalid or differently bound session token."
            )
        try:
            headers[token.header] = token.token
        except UnicodeError, ValueError:
            raise CopilotUnavailable(
                "Copilot returned an invalid session token."
            ) from None
        if token.expires_at is not None:
            if token.expires_at.tzinfo is None:
                raise CopilotUnavailable(
                    "Copilot returned a session expiry without a time zone."
                )
            expires = token.expires_at.timestamp()
    if endpoint.api_key is not None:
        if not endpoint.api_key or any(
            character in endpoint.api_key for character in "\r\n\0"
        ):
            raise CopilotUnavailable("Copilot returned an invalid endpoint credential.")
        try:
            endpoint.api_key.encode("ascii")
        except UnicodeError:
            raise CopilotUnavailable(
                "Copilot returned an invalid endpoint credential."
            ) from None
    return CopilotEndpoint(
        egress,
        HttpEndpoint(
            endpoint.base_url.rstrip("/"),
            MappingProxyType(dict(headers.items())),
            endpoint.api_key,
        ),
        expires,
    )
