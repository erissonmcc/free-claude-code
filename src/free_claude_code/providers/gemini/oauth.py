"""Google OAuth authorization-code flow for Vertex AI connected accounts."""

import asyncio
import base64
import hashlib
import json
import os
import secrets
import time
import uuid
from contextlib import suppress
from pathlib import Path
from urllib.parse import urlencode

import httpx
from aiohttp import web

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
    ConnectedAccountStatus,
)
from free_claude_code.config.env_files import dotenv_values_from_file
from free_claude_code.config.paths import config_dir_path, managed_env_path

GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
GOOGLE_REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GOOGLE_USERINFO_URL = "https://openidconnect.googleapis.com/v1/userinfo"
GOOGLE_SCOPES = (
    "openid",
    "email",
    "https://www.googleapis.com/auth/cloud-platform",
    "https://www.googleapis.com/auth/generative-language.retriever",
)
LOGIN_TIMEOUT_SECONDS = 600


class GoogleOAuthError(RuntimeError):
    """Google OAuth could not be completed."""


class _GoogleBrowserLogin:
    def __init__(self, *, url: str, verifier: str, runner: web.AppRunner,
                 result: asyncio.Future[dict[str, str]], redirect_uri: str) -> None:
        self.authorization_url = url
        self.verifier = verifier
        self.runner = runner
        self.result = result
        self.redirect_uri = redirect_uri
        self._closed = False

    @classmethod
    async def start(cls, client_id: str) -> "_GoogleBrowserLogin":
        state = secrets.token_urlsafe(32)
        verifier = secrets.token_urlsafe(64)
        challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).decode().rstrip("=")
        loop = asyncio.get_running_loop()
        result: asyncio.Future[dict[str, str]] = loop.create_future()
        runner: web.AppRunner | None = None
        redirect_uri = ""
        for port in (1458, 1459, 1460):
            candidate = f"http://localhost:{port}/oauth2callback"
            app = web.Application()

            async def callback(request: web.Request, expected_redirect: str = candidate) -> web.Response:
                if request.query.get("state") != state:
                    return web.Response(status=400, text="OAuth state mismatch. Return to FCC and retry.")
                if error := request.query.get("error"):
                    if not result.done():
                        result.set_exception(GoogleOAuthError(error))
                    return web.Response(status=400, text="Google sign-in failed. You can close this tab.")
                code = request.query.get("code")
                if not code:
                    return web.Response(status=400, text="Google did not return an authorization code.")
                if not result.done():
                    result.set_result({"code": code, "redirect_uri": expected_redirect})
                return web.Response(text="Google account connected to FCC. You can close this tab.", content_type="text/plain")

            app.router.add_get("/oauth2callback", callback)
            candidate_runner = web.AppRunner(app, access_log=None)
            await candidate_runner.setup()
            try:
                await web.TCPSite(candidate_runner, "127.0.0.1", port).start()
            except OSError:
                await candidate_runner.cleanup()
                continue
            runner, redirect_uri = candidate_runner, candidate
            break
        if runner is None:
            raise GoogleOAuthError("Could not start the Google OAuth callback on localhost ports 1458–1460.")
        params = {
            "client_id": client_id, "redirect_uri": redirect_uri,
            "response_type": "code", "scope": " ".join(GOOGLE_SCOPES),
            "state": state, "code_challenge": challenge,
            "code_challenge_method": "S256", "access_type": "offline",
            "prompt": "consent", "include_granted_scopes": "true",
        }
        return cls(url=f"{GOOGLE_AUTH_URL}?{urlencode(params)}", verifier=verifier,
                   runner=runner, result=result, redirect_uri=redirect_uri)

    async def wait(self) -> dict[str, str]:
        return await asyncio.wait_for(self.result, timeout=LOGIN_TIMEOUT_SECONDS)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if not self.result.done():
            self.result.cancel()
        await self.runner.cleanup()


class GoogleOAuthAuthManager:
    """Persist Google OAuth credentials and expose safe connected-account state.

    Configure GOOGLE_OAUTH_CLIENT_ID and GOOGLE_OAUTH_CLIENT_SECRET using a Google
    Cloud OAuth client. This authorizes Gemini API access through a Google Cloud project;
    it does not grant a paid AI Studio subscription or bypass API quotas/billing.
    """

    provider_id = "gemini_oauth"

    def __init__(self, *, credential_path: Path | None = None) -> None:
        self._path = credential_path or (config_dir_path() / "auth" / "google_gemini.json")
        managed_values: dict[str, str] = {}
        with suppress(FileNotFoundError, OSError, ValueError):
            managed_values = dotenv_values_from_file(managed_env_path())
        self._client_id = (os.getenv("GOOGLE_OAUTH_CLIENT_ID") or managed_values.get("GOOGLE_OAUTH_CLIENT_ID", "")).strip()
        self._client_secret = (os.getenv("GOOGLE_OAUTH_CLIENT_SECRET") or managed_values.get("GOOGLE_OAUTH_CLIENT_SECRET", "")).strip()
        self._project_id = (os.getenv("GEMINI_OAUTH_PROJECT_ID") or managed_values.get("GEMINI_OAUTH_PROJECT_ID", "")).strip()
        self._credentials: dict[str, object] | None = None
        self._revision = 0
        self._last_error: str | None = None
        self._login_task: asyncio.Task[None] | None = None
        self._browser: _GoogleBrowserLogin | None = None
        self._attempt_id: str | None = None
        self._authorization_url: str | None = None
        self._expires_at: int | None = None
        self._operation_lock = asyncio.Lock()
        self._closed = False
        self._load_saved()

    def _load_saved(self) -> None:
        try:
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if isinstance(payload, dict) and isinstance(payload.get("refresh_token"), str):
                self._credentials = payload
                self._revision = 1
        except FileNotFoundError:
            pass
        except (OSError, ValueError, json.JSONDecodeError):
            self._last_error = "Saved Google credentials are invalid. Disconnect and reconnect."

    def is_connected(self) -> bool:
        return self._credentials is not None

    def connected_provider_ids(self) -> tuple[str, ...]:
        return (self.provider_id,) if self.is_connected() else ()

    def status(self) -> ConnectedAccountStatus:
        connecting = self._login_task is not None and not self._login_task.done()
        email = self._credentials.get("email") if self._credentials else None
        configured = bool(self._client_id and self._client_secret and self._project_id)
        if not self._client_id or not self._client_secret:
            setup_message = "Set Google OAuth Client ID and Client Secret in provider settings, then restart FCC."
        elif not self._project_id:
            setup_message = "Set Google Cloud Project ID (Gemini OAuth) in provider settings, then restart FCC."
        else:
            setup_message = None
        message = self._last_error or (None if configured else setup_message)
        return ConnectedAccountStatus(
            provider_id=self.provider_id,
            state=ConnectedAccountState.CONNECTING if connecting else
                  ConnectedAccountState.ERROR if self._last_error else
                  ConnectedAccountState.CONNECTED if self._credentials else
                  ConnectedAccountState.DISCONNECTED,
            connected=self._credentials is not None, revision=self._revision,
            attempt_id=self._attempt_id if connecting else None,
            email=email if isinstance(email, str) else None,
            mode=ConnectedAccountLoginMode.BROWSER if connecting else None,
            authorization_url=self._authorization_url if connecting else None,
            expires_at=self._expires_at if connecting else None, message=message,
            supported_login_modes=(ConnectedAccountLoginMode.BROWSER,),
            default_login_mode=ConnectedAccountLoginMode.BROWSER,
        )

    async def start_login(self, mode: ConnectedAccountLoginMode) -> ConnectedAccountStatus:
        if mode is not ConnectedAccountLoginMode.BROWSER:
            raise GoogleOAuthError("Google OAuth supports browser sign-in only.")
        if not self._client_id or not self._client_secret:
            raise GoogleOAuthError("Configure Google OAuth Client ID and Client Secret in FCC provider settings first.")
        if not self._project_id:
            raise GoogleOAuthError("Configure Google Cloud Project ID (Gemini OAuth) in FCC provider settings first.")
        async with self._operation_lock:
            if self._closed:
                raise GoogleOAuthError("Google authentication manager is closed.")
            if self._login_task is not None and not self._login_task.done():
                return self.status()
            self._last_error = None
            self._attempt_id = f"google_login_{uuid.uuid4().hex}"
            browser = await _GoogleBrowserLogin.start(self._client_id)
            self._browser = browser
            self._authorization_url = browser.authorization_url
            self._expires_at = int(time.time()) + LOGIN_TIMEOUT_SECONDS
            self._login_task = asyncio.create_task(self._complete_login(browser), name="google-oauth-login")
            return self.status()

    async def _complete_login(self, browser: _GoogleBrowserLogin) -> None:
        try:
            grant = await browser.wait()
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(GOOGLE_TOKEN_URL, data={
                    "code": grant["code"], "client_id": self._client_id,
                    "client_secret": self._client_secret, "redirect_uri": grant["redirect_uri"],
                    "grant_type": "authorization_code", "code_verifier": browser.verifier,
                })
                response.raise_for_status()
                tokens = response.json()
                access_token = tokens.get("access_token")
                refresh_token = tokens.get("refresh_token") or (self._credentials or {}).get("refresh_token")
                if not isinstance(access_token, str) or not isinstance(refresh_token, str):
                    raise GoogleOAuthError("Google did not return renewable tokens. Revoke the previous grant and retry consent.")
                user_response = await client.get(GOOGLE_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
                user_response.raise_for_status()
                profile = user_response.json()
            saved: dict[str, object] = {
                "version": 1, "refresh_token": refresh_token, "access_token": access_token,
                "expires_at": int(time.time()) + int(tokens.get("expires_in", 3600)),
                "email": profile.get("email") if isinstance(profile, dict) else None,
            }
            await asyncio.to_thread(self._write_saved, saved)
            self._credentials = saved
            self._revision += 1
            self._last_error = None
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._last_error = f"Google sign-in failed ({type(exc).__name__}). Check the OAuth client and Vertex AI project configuration."
        finally:
            await browser.close()
            self._browser = None
            self._authorization_url = None
            self._attempt_id = None
            self._expires_at = None

    def _write_saved(self, payload: dict[str, object]) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self._path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload), encoding="utf-8")
        os.chmod(temporary, 0o600)
        temporary.replace(self._path)

    async def access_token(self) -> str:
        """Return a current Google access token, refreshing it when near expiry."""
        payload = self._credentials
        if payload is None:
            raise RuntimeError("Connect Google in FCC Admin before using Gemini OAuth.")
        expires_at = payload.get("expires_at")
        token = payload.get("access_token")
        if isinstance(token, str) and isinstance(expires_at, int) and expires_at > int(time.time()) + 300:
            return token
        refresh_token = payload.get("refresh_token")
        if not isinstance(refresh_token, str) or not self._client_id or not self._client_secret:
            raise RuntimeError("Google OAuth credentials need reconnecting; check the client ID and secret.")
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(GOOGLE_TOKEN_URL, data={
                "client_id": self._client_id,
                "client_secret": self._client_secret,
                "refresh_token": refresh_token,
                "grant_type": "refresh_token",
            })
            response.raise_for_status()
            result = response.json()
        new_token = result.get("access_token")
        if not isinstance(new_token, str) or not new_token:
            raise RuntimeError("Google token refresh returned no access token.")
        payload = {**payload, "access_token": new_token,
                   "expires_at": int(time.time()) + int(result.get("expires_in", 3600))}
        await asyncio.to_thread(self._write_saved, payload)
        self._credentials = payload
        return new_token

    async def cancel_login(self) -> ConnectedAccountStatus:
        async with self._operation_lock:
            task, browser = self._login_task, self._browser
            self._login_task = None
            self._browser = None
            self._authorization_url = None
            self._attempt_id = None
            self._expires_at = None
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            if browser is not None:
                await browser.close()
            self._last_error = None
            return self.status()

    async def disconnect(self) -> ConnectedAccountStatus:
        async with self._operation_lock:
            task, browser = self._login_task, self._browser
            self._login_task = None
            self._browser = None
            if task is not None and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError, Exception):
                    await task
            if browser is not None:
                await browser.close()
            payload = self._credentials
            if payload and isinstance(payload.get("refresh_token"), str):
                with suppress(httpx.HTTPError):
                    async with httpx.AsyncClient(timeout=10) as client:
                        await client.post(GOOGLE_REVOKE_URL, data={"token": payload["refresh_token"]})
            self._path.unlink(missing_ok=True)
            self._credentials = None
            self._last_error = None
            self._authorization_url = None
            self._attempt_id = None
            self._expires_at = None
            self._revision += 1
            return self.status()

    async def close(self) -> None:
        self._closed = True
        await self.cancel_login()
