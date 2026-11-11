import asyncio
from pathlib import Path

import pytest

from free_claude_code.application.connected_accounts import (
    ConnectedAccountLoginMode,
    ConnectedAccountState,
)
from free_claude_code.config.provider_catalog import (
    PROVIDER_CATALOG,
    ProviderAuthKind,
)
from free_claude_code.providers.gemini.oauth import (
    GoogleOAuthAuthManager,
    GoogleOAuthError,
)


def test_gemini_oauth_is_a_connected_account_provider() -> None:
    descriptor = PROVIDER_CATALOG["gemini_oauth"]

    assert descriptor.auth_kind is ProviderAuthKind.CONNECTED_ACCOUNT
    assert descriptor.configuration_attrs() == ("gemini_oauth_project_id",)
    assert descriptor.default_base_url == PROVIDER_CATALOG["gemini"].default_base_url


def test_google_oauth_status_does_not_expose_credentials(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(
        "free_claude_code.providers.gemini.oauth.dotenv_values_from_file",
        lambda _path: {},
    )
    manager = GoogleOAuthAuthManager(credential_path=tmp_path / "google.json")

    status = manager.status().as_dict()

    assert status["state"] is ConnectedAccountState.DISCONNECTED
    assert status["connected"] is False
    assert "access_token" not in status
    assert "refresh_token" not in status
    assert "client_secret" not in status


def test_google_oauth_login_requires_configured_client(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GOOGLE_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.setattr(
        "free_claude_code.providers.gemini.oauth.dotenv_values_from_file",
        lambda _path: {},
    )
    manager = GoogleOAuthAuthManager(credential_path=tmp_path / "google.json")

    with pytest.raises(GoogleOAuthError, match="Client ID"):
        asyncio.run(manager.start_login(ConnectedAccountLoginMode.BROWSER))
