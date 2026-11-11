"""Shared local-only security boundary for Admin product surfaces."""

import ipaddress
import os
import secrets
from urllib.parse import urlsplit

from fastapi import HTTPException, Request


def _is_loopback_host(host: str | None) -> bool:
    if host is None:
        return False
    normalized = host.strip().strip("[]").lower()
    if normalized == "localhost":
        return True
    try:
        return ipaddress.ip_address(normalized).is_loopback
    except ValueError:
        return False


def _origin_is_local(origin: str | None) -> bool:
    if not origin:
        return True
    try:
        parsed = urlsplit(origin)
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.scheme in {"http", "https"}
        and parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_host(parsed.hostname)
    )


def _authority_is_local(authority: str | None) -> bool:
    if not authority:
        return False
    try:
        parsed = urlsplit(f"//{authority}")
        _ = parsed.port
    except ValueError:
        return False
    return (
        parsed.username is None
        and parsed.password is None
        and not parsed.path
        and not parsed.query
        and not parsed.fragment
        and _is_loopback_host(parsed.hostname)
    )


def require_loopback_admin(request: Request) -> None:
    """Allow local Admin access or authenticated access to a hosted instance."""

    client_host = request.client.host if request.client else None
    if _is_loopback_host(client_host) and _authority_is_local(
        request.headers.get("host")
    ) and _origin_is_local(request.headers.get("origin")):
        return

    expected = os.environ.get("FCC_ADMIN_AUTH_TOKEN", "").strip()
    authorization = request.headers.get("authorization", "")
    scheme, _, supplied = authorization.partition(" ")
    if (
        not expected
        or scheme.casefold() != "bearer"
        or not secrets.compare_digest(supplied.strip(), expected)
    ):
        raise HTTPException(status_code=401, detail="Admin authentication required")
