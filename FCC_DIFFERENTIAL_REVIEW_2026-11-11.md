# FCC Admin Login — Differential Security Review

## Executive Summary

| Severity | Count |
|---|---:|
| Critical | 0 |
| High | 0 |
| Medium | 0 |
| Low / informational | 1 |

**Overall risk:** Medium, because this changes authentication for public Admin access.
**Recommendation:** Approve, with the operational note below.

## Scope and Baseline

- Baseline: backend `HEAD` at `8ac5a46f`; prior remote Admin Bearer gate was added in `32237585` (`feat: add context management and remote admin auth`).
- Reviewed: `admin_security.py`, `admin_routes.py`, `admin_static/login.html`, `tests/api/test_admin_login.py`, changed Admin assertions in `tests/api/test_admin.py`, and the Render model environment setting.
- The existing Render admin token is generated from 32 random bytes (64 hexadecimal characters), is compared with `secrets.compare_digest`, and was not included in source or this report.

## What Changed

- Unauthenticated `GET /admin` now returns a mobile login page. `/admin/login` serves the same page directly.
- `POST /admin/api/auth/login` validates the configured secret and sets `fcc_admin_session` with `HttpOnly`, `Secure`, `SameSite=Strict`, `Path=/admin`, and a 12-hour lifetime.
- Existing Bearer access and loopback access remain available. Admin APIs and assets still use the shared authorization guard.
- Logout removes the cookie. The UI does not save the token to localStorage or put it in the URL.
- Removed `MODEL` from the Render environment and Blueprint so the Admin can change the default model instead of treating it as a locked process value.

## Findings

### Informational: session cookie is a bearer credential

**Files:** `src/free_claude_code/api/admin_routes.py:140`, `src/free_claude_code/api/admin_security.py:63`
**Blast radius:** One shared Admin guard used by roughly 37 Admin route handlers, plus the code-session router dependency.
**Test coverage:** Yes for cookie creation, required attributes, guarded API access, Bearer compatibility, invalid login, and logout.

The cookie contains the same high-entropy value as `FCC_ADMIN_AUTH_TOKEN`, so a stolen cookie grants Admin access until it expires or the environment token is rotated. `HttpOnly`, `Secure`, `SameSite=Strict`, the `/admin` path, and the 12-hour expiry reduce exposure. This is the expected stateless tradeoff; no blocking issue found.

## Adversarial Analysis

**Attacker:** Unauthenticated internet client.
**Entry points:** Public `GET /admin`, `/admin/login`, and `POST /admin/api/auth/login`.
**Attack considered:** Guess the Admin token, plant an auth cookie from another site, or reach Admin APIs without credentials.

- A guessed token must match a 256-bit random value; invalid attempts return 401 and do not set a cookie.
- Cross-site requests do not receive the `SameSite=Strict` cookie. The login endpoint requires the secret in a JSON body; the token is not placed in a URL.
- `/admin` returns only the login page before authentication. Static Admin assets and APIs continue to reject unauthenticated remote requests.
- No privilege bypass or regression of the prior Bearer guard was found.

## Verification

- Focused auth/Admin checks: **23 passed**.
- `render blueprints validate render.yaml --output json`: valid.
- `git diff --check`: passed.
- Full pytest collection is blocked by the Termux Python 3.14.7 ARM64 environment: four existing modules fail to import `cryptography` with `cannot locate symbol PyBaseObject_Type`; this occurs before their tests execute. The targeted Admin tests run successfully with `-o addopts=''` because the configured xdist plugin is unavailable here.
- No browser interaction or on-device login has been verified yet; live verification remains required after Render deploy.

## History and Coverage

- Git blame confirms the remote Bearer check came from `32237585`; the change retains that path and adds the cookie as an alternative.
- New authentication behavior is covered by five focused route tests. Existing Admin page tests were updated to assert public login rendering while assets and APIs remain protected.

## Recommendation

Deploy, then verify the public `/admin` login over HTTPS and confirm that a valid cookie loads the Admin shell while invalid credentials stay on the login page. After the user adds an OpenAI credential in Admin, select an OpenAI model there; the current service has NVIDIA NIM configured and no OpenAI account credential.

## Limitations

This is a focused review of the authentication delta, not a complete review of the existing Admin application, its third-party scripts, or the Render account configuration.
