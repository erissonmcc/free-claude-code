# Render Admin Mobile Login Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Open the hosted FCC Admin on a phone through a public Render link and authenticate with a secure browser session.

**Architecture:** A public login page posts the admin token to a narrow login endpoint. On success, the server sets a 12-hour HttpOnly, Secure, SameSite Strict cookie; existing API Bearer authentication remains unchanged. Unauthenticated Admin pages render the login screen.

**Tech Stack:** FastAPI, Python, static HTML, pytest.

**Spec:** `docs/superpowers/specs/2026-11-11-render-admin-login-design.md`

## Global Constraints

- Keep `FCC_ADMIN_AUTH_TOKEN` comparison constant-time.
- Never put the token in a URL or browser persistent storage.
- Preserve local loopback access and Android Bearer authentication.

## Review Focus

- Wrong or missing login token must not create a session.
- Cookie must be HttpOnly, Secure, SameSite Strict, scoped to `/admin`, and expire after 12 hours.
- An unauthenticated browser opening `/admin` must receive the login page.
- A valid cookie and an existing Bearer token must both retain access to guarded APIs.
- Logout must clear the session cookie.

---

### Task 1: Login and cookie authentication

**Files:**
- Modify: `src/free_claude_code/api/admin_security.py`
- Modify: `src/free_claude_code/api/admin_routes.py`
- Test: `tests/api/test_admin_login.py`

**Interfaces:**
- Produce `is_admin_authenticated(request: Request) -> bool` and a login POST that sets the session cookie.
- Existing `require_loopback_admin(request)` remains the guard used by Admin APIs.

- [ ] Write tests for invalid token, valid token cookie, cookie attributes, bearer compatibility, and logout.
- [ ] Run the focused test and confirm the expected failures.
- [ ] Implement cookie-aware authentication and login/logout routes.
- [ ] Run the focused test and the existing API auth test.

### Task 2: Mobile login page

**Files:**
- Create: `src/free_claude_code/api/admin_static/login.html`
- Modify: `src/free_claude_code/api/admin_routes.py`
- Test: `tests/api/test_admin_login.py`

**Interfaces:**
- `/admin` displays the Admin page with an authenticated session and the login page otherwise.
- `/admin/login` serves the same login page; successful login navigates to `/admin`.

- [ ] Add a route test asserting unauthenticated `/admin` includes the login form and authenticated `/admin` includes the Admin shell.
- [ ] Confirm the new route test fails before implementation.
- [ ] Implement a responsive Portuguese login page and redirect after successful login.
- [ ] Run focused auth tests and `git diff --check`.

### Task 3: Review and deployment

**Files:**
- Review: auth and route diffs.

- [ ] Inspect recent auth history and all callers of the guard.
- [ ] Validate backend tests and the Render Blueprint.
- [ ] Commit and push the backend update; verify Render deploy and public login response.
