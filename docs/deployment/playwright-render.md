# Playwright MCP on Render

The Claude Agent SDK starts `@playwright/mcp` directly inside the Node SDK worker when `FCC_PLAYWRIGHT_MCP=1`. It runs headless Chromium inside the same Render service, without a Termux bridge or external CDP connection. File and shell tools continue to run on the phone.

`scripts/render/build.sh` installs the pinned MCP package and Chromium, then checks MCP tool discovery, browser navigation, screenshots and accessibility snapshots. `PLAYWRIGHT_BROWSERS_PATH` must be identical at build and runtime (`/opt/render/project/src/.playwright-browsers`). A failed browser check blocks the deployment.

Each conversation uses `$HOME/.fcc/playwright/<session-id>` for its browser profile and artifacts. The Render persistent disk keeps profiles between turns and deployments. This browser has its own cookies and logins; it does not reuse the Termux or Cloud Shell browser session. The MCP uses the SDK in-process transport and opens no public browser port. Chromium runs headless with one process to reduce memory usage. The browser closes at turn completion; its profile stays on disk.

Validation: `node --test tests/runtime/claude_playwright.test.cjs tests/runtime/claude_device_tool_bridge.test.cjs tests/runtime/claude_agent_stream.test.cjs` and `python -m pytest tests/runtime/test_claude_agent_events.py -q -o addopts=''`. The deployment additionally runs the real Chromium check.
