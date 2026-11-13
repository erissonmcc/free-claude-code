#!/bin/sh
set -eu
pip install uv
uv sync --no-dev
npm install --no-save @anthropic-ai/claude-code @anthropic-ai/claude-agent-sdk @playwright/mcp@0.0.85
export PLAYWRIGHT_BROWSERS_PATH="${PLAYWRIGHT_BROWSERS_PATH:-/opt/render/project/src/.playwright-browsers}"
node node_modules/playwright/cli.js install chromium
node scripts/render/check-browser.cjs
