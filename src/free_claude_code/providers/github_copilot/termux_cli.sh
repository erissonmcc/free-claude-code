#!/data/data/com.termux/files/usr/bin/bash
set -eu

VERSION="1.0.83"
COPILOT_HOME="${HOME}/.copilot-termux"
COPILOT_DIR="${COPILOT_HOME}/${VERSION}"
PREFIX="${PREFIX:-/data/data/com.termux/files/usr}"
PACKAGE_DIR="${PREFIX}/lib/node_modules/@bash0816/copilot-termux"
NODE="${PREFIX}/bin/node"
PATCH="${PACKAGE_DIR}/lib/platform-patch.js"
COMPAT="${PACKAGE_DIR}/lib/bionic-compat.so"
INDEX="${COPILOT_DIR}/index.js"

if [ ! -x "$NODE" ]; then
  echo "FCC Copilot: Termux Node.js not found at $NODE" >&2
  exit 1
fi
if [ ! -f "$PATCH" ] || [ ! -f "$COMPAT" ]; then
  echo "FCC Copilot: @bash0816/copilot-termux compatibility files are missing" >&2
  exit 1
fi
if [ ! -f "$INDEX" ]; then
  echo "FCC Copilot: Copilot CLI $VERSION is not installed; run copilot-termux setup" >&2
  exit 1
fi

# 1.0.83 ships a linux-musl ARM64 runtime that can be loaded by Termux Node
# with the bionic compatibility layer. Android's glibc loader path is not used.
RUNTIME="$COPILOT_DIR/prebuilds/linux-arm64/runtime.node"
MUSL_RUNTIME="$COPILOT_DIR/prebuilds/linuxmusl-arm64/runtime.node"
CLI_NATIVE="$COPILOT_DIR/prebuilds/linux-arm64/cli-native.node"
MUSL_CLI_NATIVE="$COPILOT_DIR/prebuilds/linuxmusl-arm64/cli-native.node"

if [ -f "$MUSL_RUNTIME" ]; then
  if [ ! -f "$RUNTIME.fcc-original" ] && [ -f "$RUNTIME" ]; then
    cp -f "$RUNTIME" "$RUNTIME.fcc-original"
  fi
  cp -f "$MUSL_RUNTIME" "$RUNTIME"
fi
if [ -f "$MUSL_CLI_NATIVE" ]; then
  if [ ! -f "$CLI_NATIVE.fcc-original" ] && [ -f "$CLI_NATIVE" ]; then
    cp -f "$CLI_NATIVE" "$CLI_NATIVE.fcc-original"
  fi
  cp -f "$MUSL_CLI_NATIVE" "$CLI_NATIVE"
fi

unset LD_LIBRARY_PATH
export LD_PRELOAD="$COMPAT"
unset COPILOT_TERMUX_GLIBC_MODE

# Reuse the OAuth token obtained by the Termux device-code flow.
TOKEN_FILE="$COPILOT_HOME/oauth-token"
if [ -f "$TOKEN_FILE" ]; then
  COPILOT_TOKEN="$(cat "$TOKEN_FILE")"
  if [ -n "$COPILOT_TOKEN" ]; then
    export COPILOT_GITHUB_TOKEN="$COPILOT_TOKEN"
    export COPILOT_SDK_AUTH_TOKEN="$COPILOT_TOKEN"
    export GH_TOKEN="$COPILOT_TOKEN"
    export GITHUB_TOKEN="$COPILOT_TOKEN"
    export GITHUB_COPILOT_GITHUB_TOKEN="$COPILOT_TOKEN"
    export GITHUB_COPILOT_AGENT_GITHUB_TOKEN="$COPILOT_TOKEN"
  fi
fi

# Android/Termux: the native OAuth device-code helper inside Copilot 1.0.83
# fails before reaching GitHub. Reimplement only that transport here, then hand
# the resulting OAuth token back to Copilot's official --with-token path so the
# CLI remains the owner of credential storage and refresh behavior.
if [ "${1:-}" = "login" ] && [ "${2:-}" = "--device-code" ]; then
  if [ -n "${COPILOT_TOKEN:-}" ]; then
    echo "Signed in successfully." >&2
    exit 0
  fi
  export FCC_COPILOT_CLIENT_ID="Iv1.b507a08c87ecfe98"
  TOKEN="$($NODE - <<'NODE'
const clientId = process.env.FCC_COPILOT_CLIENT_ID;
const sleep = ms => new Promise(r => setTimeout(r, ms));
async function post(url, fields) {
  const r = await fetch(url, {
    method: 'POST',
    headers: {
      'Accept': 'application/json',
      'Content-Type': 'application/x-www-form-urlencoded',
      'User-Agent': 'GitHub-Copilot-CLI/1.0.83'
    },
    body: new URLSearchParams(fields)
  });
  const text = await r.text();
  let data;
  try { data = JSON.parse(text); } catch { throw new Error(`GitHub returned HTTP ${r.status}`); }
  if (!r.ok && !data.error) throw new Error(`GitHub returned HTTP ${r.status}`);
  return data;
}
(async () => {
  const d = await post('https://github.com/login/device/code', {
    client_id: clientId,
    scope: 'read:user read:org repo gist codespace'
  });
  if (!d.device_code || !d.user_code || !d.verification_uri) throw new Error(d.error_description || d.error || 'GitHub did not return a device code');
  console.error(`To authenticate, visit ${d.verification_uri} and enter code ${d.user_code}`);
  console.error('Waiting for authorization...');
  let interval = Math.max(Number(d.interval) || 5, 5) * 1000;
  const deadline = Date.now() + (Number(d.expires_in) || 900) * 1000;
  while (Date.now() < deadline) {
    await sleep(interval);
    const t = await post('https://github.com/login/oauth/access_token', {
      client_id: clientId,
      device_code: d.device_code,
      grant_type: 'urn:ietf:params:oauth:grant-type:device_code'
    });
    if (t.access_token) {
      process.stdout.write(t.access_token);
      return;
    }
    if (t.error === 'authorization_pending') continue;
    if (t.error === 'slow_down') { interval += 5000; continue; }
    throw new Error(t.error_description || t.error || 'GitHub device authorization failed');
  }
  throw new Error('GitHub device code expired');
})().catch(e => { console.error(`Copilot device login failed: ${e.message}`); process.exit(1); });
NODE
  )"
  status=$?
  if [ "$status" -ne 0 ] || [ -z "$TOKEN" ]; then
    exit "${status:-1}"
  fi
  mkdir -p "$COPILOT_HOME"
umask 077
printf '%s' "$TOKEN" > "$COPILOT_HOME/oauth-token"
export COPILOT_GITHUB_TOKEN="$TOKEN"
export COPILOT_SDK_AUTH_TOKEN="$TOKEN"
export GITHUB_COPILOT_GITHUB_TOKEN="$TOKEN"
export GITHUB_COPILOT_AGENT_GITHUB_TOKEN="$TOKEN"
echo "Signed in successfully." >&2
exit 0
fi

exec "$NODE" --require "$PATCH" "$INDEX" "$@"
