"""Native Copilot CLI checks without importing the Copilot SDK."""

import asyncio
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path
from contextlib import suppress

from .lifecycle import drain_owned
from .types import CopilotUnavailable

COPILOT_CLI_VERSION = "1.0.83"


_TOKEN_ENV = frozenset(
    {
        "COPILOT_GITHUB_TOKEN",
        "GH_TOKEN",
        "GITHUB_TOKEN",
        "GITHUB_COPILOT_API_TOKEN",
        "COPILOT_API_URL",
        "COPILOT_SDK_AUTH_TOKEN",
        "COPILOT_DISABLE_KEYTAR",
        "COPILOT_OFFLINE",
    }
)


def profile_environment(source: Mapping[str, str] | None = None) -> dict[str, str]:
    """Keep native profile selection while excluding token and BYOK overrides."""
    return {
        key: value
        for key, value in (os.environ if source is None else source).items()
        if key.upper() not in _TOKEN_ENV
        and not key.upper().startswith("COPILOT_PROVIDER_")
    }


async def verified_cli_path(env: Mapping[str, str]) -> str:
    # Termux/Android can override the CLI launcher (for example, a proot-distro
    # shim) without changing the pinned Copilot SDK contract.
    configured = env.get("FCC_COPILOT_CLI", "").strip()
    if not configured and sys.platform == "android":
        termux_launcher = Path(__file__).with_name("termux_cli.sh")
        if termux_launcher.is_file() and os.access(termux_launcher, os.X_OK):
            configured = str(termux_launcher)
    path = configured or shutil.which("copilot", path=env.get("PATH"))
    if path and configured and not os.path.isabs(path):
        path = shutil.which(path, path=env.get("PATH"))
    if path is None:
        raise CopilotUnavailable(
            f"Install GitHub Copilot CLI {COPILOT_CLI_VERSION}, then connect again."
        )
    process = await asyncio.create_subprocess_exec(
        path,
        "--version",
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0,
    )
    try:
        async with asyncio.timeout(15):
            stdout, _ = await process.communicate()
    except TimeoutError:
        raise CopilotUnavailable(
            "Copilot CLI did not report its version in time. Check the CLI installation and reconnect."
        ) from None
    finally:
        if process.returncode is None:
            await drain_owned(asyncio.create_task(_close_cli_probe(process)))
    if (
        process.returncode != 0
        or re.match(
            rf"GitHub Copilot CLI {re.escape(COPILOT_CLI_VERSION)}\.?(?:\r?\n|\Z)",
            stdout.decode("utf-8", errors="replace"),
        )
        is None
    ):
        if sys.platform == "android":
            raise CopilotUnavailable(
                f"GitHub Copilot CLI {COPILOT_CLI_VERSION} could not start with the Termux compatibility launcher. "
                "Check the Termux Copilot installation and reconnect."
            )
        raise CopilotUnavailable(
            f"FCC requires GitHub Copilot CLI {COPILOT_CLI_VERSION} for its pinned SDK. Install that version and reconnect."
        )
    return path


async def _close_cli_probe(process: asyncio.subprocess.Process) -> None:
    with suppress(ProcessLookupError):
        process.kill()
    await process.communicate()
