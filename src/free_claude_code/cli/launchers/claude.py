"""Installed Claude launcher: native command handling with an FCC connection."""

from collections.abc import Sequence
import json
import os
import shutil
import subprocess
import sys

from free_claude_code.harnesses.claude import CLAUDE_BINARY_NAME, build_claude_proxy_env
from free_claude_code.harnesses.launch import PreparedLaunch
from free_claude_code.harnesses.resources import LaunchResources

from .runner import HarnessSpec, LaunchContext, launch_harness


def _configure(
    ctx: LaunchContext, args: list[str], _files: LaunchResources
) -> PreparedLaunch:
    return PreparedLaunch(
        [ctx.binary_path, *args],
        build_claude_proxy_env(
            proxy_root_url=ctx.proxy_root_url,
            auth_token=ctx.auth_token,
            base_env=ctx.base_env,
        ),
    )


SPEC = HarnessSpec(
    binary_name=CLAUDE_BINARY_NAME,
    display_name="Claude Code",
    install_hint="Install Claude Code with: npm install -g @anthropic-ai/claude-code",
    configure=_configure,
)


def launch(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if os.environ.get("TERMUX_VERSION") and shutil.which("am"):
        _launch_android(args)
        return
    launch_harness(SPEC, args)


def launch_cli(argv: Sequence[str] | None = None) -> None:
    """Open the native Claude Code CLI even on Termux, bypassing the Android app."""
    launch_harness(SPEC, argv)


def _launch_android(args: list[str]) -> None:
    package = os.environ.get("FCC_ANDROID_PACKAGE", "com.fcc.claude").strip()
    if not package or any(character.isspace() for character in package):
        raise SystemExit("FCC_ANDROID_PACKAGE must be a valid Android package name.")
    command = [
        "am",
        "start",
        "--activity-single-top",
        "-n",
        f"{package}/.MainActivity",
        "--es",
        "fcc_cwd",
        os.getcwd(),
        "--es",
        "fcc_args_json",
        json.dumps(args, ensure_ascii=False, separators=(",", ":")),
    ]
    # Android parses the original arguments once. A second prompt parser here
    # would also consume option values (for example --model sonnet) as text.
    try:
        result = subprocess.run(command, check=False, text=True, capture_output=True)
    except OSError as exc:
        raise SystemExit(f"Could not open FCC Claude Android app: {exc}") from None
    if result.returncode:
        detail = (result.stderr or result.stdout).strip()
        raise SystemExit(
            "Could not open FCC Claude Android app. Install the app package "
            f"{package} first. {detail}"
        )
