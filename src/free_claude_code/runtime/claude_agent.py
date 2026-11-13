"""Claude Agent SDK adapter for the shared FCC coding-session service."""

import asyncio
import json
import os
import shutil
import sys
from pathlib import Path
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from free_claude_code.application.code_sessions.models import (
    CodeCatalog,
    CodeConflictError,
    CodeMode,
    CodeModel,
    CodeUnavailableError,
    CodeValidationError,
    HarnessEvent,
    NativeThread,
    PromptRequest,
    ItemUpdate,
)
from free_claude_code.application.code_sessions.ports import EventSink, HarnessSelection
from free_claude_code.application.model_catalog import CatalogModel, read_model_catalog
from free_claude_code.application.ports import RequestRuntimePort
from free_claude_code.config.model_refs import split_provider_model_ref
from free_claude_code.config.server_urls import local_proxy_root_url
from free_claude_code.core.json_types import JsonObject
from free_claude_code.harnesses.claude import build_claude_proxy_env


_MESSAGE_ACK_TIMEOUT = 10.0
_DEVICE_TOOL_RESULT_LIMIT = 512 * 1024


async def _response_lines(stream):
    buffer = b""
    while chunk := await stream.read(64 * 1024):
        buffer += chunk
        while b"\n" in buffer:
            line, buffer = buffer.split(b"\n", 1)
            if len(line) > 16 * 1024 * 1024:
                raise ValueError("Claude Agent SDK response exceeds the size limit")
            yield line
        if len(buffer) > 16 * 1024 * 1024:
            raise ValueError("Claude Agent SDK response exceeds the size limit")
    if buffer:
        yield buffer


def _fast_prompt(text: str) -> tuple[str, bool]:
    if text.startswith("/fast ") and text[6:].strip():
        return text[6:], True
    return text, False


def _node_sdk_available() -> bool:
    if not shutil.which("node"):
        return False
    candidates = [
        os.environ.get("FCC_CLAUDE_AGENT_SDK_PATH", ""),
        os.path.join(os.environ.get("PREFIX", ""), "lib", "node_modules", "@anthropic-ai", "claude-agent-sdk"),
        os.path.join(os.getcwd(), "node_modules", "@anthropic-ai", "claude-agent-sdk"),
        str(Path(__file__).resolve().parents[3] / "node_modules" / "@anthropic-ai" / "claude-agent-sdk"),
    ]
    return any(path and os.path.isdir(path) for path in candidates)


def _model_option(model: CatalogModel) -> CodeModel:
    provider_id, model_name = split_provider_model_ref(model.provider_model_ref)
    return CodeModel(
        id=model.provider_model_ref,
        display_name=model.display_name,
        provider_id=provider_id,
        model_name=model_name,
        reasoning_efforts=("low", "medium", "high", "max")
        if model.supports_reasoning is not False
        else ("off",),
        default_reasoning_effort="medium"
        if model.supports_reasoning is not False
        else "off",
        context_window_tokens=model.context_window_tokens,
    )


def _sdk_permission_mode(mode: CodeMode) -> str:
    return {
        "config": "default",
        "ask": "default",
        "auto_review": "auto",
        "full_access": "bypassPermissions",
    }[mode]


@dataclass(frozen=True, slots=True)
class _ClaudeSelection:
    runtime: RequestRuntimePort = field(repr=False)
    binary_path: str
    proxy_root_url: str
    auth_token: str = field(repr=False)
    base_env: Mapping[str, str] = field(repr=False)
    model: str
    reasoning_effort: str | None
    mode: CodeMode
    configuration_key: str

    async def open(self, cwd: str, sink: EventSink) -> "ClaudeAgentConnection":
        snapshot = await self.runtime.wait_for_catalog()
        model = next(
            (
                entry
                for entry in read_model_catalog(snapshot, snapshot.settings).models
                if entry.provider_model_ref == self.model
            ),
            None,
        )
        if model is None:
            raise CodeValidationError("This model is no longer available. Choose another.")
        env = build_claude_proxy_env(
            proxy_root_url=self.proxy_root_url,
            auth_token=self.auth_token,
            base_env=self.base_env,
        )
        selected_model = (
            model.wire_slug
            if model.supports_reasoning is False
            else model.provider_model_ref
        )
        return ClaudeAgentConnection(self, cwd, sink, env, selected_model)


class ClaudeAgentHarnessFactory:
    def __init__(
        self,
        runtime: RequestRuntimePort,
        *,
        binary: str | None = None,
        env: Mapping[str, str] | None = None,
        auth_token: str | None = None,
    ) -> None:
        self._runtime = runtime
        self._binary = binary
        self._env = env
        self._auth_token = auth_token

    def availability(self) -> tuple[bool, str | None]:
        if not (self._binary or shutil.which("claude")):
            return False, "Claude Code is not installed in Termux. Install it with npm."
        if not _node_sdk_available():
            return False, "Claude Agent SDK for Node.js is missing. Install @anthropic-ai/claude-agent-sdk in Termux."
        return True, None

    def catalog(self) -> CodeCatalog:
        settings = self._runtime.current_settings()
        catalog = read_model_catalog(self._runtime, settings)
        return CodeCatalog(
            settings.model,
            tuple(_model_option(model) for model in catalog.models),
        )

    async def prepare(
        self, model: str, reasoning_effort: str | None, mode: CodeMode
    ) -> _ClaudeSelection:
        binary = self._binary or shutil.which("claude")
        if binary is None:
            raise CodeUnavailableError("Claude Code is not installed in Termux.")
        if not _node_sdk_available():
            raise CodeUnavailableError("Claude Agent SDK for Node.js is missing in Termux.")
        snapshot = await self._runtime.wait_for_catalog()
        settings = snapshot.settings
        catalog = read_model_catalog(snapshot, settings)
        selected = next(
            (entry for entry in catalog.models if entry.provider_model_ref == model), None
        )
        if selected is None:
            raise CodeValidationError("This model is unavailable. Choose another model.")
        option = _model_option(selected)
        if reasoning_effort is not None and reasoning_effort not in option.reasoning_efforts:
            raise CodeValidationError("This effort is unavailable. Choose another effort.")
        return _ClaudeSelection(
            runtime=self._runtime,
            binary_path=binary,
            proxy_root_url=local_proxy_root_url(settings),
            auth_token=(self._auth_token or settings.proxy_auth_token).strip(),
            base_env=dict(self._env if self._env is not None else os.environ),
            model=model,
            reasoning_effort=reasoning_effort or option.default_reasoning_effort,
            mode=mode,
            configuration_key=f"{model}:{reasoning_effort}:{mode}",
        )

    async def open_history(self, cwd: str, sink: EventSink) -> "ClaudeAgentConnection":
        selection = await self.prepare(
            self.catalog().default_model, None, "config"
        )
        return await selection.open(cwd, sink)


class ClaudeAgentConnection:
    def __init__(
        self,
        selection: _ClaudeSelection,
        cwd: str,
        sink: EventSink,
        env: dict[str, str],
        model: str,
    ) -> None:
        self.generation = str(uuid.uuid4())
        self.thread_id: str | None = None
        self._selection, self._cwd, self._sink, self._env = selection, cwd, sink, env
        self._model = model
        self._process: asyncio.subprocess.Process | None = None
        self._turn_id: str | None = None
        self._reader: asyncio.Task[None] | None = None
        self._message_acks: dict[str, asyncio.Future[bool]] = {}
        self._permission_tasks: set[asyncio.Task[None]] = set()
        self._device_tool_tasks: set[asyncio.Task[None]] = set()
        self._permissions: dict[str | int, asyncio.Future[JsonObject]] = {}
        self._permission_inputs: dict[str, tuple[str, dict[str, Any]]] = {}
        self._device_tools: dict[str, dict[str, Any]] = {}
        self._closing = False
        self._new_thread = False

    def supports(self, selection: HarnessSelection) -> bool:
        return not self._closing and self._selection.configuration_key == selection.configuration_key

    async def create_thread(self) -> NativeThread:
        self.thread_id = str(uuid.uuid4())
        self._new_thread = True
        return NativeThread(self.thread_id, permission_defaults={})

    async def resume_thread(self, thread_id: str) -> NativeThread:
        self.thread_id = thread_id
        self._new_thread = False
        return NativeThread(thread_id, permission_defaults={})

    async def fork_thread(self, thread_id: str, item_id: str, title: str, *, prompt: str | None = None) -> NativeThread | None:
        worker = str(Path(__file__).with_name("claude_agent_worker.cjs"))
        process = await asyncio.create_subprocess_exec(
            shutil.which("node") or "node", worker, cwd=self._cwd, env=self._env,
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        try:
            output, error = await asyncio.wait_for(process.communicate(json.dumps({
                "type": "fork", "sessionId": thread_id, "itemId": item_id,
                "cwd": self._cwd, "title": title, "rewind": prompt is not None, "prompt": prompt,
            }).encode() + b"\n"), timeout=30)
            messages = [json.loads(line) for line in output.splitlines() if line.strip()]
            result = next((message for message in messages if message.get("type") == "fork"), None)
            if process.returncode or result is None:
                detail = next((message.get("error") for message in messages if message.get("type") == "error"), "SDK não retornou a ramificação.")
                raise CodeUnavailableError(str(detail))
            if prompt is not None and result["sessionId"] is None:
                return None
            return NativeThread(str(uuid.UUID(result["sessionId"])), permission_defaults={})
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    async def read_thread(self, thread_id: str) -> NativeThread:
        return NativeThread(thread_id)

    async def start_turn(
        self,
        text: str,
        selection: HarnessSelection,
        client_id: str,
        permission_defaults: JsonObject,
    ) -> str:
        if not self.thread_id or (self._process is not None and self._process.returncode is None):
            raise CodeUnavailableError("Claude session is not connected.")
        self._turn_id = str(uuid.uuid4())
        node = shutil.which("node")
        if node is None:
            raise CodeUnavailableError("Node.js is not installed in Termux.")
        worker = os.path.join(os.path.dirname(__file__), "claude_agent_worker.cjs")
        self._process = await asyncio.create_subprocess_exec(
            node, worker,
            cwd=self._cwd,
            env=self._env,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        prompt, fast = _fast_prompt(text)
        request = {
            "cwd": self._cwd,
            "fast": fast,
            "model": self._model,
            "effort": self._selection.reasoning_effort if self._selection.reasoning_effort != "off" else None,
            "permissionMode": _sdk_permission_mode(self._selection.mode),
            "env": {key: value for key, value in self._env.items() if key != "FCC_GOOGLE_SHELL_CREDENTIALS"},
            "python": sys.executable,
            "binary": self._selection.binary_path,
            "sessionId": self.thread_id,
            "resume": not self._new_thread,
            "prompt": prompt,
        }
        assert self._process.stdin is not None
        self._process.stdin.write((json.dumps(request, ensure_ascii=False) + "\n").encode())
        await self._process.stdin.drain()
        self._new_thread = False
        await self._sink(
            HarnessEvent(
                generation=self.generation,
                thread_id=self.thread_id,
                kind="turn_started",
                turn_id=self._turn_id,
                status="running",
            )
        )
        self._reader = asyncio.create_task(self._read_response(self._turn_id))
        return self._turn_id

    async def send_message(self, operation_id: str, text: str) -> None:
        process = self._process
        if (self._closing or process is None or process.returncode is not None
                or process.stdin is None or process.stdin.is_closing()):
            raise CodeUnavailableError("Claude worker is not available.")
        if operation_id in self._message_acks:
            raise CodeConflictError("This message is already pending.")
        future = asyncio.get_running_loop().create_future()
        self._message_acks[operation_id] = future
        try:
            async with asyncio.timeout(_MESSAGE_ACK_TIMEOUT):
                process.stdin.write((json.dumps({"type": "user_message", "operation_id": operation_id,
                                                "text": text}, ensure_ascii=False) + "\n").encode())
                drain = asyncio.create_task(process.stdin.drain())
                try:
                    done, _ = await asyncio.wait((drain, future), return_when=asyncio.FIRST_COMPLETED)
                    if drain in done:
                        await drain
                    if not await future:
                        raise CodeUnavailableError("Claude worker closed before acknowledging the message.")
                finally:
                    if not drain.done():
                        drain.cancel()
                    await asyncio.gather(drain, return_exceptions=True)
        finally:
            self._message_acks.pop(operation_id, None)

    def _fail_message_acks(self) -> None:
        for future in self._message_acks.values():
            if not future.done():
                future.set_result(False)

    async def _read_response(self, turn_id: str) -> None:
        terminal_message: dict[str, Any] | None = None
        try:
            process = self._process
            assert process is not None and process.stdout is not None
            async for line in _response_lines(process.stdout):
                message = json.loads(line)
                if message.get("type") == "message_ack":
                    operation_id = message.get("operation_id")
                    future = self._message_acks.get(operation_id) if isinstance(operation_id, str) else None
                    if future is not None and not future.done():
                        future.set_result(True)
                elif message.get("type") == "usage":
                    tokens = message.get("tokens")
                    if isinstance(tokens, int) and not isinstance(tokens, bool) and tokens >= 0:
                        await self._sink(HarnessEvent(generation=self.generation, thread_id=self.thread_id,
                            kind="run_usage", turn_id=turn_id, consumed_tokens=tokens))
                elif message.get("type") == "text":
                    await self._emit_item(turn_id, "text", item_id=message.get("id"),
                        text=message.get("text", ""), complete=message.get("complete", True))
                elif message.get("type") == "tool_call":
                    data = message.get("input", {})
                    await self._emit_item(turn_id, "tool_call", item_id=message.get("id"),
                        title=message.get("name", "tool"), text=json.dumps(data, ensure_ascii=False), raw={"input": data}, complete=True)
                elif message.get("type") == "tool_result":
                    result = message.get("content", "")
                    if isinstance(result, list):
                        result = [
                            {"type": "image"} if isinstance(block, dict) and block.get("type") == "image" else block
                            for block in result
                        ]
                    tool_id = message.get("id")
                    await self._emit_item(turn_id, "tool_result", item_id=f"result:{tool_id}" if tool_id else None,
                        text=result if isinstance(result, str) else json.dumps(result, ensure_ascii=False),
                        raw={"is_error": message.get("error", False)}, complete=True)
                elif message.get("type") == "permission":
                    task = asyncio.create_task(self._emit_permission(turn_id, message))
                    self._permission_tasks.add(task)
                    task.add_done_callback(self._permission_done)
                elif message.get("type") == "device_tool":
                    task = asyncio.create_task(self._emit_device_tool(turn_id, message))
                    self._device_tool_tasks.add(task)
                    task.add_done_callback(self._device_tool_done)
                elif message.get("type") == "result":
                    terminal_message = message
                elif message.get("type") == "error":
                    terminal_message = message
            self._fail_message_acks()
            await process.wait()
            if terminal_message and terminal_message.get("type") == "result":
                await self._sink(
                    HarnessEvent(
                        generation=self.generation,
                        thread_id=self.thread_id,
                        kind="turn_completed",
                        turn_id=turn_id,
                        status="completed" if terminal_message.get("subtype") == "success" else "failed",
                        message=terminal_message.get("result") if terminal_message.get("subtype") != "success" else None,
                        context_used_tokens=(terminal_message.get("usage") or {}).get("input_tokens"),
                        raw={"subtype": terminal_message.get("subtype"), "cost_usd": terminal_message.get("cost")},
                    )
                )
            elif terminal_message and terminal_message.get("type") == "error":
                await self._sink(HarnessEvent(generation=self.generation, thread_id=self.thread_id,
                    kind="turn_completed", turn_id=turn_id, status="failed",
                    message=terminal_message.get("message", "Claude Agent SDK failed.")))
            else:
                await self._sink(HarnessEvent(generation=self.generation, thread_id=self.thread_id,
                    kind="turn_completed", turn_id=turn_id, status="failed",
                    message="Claude Agent SDK worker exited before completing the turn."))
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            process = self._process
            if process is not None and process.returncode is None:
                process.terminate()
                await process.wait()
            await self._sink(
                HarnessEvent(
                    generation=self.generation,
                    thread_id=self.thread_id,
                    kind="turn_completed",
                    turn_id=turn_id,
                    status="failed",
                    message=f"Claude Agent SDK failed: {type(exc).__name__}",
                )
            )
        finally:
            self._fail_message_acks()
            for task in tuple(self._permission_tasks):
                task.cancel()
            if self._permission_tasks:
                await asyncio.gather(*tuple(self._permission_tasks), return_exceptions=True)
            for task in tuple(self._device_tool_tasks):
                task.cancel()
            if self._device_tool_tasks:
                await asyncio.gather(*tuple(self._device_tool_tasks), return_exceptions=True)

    def _permission_done(self, task: asyncio.Task[None]) -> None:
        self._permission_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            process = self._process
            if process is not None and process.returncode is None:
                process.terminate()

    async def _emit_permission(self, turn_id: str, message: dict[str, Any]) -> None:
        request_id = str(uuid.uuid4())
        loop = asyncio.get_running_loop()
        future: asyncio.Future[JsonObject] = loop.create_future()
        self._permissions[request_id] = future
        tool = message.get("tool", "tool")
        tool_input = message.get("input", {})
        self._permission_inputs[request_id] = (tool, tool_input)
        await self._sink(HarnessEvent(generation=self.generation, thread_id=self.thread_id, kind="prompt",
            turn_id=turn_id, prompt=PromptRequest(request_id=request_id, kind="approval",
                form={"tool": message.get("tool", "tool"), "input": message.get("input", {})},
                raw=message, item_id=request_id)))
        try:
            response = await future
            process = self._process
            if process and process.stdin:
                updated_input = dict(tool_input)
                if tool == "AskUserQuestion" and response.get("approved") is True:
                    updated_input["answers"] = response["answers"]
                wire = {"type": "permission_response", "id": message["id"],
                    "approved": response.get("approved") is True,
                    "input": updated_input, "message": response.get("message")}
                process.stdin.write((json.dumps(wire, ensure_ascii=False) + "\n").encode())
                await process.stdin.drain()
        finally:
            self._permissions.pop(request_id, None)
            self._permission_inputs.pop(request_id, None)

    def _device_tool_done(self, task: asyncio.Task[None]) -> None:
        self._device_tool_tasks.discard(task)
        if not task.cancelled() and task.exception() is not None:
            process = self._process
            if process is not None and process.returncode is None:
                process.terminate()

    async def _emit_device_tool(self, turn_id: str, message: dict[str, Any]) -> None:
        worker_id = message.get("id")
        name = message.get("name")
        tool_input = message.get("input", {})
        if not isinstance(worker_id, str) or not worker_id or not isinstance(name, str) or not isinstance(tool_input, dict):
            raise CodeValidationError("Claude sent an invalid phone tool request.")
        request_id = str(uuid.uuid4())
        future: asyncio.Future[JsonObject] = asyncio.get_running_loop().create_future()
        self._device_tools[request_id] = {"worker_id": worker_id, "future": future}
        await self._sink(HarnessEvent(
            generation=self.generation,
            thread_id=self.thread_id,
            kind="prompt",
            turn_id=turn_id,
            prompt=PromptRequest(
                request_id=request_id,
                kind="device_tool",
                form={"tool": name, "input": tool_input},
                raw=message,
                item_id=worker_id,
            ),
        ))
        try:
            response = await future
            process = self._process
            if process is None or process.stdin is None or getattr(process.stdin, "is_closing", lambda: False)():
                return
            wire = {
                "type": "device_tool_result",
                "id": worker_id,
                "result": response["device_tool_result"],
                "is_error": response["is_error"],
            }
            process.stdin.write((json.dumps(wire, ensure_ascii=False) + "\n").encode())
            await process.stdin.drain()
        finally:
            self._device_tools.pop(request_id, None)

    async def _emit_item(
        self,
        turn_id: str,
        kind: str,
        *,
        item_id: str | None = None,
        text: str = "",
        title: str = "",
        raw: JsonObject | None = None,
        complete: bool = False,
    ) -> None:
        await self._sink(
            HarnessEvent(
                generation=self.generation,
                thread_id=self.thread_id,
                kind="item",
                turn_id=turn_id,
                item=ItemUpdate(
                    turn_id=turn_id,
                    item_id=item_id or str(uuid.uuid4()),
                    kind=kind,
                    text=text,
                    title=title,
                    raw=raw or {},
                    complete=complete,
                ),
            )
        )

    def prepare_answer(self, request_id: str | int, answer: JsonObject) -> JsonObject:
        device_tool = self._device_tools.get(str(request_id))
        if device_tool is not None:
            result = answer.get("device_tool_result")
            is_error = answer.get("is_error", False)
            if not isinstance(result, str) or len(result) > _DEVICE_TOOL_RESULT_LIMIT:
                raise CodeValidationError("The phone tool result is missing or too large.")
            if not isinstance(is_error, bool):
                raise CodeValidationError("The phone tool result status is invalid.")
            return {"device_tool_result": result, "is_error": is_error}
        if str(request_id) not in self._permissions:
            raise CodeConflictError("This Claude permission prompt is no longer active.")
        if not isinstance(answer.get("approved"), bool):
            raise CodeValidationError("Choose whether to allow this Claude action.")
        tool, tool_input = self._permission_inputs[str(request_id)]
        if tool == "AskUserQuestion" and answer["approved"]:
            questions = tool_input.get("questions")
            answers = answer.get("answers")
            if (
                not isinstance(questions, list)
                or not questions
                or not isinstance(answers, dict)
                or any(
                    not isinstance(question, dict)
                    or not isinstance(question.get("question"), str)
                    or not isinstance(answers.get(question["question"]), str)
                    or not answers[question["question"]].strip()
                    for question in questions
                )
            ):
                raise CodeValidationError("Answer each Claude question before continuing.")
        return answer

    async def respond(self, request_id: str | int, response: JsonObject) -> None:
        device_tool = self._device_tools.get(str(request_id))
        if device_tool is not None:
            future = device_tool["future"]
            if future.done():
                raise CodeConflictError("This phone tool request has already been answered.")
            future.set_result(response)
            return
        future = self._permissions.get(str(request_id))
        if future is None or future.done():
            raise CodeConflictError("This Claude permission prompt is no longer active.")
        future.set_result(response)

    async def interrupt(self, turn_id: str) -> None:
        if self._process is not None and self._turn_id == turn_id:
            self._process.terminate()

    async def delete_thread(self, thread_id: str) -> None:
        # Claude's public Agent SDK does not expose deleting one native transcript.
        return None

    async def close(self) -> None:
        self._closing = True
        self._fail_message_acks()
        for future in self._permissions.values():
            if not future.done():
                future.set_result({"approved": False, "message": "FCC session closed."})
        for device_tool in self._device_tools.values():
            future = device_tool["future"]
            if not future.done():
                future.cancel()
        reader = self._reader
        if reader is not None and not reader.done():
            reader.cancel()
            await asyncio.gather(reader, return_exceptions=True)
        if self._process is not None:
            if self._process.returncode is None:
                self._process.terminate()
            await self._process.wait()
            self._process = None
