import asyncio
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from free_claude_code.application.code_sessions.models import CodeUnavailableError, CodeValidationError
from free_claude_code.runtime.claude_agent import ClaudeAgentConnection


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("config", "default"),
        ("ask", "default"),
        ("auto_review", "auto"),
        ("full_access", "bypassPermissions"),
    ],
)
def test_sdk_permission_mode_matches_the_selected_session_mode(mode, expected):
    from free_claude_code.runtime.claude_agent import _sdk_permission_mode

    assert _sdk_permission_mode(mode) == expected


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("/fast olá", ("olá", True)),
        ("/fast     código\n    continuação", ("    código\n    continuação", True)),
        ("/fast\n\nAttachments:\n/tmp/image.jpg", ("/fast\n\nAttachments:\n/tmp/image.jpg", False)),
        ("/fast olá\n\nAttachments:\n/tmp/image.jpg", ("olá\n\nAttachments:\n/tmp/image.jpg", True)),
        ("/fast", ("/fast", False)),
        ("/fast   ", ("/fast   ", False)),
        ("/fastalgo", ("/fastalgo", False)),
        ("normal", ("normal", False)),
    ],
)
def test_fast_command_only_applies_to_the_current_prompt(text, expected):
    from free_claude_code.runtime.claude_agent import _fast_prompt

    assert _fast_prompt(text) == expected


@pytest.mark.parametrize("fast", [True, False])
def test_worker_passes_thinking_option_only_for_fast_turn(fast):
    from free_claude_code.runtime import claude_agent

    worker = Path(claude_agent.__file__).with_name("claude_agent_worker.cjs")
    script = """
const Module = require('node:module');
const original = Module._load;
Module._load = function(name, ...args) {
  if (name.endsWith('@anthropic-ai/claude-agent-sdk')) {
    return { query: async function* ({prompt, options}) {
      const first = await prompt[Symbol.asyncIterator]().next();
      yield {type: 'result', subtype: 'success', result: JSON.stringify({prompt: first.value.message.content, thinking: options.thinking ?? null})};
    }};
  }
  return original.call(this, name, ...args);
};
require(process.argv[1]);
"""
    request = {"fast": fast, "prompt": "olá", "sessionId": "session"}
    result = subprocess.run(
        ["node", "-e", script, str(worker)],
        input=json.dumps(request) + "\n", text=True, capture_output=True, check=True, timeout=10,
    )
    messages = [json.loads(line) for line in result.stdout.splitlines()]
    payload = json.loads(next(message["result"] for message in messages if message["type"] == "result"))
    assert payload == {"prompt": "olá", "thinking": {"type": "disabled"} if fast else None}


def test_send_message_requires_matching_worker_ack():
    async def exercise():
        writes = []
        async def drain():
            pass
        async def receive(event):
            pass
        stream = asyncio.StreamReader()
        async def wait():
            return 0
        connection = ClaudeAgentConnection(None, '/tmp', receive, {})
        connection.thread_id = 'session'
        connection._process = SimpleNamespace(returncode=None, stdout=stream, wait=wait,
            stdin=SimpleNamespace(write=writes.append, drain=drain, is_closing=lambda: False))
        reader = asyncio.create_task(connection._read_response('turn'))
        sending = asyncio.create_task(connection.send_message('op-1', 'hello'))
        await asyncio.sleep(0)
        assert json.loads(writes[0]) == {'type': 'user_message', 'operation_id': 'op-1', 'text': 'hello'}
        assert not sending.done()
        stream.feed_data(b'{"type":"message_ack","operation_id":"other"}\n')
        await asyncio.sleep(0)
        assert not sending.done()
        stream.feed_data(b'{"type":"message_ack","operation_id":"op-1"}\n')
        await asyncio.wait_for(sending, 1)
        stream.feed_eof()
        await reader
    asyncio.run(exercise())


def test_concurrent_messages_are_correlated_and_close_releases_pending():
    async def exercise():
        writes = []
        async def drain():
            pass
        async def receive(event):
            pass
        async def wait():
            return 0
        stream = asyncio.StreamReader()
        connection = ClaudeAgentConnection(None, '/tmp', receive, {})
        connection.thread_id = 'session'
        connection._process = SimpleNamespace(returncode=0, stdout=stream, wait=wait,
            stdin=SimpleNamespace(write=writes.append, drain=drain, is_closing=lambda: False))
        connection._process.returncode = None
        reader = asyncio.create_task(connection._read_response('turn'))
        first = asyncio.create_task(connection.send_message('first', 'one'))
        second = asyncio.create_task(connection.send_message('second', 'two'))
        await asyncio.sleep(0)
        stream.feed_data(b'{"type":"message_ack","operation_id":"second"}\n')
        await asyncio.wait_for(second, 1)
        assert not first.done()
        stream.feed_eof()
        with pytest.raises(CodeUnavailableError):
            await asyncio.wait_for(first, 1)
        await reader
        assert [json.loads(frame)['operation_id'] for frame in writes] == ['first', 'second']
    asyncio.run(exercise())


def test_close_releases_message_waiting_for_ack():
    async def exercise():
        async def drain():
            pass
        async def wait():
            return 0
        connection = ClaudeAgentConnection(None, '/tmp', None, {})
        connection._process = SimpleNamespace(returncode=None, wait=wait,
            terminate=lambda: None, stdin=SimpleNamespace(write=lambda data: None,
            drain=drain, is_closing=lambda: False))
        sending = asyncio.create_task(connection.send_message('op', 'hello'))
        await asyncio.sleep(0)
        await connection.close()
        with pytest.raises(CodeUnavailableError):
            await asyncio.wait_for(sending, 1)
    asyncio.run(exercise())


def test_close_releases_message_blocked_in_drain():
    async def exercise():
        entered = asyncio.Event()
        async def drain():
            entered.set()
            await asyncio.Event().wait()
        async def wait():
            return 0
        connection = ClaudeAgentConnection(None, '/tmp', None, {})
        connection._process = SimpleNamespace(returncode=None, wait=wait,
            terminate=lambda: None, stdin=SimpleNamespace(write=lambda data: None,
            drain=drain, is_closing=lambda: False))
        sending = asyncio.create_task(connection.send_message('op', 'hello'))
        await entered.wait()
        await connection.close()
        with pytest.raises(CodeUnavailableError):
            await asyncio.wait_for(sending, 0.2)
    asyncio.run(exercise())


def test_ack_is_read_while_permission_is_pending():
    async def exercise():
        events = []
        async def receive(event):
            events.append(event)
        async def drain():
            pass
        async def wait():
            return 0
        stream = asyncio.StreamReader()
        connection = ClaudeAgentConnection(None, '/tmp', receive, {})
        connection.thread_id = 'session'
        connection._process = SimpleNamespace(returncode=None, stdout=stream, wait=wait,
            stdin=SimpleNamespace(write=lambda data: None, drain=drain, is_closing=lambda: False))
        reader = asyncio.create_task(connection._read_response('turn'))
        stream.feed_data(b'{"type":"permission","id":"p","tool":"Bash","input":{}}\n')
        sending = asyncio.create_task(connection.send_message('op', 'hello'))
        await asyncio.sleep(0)
        stream.feed_data(b'{"type":"message_ack","operation_id":"op"}\n')
        await asyncio.wait_for(sending, 1)
        stream.feed_eof()
        await reader
        assert any(event.kind == 'prompt' for event in events)
    asyncio.run(exercise())


def test_send_message_times_out_without_ack(monkeypatch):
    from free_claude_code.runtime import claude_agent
    monkeypatch.setattr(claude_agent, '_MESSAGE_ACK_TIMEOUT', 0.02, raising=False)

    async def exercise():
        writes = []
        async def drain():
            pass
        connection = ClaudeAgentConnection(None, '/tmp', None, {})
        connection._process = SimpleNamespace(returncode=None, stdin=SimpleNamespace(
            write=writes.append, drain=drain, is_closing=lambda: False))
        with pytest.raises(TimeoutError):
            await connection.send_message('op-1', 'Olá\nsegunda linha')
        assert [json.loads(frame) for frame in writes] == [
            {'type': 'user_message', 'operation_id': 'op-1', 'text': 'Olá\nsegunda linha'}]

    asyncio.run(exercise())


def test_send_message_rejects_unavailable_worker():
    async def exercise():
        connection = ClaudeAgentConnection(None, '/tmp', None, {})
        for process in (None, SimpleNamespace(returncode=0, stdin=None),
                        SimpleNamespace(returncode=None, stdin=None),
                        SimpleNamespace(returncode=None, stdin=SimpleNamespace(is_closing=lambda: True))):
            connection._process = process
            with pytest.raises(CodeUnavailableError):
                await connection.send_message('op-1', 'mensagem')

    asyncio.run(exercise())


def test_worker_streams_input_while_permission_pending_and_exits_with_stdin_open():
    from free_claude_code.runtime import claude_agent

    worker = Path(claude_agent.__file__).with_name('claude_agent_worker.cjs')
    script = r"""
const Module = require('node:module');
const original = Module._load;
Module._load = function(name, ...args) {
  if (name.endsWith('@anthropic-ai/claude-agent-sdk')) {
    return { query: async function* ({prompt, options}) {
      const it = prompt[Symbol.asyncIterator]();
      const first = await it.next();
      const decision = await options.canUseTool('Bash', {command: 'pwd'});
      const second = await it.next();
      yield {type: 'result', subtype: 'success', result: JSON.stringify({first: first.value, second: second.value, decision})};
      await it.next();
    }};
  }
  return original.call(this, name, ...args);
};
require(process.argv[1]);
"""
    async def exercise():
        process = await asyncio.create_subprocess_exec('node', '-e', script, str(worker),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            process.stdin.write((json.dumps({'prompt': 'inicial', 'sessionId': 'session'}) + '\n').encode())
            await process.stdin.drain()
            permission = json.loads(await asyncio.wait_for(process.stdout.readline(), 3))
            assert permission['type'] == 'permission'
            for frame in ({'type': 'user_message', 'operation_id': 'op-1', 'text': 'nova'},
                          {'type': 'permission_response', 'id': permission['id'], 'approved': True, 'input': {'command': 'pwd'}}):
                process.stdin.write((json.dumps(frame) + '\n').encode())
            await process.stdin.drain()
            ack = json.loads(await asyncio.wait_for(process.stdout.readline(), 3))
            assert ack == {'type': 'message_ack', 'operation_id': 'op-1'}
            result = json.loads(await asyncio.wait_for(process.stdout.readline(), 3))
            assert result['type'] == 'result'
            payload = json.loads(result['result'])
            assert payload['first']['message'] == {'role': 'user', 'content': 'inicial'}
            assert payload['second']['message'] == {'role': 'user', 'content': 'nova'}
            assert payload['decision'] == {'behavior': 'allow', 'updatedInput': {'command': 'pwd'}}
            assert await asyncio.wait_for(process.wait(), 3) == 0
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(exercise())


def test_worker_does_not_assume_result_consumed_queued_message():
    from free_claude_code.runtime import claude_agent

    worker = Path(claude_agent.__file__).with_name('claude_agent_worker.cjs')
    script = r"""
const Module = require('node:module');
const original = Module._load;
Module._load = function(name, ...args) {
  if (name.endsWith('@anthropic-ai/claude-agent-sdk')) {
    return { query: async function* ({prompt, options}) {
      const it = prompt[Symbol.asyncIterator]();
      await it.next();
      await options.canUseTool('Bash', {});
      yield {type: 'result', subtype: 'success'};
      const next = await it.next();
      yield {type: 'assistant', message: {id: 'after', content: [{type: 'text', text: next.value.message.content}]}};
      await it.next();
    }};
  }
  return original.call(this, name, ...args);
};
require(process.argv[1]);
"""
    async def exercise():
        process = await asyncio.create_subprocess_exec('node', '-e', script, str(worker),
            stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            process.stdin.write(b'{"prompt":"first","sessionId":"s"}\n')
            await process.stdin.drain()
            permission = json.loads(await asyncio.wait_for(process.stdout.readline(), 3))
            for frame in ({'type': 'user_message', 'operation_id': 'op-1', 'text': 'not yet consumed'},
                          {'type': 'permission_response', 'id': permission['id'], 'approved': True, 'input': {}}):
                process.stdin.write((json.dumps(frame) + '\n').encode())
            await process.stdin.drain()
            assert json.loads(await asyncio.wait_for(process.stdout.readline(), 3)) == {'type': 'message_ack', 'operation_id': 'op-1'}
            assert json.loads(await asyncio.wait_for(process.stdout.readline(), 3))['type'] == 'result'
            text = json.loads(await asyncio.wait_for(process.stdout.readline(), 3))
            assert text['type'] == 'text'
            assert text['text'] == 'not yet consumed'
            assert await asyncio.wait_for(process.wait(), 3) == 0
        finally:
            if process.returncode is None:
                process.kill()
                await process.wait()

    asyncio.run(exercise())


def test_tool_call_and_result_have_distinct_transcript_identities():
    async def exercise():
        events = []

        async def receive(event):
            events.append(event)

        stream = asyncio.StreamReader()
        for message in (
            {"type": "tool_call", "id": "call_1", "name": "Bash", "input": {"command": "ls"}},
            {"type": "tool_result", "id": "call_1", "content": "file.txt"},
        ):
            stream.feed_data((json.dumps(message) + "\n").encode())
        stream.feed_eof()

        async def wait():
            return 0

        connection = ClaudeAgentConnection(None, "/tmp", receive, {})
        connection.thread_id = "thread"
        connection._process = SimpleNamespace(stdout=stream, wait=wait)
        await connection._read_response("turn")
        return [event.item for event in events if event.kind == "item"]

    items = asyncio.run(exercise())
    assert [item.kind for item in items] == ["tool_call", "tool_result"]
    assert items[0].item_id != items[1].item_id


def test_large_image_tool_result_finishes_turn():
    async def exercise():
        events = []

        async def receive(event):
            events.append(event)

        stream = asyncio.StreamReader()
        image = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "a" * 460000}}
        for message in (
            {"type": "tool_result", "id": "call_1", "content": [image]},
            {"type": "result", "subtype": "success"},
        ):
            stream.feed_data((json.dumps(message) + "\n").encode())
        stream.feed_eof()

        async def wait():
            return 0

        connection = ClaudeAgentConnection(None, "/tmp", receive, {})
        connection.thread_id = "thread"
        connection._process = SimpleNamespace(stdout=stream, wait=wait, returncode=0)
        await connection._read_response("turn")
        return events

    events = asyncio.run(exercise())
    assert any(event.kind == "turn_completed" and event.status == "completed" for event in events)
    item = next(event.item for event in events if event.kind == "item")
    assert "a" * 1000 not in item.text


def test_worker_failure_emits_terminal_failed_turn():
    async def exercise():
        events = []

        async def receive(event):
            events.append(event)

        stream = asyncio.StreamReader()
        stream.feed_data(b'{"type":"error","message":"worker failed"}\n')
        stream.feed_eof()

        async def wait():
            return 1

        connection = ClaudeAgentConnection(None, "/tmp", receive, {})
        connection.thread_id = "thread"
        connection._process = SimpleNamespace(stdout=stream, wait=wait, returncode=1)
        await connection._read_response("turn")
        return events

    events = asyncio.run(exercise())
    assert any(event.kind == "turn_completed" and event.status == "failed" for event in events)


def test_text_snapshots_update_one_entry_before_completion():
    async def exercise():
        events = []

        async def receive(event):
            events.append(event)

        stream = asyncio.StreamReader()
        for text, complete in (("Ol", False), ("Olá", False), ("Olá!", True)):
            stream.feed_data((json.dumps({"type": "text", "id": "text:message:0", "text": text, "complete": complete}) + "\n").encode())
        stream.feed_eof()

        async def wait():
            return 0

        connection = ClaudeAgentConnection(None, "/tmp", receive, {})
        connection.thread_id = "thread"
        connection._process = SimpleNamespace(stdout=stream, wait=wait)
        await connection._read_response("turn")
        return [event.item for event in events if event.kind == "item"]

    items = asyncio.run(exercise())
    assert len({item.item_id for item in items}) == 1
    assert [item.complete for item in items] == [False, False, True]
    assert [item.text for item in items] == ["Ol", "Olá", "Olá!"]


def test_clarifying_question_requires_and_forwards_selected_answer():
    async def exercise():
        events = []
        received = asyncio.Event()
        writes = []

        async def receive(event):
            events.append(event)
            received.set()

        async def drain():
            pass

        connection = ClaudeAgentConnection(None, "/tmp", receive, {})
        connection.thread_id = "thread"
        connection._process = SimpleNamespace(stdin=SimpleNamespace(write=writes.append, drain=drain))
        question = {"question": "Qual opção?", "header": "Escolha", "multiSelect": False,
                    "options": [{"label": "Sim"}, {"label": "Não"}]}
        task = asyncio.create_task(connection._emit_permission("turn", {
            "id": "permission-1", "tool": "AskUserQuestion", "input": {"questions": [question]},
        }))
        await received.wait()
        request_id = events[0].prompt.request_id
        with pytest.raises(CodeValidationError):
            connection.prepare_answer(request_id, {"approved": True})
        answer = connection.prepare_answer(request_id, {
            "approved": True, "answers": {"Qual opção?": "Não"},
        })
        await connection.respond(request_id, answer)
        await task
        return json.loads(writes[0])

    wire = asyncio.run(exercise())
    assert wire["approved"] is True
    assert wire["input"]["answers"] == {"Qual opção?": "Não"}
    assert wire["input"]["questions"][0]["question"] == "Qual opção?"
