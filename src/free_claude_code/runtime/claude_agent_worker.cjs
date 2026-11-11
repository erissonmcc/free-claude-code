const readline = require('node:readline');
const { query } = require('@anthropic-ai/claude-agent-sdk');
const { ClaudeTextStream, ClaudeUsageStream } = require('./claude_agent_stream.cjs');

const input = readline.createInterface({ input: process.stdin, crlfDelay: Infinity });
const lines = input[Symbol.asyncIterator]();
const pending = new Map();
let nextPermission = 0;
const send = (value) => process.stdout.write(`${JSON.stringify(value)}\n`);

async function readInput() {
  const value = await lines.next();
  if (value.done) throw new Error('FCC closed the Agent SDK bridge.');
  return JSON.parse(value.value);
}

async function main() {
  const request = await readInput();
  const options = {
    cwd: request.cwd,
    model: request.model,
    effort: request.effort || undefined,
    ...(request.fast ? { thinking: { type: 'disabled' } } : {}),
    permissionMode: request.permissionMode,
    settingSources: ['user', 'project', 'local'],
    env: request.env,
    includePartialMessages: true,
    systemPrompt: {
      type: 'preset',
      preset: 'claude_code',
      append: 'At the beginning of each user turn, before your first tool call, briefly tell the user what you intend to do in their language.',
      snapshot: false,
    },
    pathToClaudeCodeExecutable: request.binary,
    ...(request.resume ? { resume: request.sessionId } : { sessionId: request.sessionId }),
    canUseTool: async (toolName, inputData) => {
      const id = String(++nextPermission);
      send({ type: 'permission', id, tool: toolName, input: inputData });
      return new Promise((resolve) => pending.set(id, resolve));
    },
  };
  const queued = [];
  let wake;
  let closed = false;
  const closePrompt = () => {
    closed = true;
    if (wake) { wake(); wake = undefined; }
  };
  const enqueue = (text) => {
    if (closed) return;
    queued.push({ type: 'user', session_id: request.sessionId, parent_tool_use_id: null,
      message: { role: 'user', content: text } });
    if (wake) { wake(); wake = undefined; }
  };
  async function* prompt() {
    while (queued.length || !closed) {
      if (queued.length) yield queued.shift();
      else await new Promise((resolve) => { wake = resolve; });
    }
  }
  enqueue(request.prompt);
  const inputTask = (async () => {
    while (true) {
      const line = await lines.next();
      if (line.done) break;
      let message;
      try { message = JSON.parse(line.value); } catch { continue; }
      if (message.type === 'user_message' && typeof message.text === 'string') {
        if (!closed && typeof message.operation_id === 'string' && message.operation_id) {
          enqueue(message.text);
          send({ type: 'message_ack', operation_id: message.operation_id });
        }
      } else if (message.type === 'permission_response') {
        const resolve = pending.get(String(message.id));
        if (resolve) {
          pending.delete(String(message.id));
          resolve(message.approved
            ? { behavior: 'allow', updatedInput: message.input }
            : { behavior: 'deny', message: message.message || 'The user declined this action.' });
        }
      }
    }
    closePrompt();
    for (const resolve of pending.values()) resolve({ behavior: 'deny', message: 'FCC closed the Agent SDK bridge.' });
    pending.clear();
  })();

  try {
    const textStream = new ClaudeTextStream();
    const usageStream = new ClaudeUsageStream();
    for await (const message of query({ prompt: prompt(), options })) {
      const tokens = usageStream.accept(message);
      if (tokens !== null) send({ type: 'usage', tokens });
      for (const frame of textStream.accept(message)) send(frame);
      if (message.type === 'assistant') {
        for (const block of message.message.content || []) {
          if (block.type === 'tool_use') send({ type: 'tool_call', id: block.id, name: block.name, input: block.input });
        }
      } else if (message.type === 'user') {
        for (const block of message.message.content || []) {
          if (block.type === 'tool_result') send({ type: 'tool_result', id: block.tool_use_id, content: block.content, error: block.is_error });
        }
      } else if (message.type === 'result') {
        send({ type: 'result', subtype: message.subtype, result: message.result, usage: message.usage, cost: message.total_cost_usd });
        closePrompt();
      }
    }
  } catch (error) {
    send({ type: 'error', message: error instanceof Error ? error.message : String(error) });
    process.exitCode = 1;
  } finally {
    closePrompt();
    input.close();
    process.stdin.pause();
    await inputTask;
  }
}

main();
