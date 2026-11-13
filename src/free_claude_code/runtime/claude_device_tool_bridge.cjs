const { randomUUID } = require('node:crypto');

const MAX_RESULT_CHARS = 512 * 1024;

class DeviceToolRpc {
  constructor({ send, timeoutMs = 60_000 }) {
    this.send = send;
    this.timeoutMs = timeoutMs;
    this.pending = new Map();
    this.closed = false;
  }

  get pendingCount() {
    return this.pending.size;
  }

  request(name, input) {
    if (this.closed) return Promise.reject(new Error('The phone tool connection is closed.'));
    const id = randomUUID();
    return new Promise((resolve, reject) => {
      const timer = setTimeout(() => {
        this.pending.delete(id);
        reject(new Error('The phone did not answer the tool request before it timed out.'));
      }, this.timeoutMs);
      this.pending.set(id, { resolve, reject, timer });
      try {
        this.send({ type: 'device_tool', id, name, input });
      } catch (error) {
        clearTimeout(timer);
        this.pending.delete(id);
        reject(error);
      }
    });
  }

  resolve(message) {
    if (message?.type !== 'device_tool_result' || typeof message.id !== 'string') return false;
    const pending = this.pending.get(message.id);
    if (!pending) return false;
    clearTimeout(pending.timer);
    this.pending.delete(message.id);
    if (typeof message.result !== 'string' || message.result.length > MAX_RESULT_CHARS) {
      pending.reject(new Error('The phone tool result was invalid or exceeded the size limit.'));
      return true;
    }
    pending.resolve({ result: message.result, is_error: message.is_error === true });
    return true;
  }

  close(reason = 'The phone tool connection closed.') {
    this.closed = true;
    for (const pending of this.pending.values()) {
      clearTimeout(pending.timer);
      pending.reject(new Error(reason));
    }
    this.pending.clear();
  }
}

function createPhoneToolServer({ createSdkMcpServer, tool, z, rpc }) {
  const text = z.string();
  const definitions = [
    tool('read_file', 'Read a UTF-8 file from the Android shared-storage workspace. Use paths relative to the phone workspace.', { path: text }, args => rpc.request('read_file', args)),
    tool('write_file', 'Write UTF-8 content to a file in the Android shared-storage workspace.', { path: text, content: text }, args => rpc.request('write_file', args)),
    tool('edit_file', 'Replace text in a file in the Android shared-storage workspace.', { path: text, old_string: text, new_string: text, replace_all: z.boolean().optional() }, args => rpc.request('edit_file', args)),
    tool('list_directory', 'List one directory in the Android shared-storage workspace.', { path: text }, args => rpc.request('list_directory', args)),
    tool('find_files', 'Find files in the Android shared-storage workspace by glob pattern.', { path: text, pattern: text }, args => rpc.request('find_files', args)),
    tool('search_files', 'Search text files in the Android shared-storage workspace.', { path: text, query: text }, args => rpc.request('search_files', args)),
    tool('run_command', 'Run a shell command on the Android device. The working directory must remain inside the shared-storage workspace.', { command: text, cwd: text.optional() }, args => rpc.request('run_command', args)),
  ];
  const tools = definitions.map(definition => ({
    ...definition,
    handler: async (input, extra) => {
      try {
        const result = await definition.handler(input, extra);
        return { content: [{ type: 'text', text: result.result }], ...(result.is_error ? { isError: true } : {}) };
      } catch (error) {
        return { content: [{ type: 'text', text: error instanceof Error ? error.message : String(error) }], isError: true };
      }
    },
  }));
  return createSdkMcpServer({
    name: 'phone',
    version: '1.0.0',
    instructions: 'These tools run on the user Android device. Use them for all project files and commands. Paths are relative to shared storage. Do not use server-side file or shell tools.',
    tools,
  });
}

module.exports = { DeviceToolRpc, createPhoneToolServer, MAX_RESULT_CHARS };
