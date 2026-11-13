const test = require('node:test');
const assert = require('node:assert/strict');
test('device tool request waits for the app result and correlates it by id', async () => {
  let DeviceToolRpc;
  assert.doesNotThrow(() => ({ DeviceToolRpc } = require('../../src/free_claude_code/runtime/claude_device_tool_bridge.cjs')));
  const sent = [];
  const rpc = new DeviceToolRpc({ send: value => sent.push(value), timeoutMs: 100 });
  const resultPromise = rpc.request('read_file', { path: 'notes.txt' });
  assert.equal(sent.length, 1);
  assert.equal(sent[0].type, 'device_tool');
  assert.equal(sent[0].name, 'read_file');
  assert.deepEqual(sent[0].input, { path: 'notes.txt' });
  rpc.resolve({ type: 'device_tool_result', id: sent[0].id, result: 'local contents', is_error: false });
  assert.deepEqual(await resultPromise, { result: 'local contents', is_error: false });
  assert.equal(rpc.pendingCount, 0);
});

test('device tool request expires when the phone does not answer', async () => {
  const { DeviceToolRpc } = require('../../src/free_claude_code/runtime/claude_device_tool_bridge.cjs');
  const rpc = new DeviceToolRpc({ send() {}, timeoutMs: 10 });
  await assert.rejects(rpc.request('list_directory', { path: '.' }), /timed out/i);
  assert.equal(rpc.pendingCount, 0);
});
