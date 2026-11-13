const test = require('node:test');
const assert = require('node:assert/strict');
const { ClaudeTextStream, ClaudeUsageStream } = require('../../src/free_claude_code/runtime/claude_agent_stream.cjs');

test('usage snapshots replace cumulative counts and accumulate messages', () => {
  const usage = new ClaudeUsageStream();
  const event = (event) => usage.accept({ type: 'stream_event', event });
  assert.equal(event({ type: 'message_start', message: { id: 'a', usage: { input_tokens: 10, cache_read_input_tokens: 20, output_tokens: 1 } } }), 31);
  assert.equal(event({ type: 'message_delta', usage: { output_tokens: 5 } }), 35);
  assert.equal(event({ type: 'message_delta', usage: { output_tokens: 5 } }), null);
  assert.equal(event({ type: 'message_start', message: { id: 'b', usage: { input_tokens: 4, output_tokens: 0 } } }), 39);
  assert.equal(usage.accept({ type: 'stream_event', parent_tool_use_id: 'child', event: { type: 'message_delta', usage: { output_tokens: 100 } } }), null);
});

function start(stream, id = 'message') {
  stream.accept({ type: 'stream_event', event: { type: 'message_start', message: { id } } });
}
function block(stream, index) {
  stream.accept({ type: 'stream_event', event: { type: 'content_block_start', index, content_block: { type: 'text', text: '' } } });
}
function delta(stream, index, text) {
  return stream.accept({ type: 'stream_event', event: { type: 'content_block_delta', index, delta: { type: 'text_delta', text } } });
}

test('partial text arrives before final assistant and updates the same entry', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  block(stream, 0);
  const first = delta(stream, 0, 'Olá')[0];
  const second = delta(stream, 0, '!')[0];
  const final = stream.accept({ type: 'assistant', message: { id: 'message', content: [{ type: 'text', text: 'Olá!' }] } })[0];
  assert.deepEqual([first.text, second.text, final.text], ['Olá', 'Olá!', 'Olá!']);
  assert.deepEqual([first.complete, second.complete, final.complete], [false, false, true]);
  assert.equal(first.id, final.id);
  assert.deepEqual(stream.accept({ type: 'stream_event', event: { type: 'content_block_stop', index: 0 } }), []);
});

test('multiple blocks and model messages retain separate identities', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  block(stream, 1);
  const first = delta(stream, 1, 'Primeiro')[0];
  stream.accept({ type: 'stream_event', event: { type: 'content_block_stop', index: 1 } });
  block(stream, 3);
  const second = delta(stream, 3, 'Segundo')[0];
  start(stream, 'next-message');
  block(stream, 0);
  const third = delta(stream, 0, 'Terceiro')[0];
  assert.equal(new Set([first.id, second.id, third.id]).size, 3);
});

test('fallback assistant and subagent text remain visible', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  block(stream, 0);
  delta(stream, 0, 'Principal');
  const child = stream.accept({ type: 'assistant', parent_tool_use_id: 'agent', uuid: 'child', message: { id: 'message', content: [{ type: 'text', text: 'Subagente' }] } })[0];
  assert.equal(child.text, 'Subagente');
  assert.equal(child.complete, true);
  assert.notEqual(child.id, 'text:message:0');
});

test('assistant text retains its streamed identity after a tool block starts', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  block(stream, 0);
  const first = delta(stream, 0, 'Vou verificar.')[0];
  stream.accept({ type: 'stream_event', event: { type: 'content_block_start', index: 1, content_block: { type: 'tool_use' } } });
  const final = stream.accept({ type: 'assistant', message: { id: 'message', content: [{ type: 'text', text: 'Vou verificar.' }] } })[0];
  assert.equal(final.id, first.id);
});

test('assistant confirmation reconciles each streamed text block by its own index', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  block(stream, 0);
  const first = delta(stream, 0, 'Primeira parte')[0];
  stream.accept({ type: 'stream_event', event: { type: 'content_block_stop', index: 0 } });
  block(stream, 2);
  const second = delta(stream, 2, 'Segunda parte')[0];

  const confirmed = stream.accept({
    type: 'assistant',
    message: { id: 'message', content: [
      { type: 'text', text: 'Primeira parte' },
      { type: 'tool_use', id: 'tool-1', name: 'Bash', input: {} },
      { type: 'text', text: 'Segunda parte' },
    ] },
  });

  assert.deepEqual(confirmed.map(item => item.id), [second.id]);
  assert.deepEqual(confirmed.map(item => item.text), ['Segunda parte']);
  assert.equal(stream.blocks.get(0).text, 'Primeira parte');
  assert.equal(stream.blocks.get(0).complete, true);
  assert.equal(confirmed[0].complete, true);
});

test('turn completion finalizes cut-off text without dropping it', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  block(stream, 0);
  delta(stream, 0, 'Parcial');
  const final = stream.accept({ type: 'result' })[0];
  assert.equal(final.text, 'Parcial');
  assert.equal(final.complete, true);
});

test('single-block SDK confirmations keep the stream index after thinking', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  stream.accept({ type: 'stream_event', event: { type: 'content_block_start', index: 0, content_block: { type: 'thinking', thinking: '' } } });
  stream.accept({ type: 'assistant', uuid: 'thinking', message: { id: 'message', content: [{ type: 'thinking', thinking: 'Vou verificar' }] } });
  block(stream, 1);
  const first = delta(stream, 1, 'Vou consultar o horário.')[0];
  stream.accept({ type: 'stream_event', event: { type: 'content_block_stop', index: 1 } });
  assert.deepEqual(stream.accept({ type: 'assistant', uuid: 'text', message: { id: 'message', content: [{ type: 'text', text: first.text }] } }), []);
});

test('successive single-block confirmations do not overwrite or duplicate text', () => {
  const stream = new ClaudeTextStream();
  start(stream);
  const entries = new Map();
  for (const index of [0, 1]) {
    block(stream, index);
    const text = index === 0 ? 'Primeiro' : 'Segundo';
    for (const frame of delta(stream, index, text)) entries.set(frame.id, frame);
    for (const frame of stream.accept({ type: 'stream_event', event: { type: 'content_block_stop', index } })) entries.set(frame.id, frame);
    for (const frame of stream.accept({ type: 'assistant', uuid: `block-${index}`, message: { id: 'message', content: [{ type: 'text', text }] } })) entries.set(frame.id, frame);
  }
  assert.deepEqual([...entries.keys()], ['text:message:0', 'text:message:1']);
  assert.equal(entries.size, 2);
  assert.deepEqual([...entries.values()].map(frame => frame.text), ['Primeiro', 'Segundo']);
});
