const {test} = require('node:test');
const assert = require('node:assert/strict');
const {forkAtMessage} = require('../../src/free_claude_code/runtime/claude_session_fork.cjs');
test('forks at selected native assistant UUID, excluding newer turns and subagent messages', async () => {
  let call;
  const sdk = {
    getSessionMessages: async () => [
      {type:'assistant', uuid:'a', message:{id:'msg1'}},
      {type:'assistant', uuid:'sub', parent_tool_use_id:'tool', message:{id:'msg1'}},
      {type:'assistant', uuid:'b', message:{id:'msg2'}},
    ],
    forkSession: async (...args) => {call=args;return {sessionId:'fork'};},
  };
  assert.deepEqual(await forkAtMessage(sdk, {sessionId:'source', itemId:'text:msg1:0', cwd:'/work', title:'Title'}), {sessionId:'fork'});
  assert.deepEqual(call, ['source',{dir:'/work',upToMessageId:'a',title:'Title'}]);
});
test('missing or ambiguous selected message refuses to fork rather than copying latest history', async () => {
  for(const messages of [[], [1,2].map(i=>({type:'assistant',uuid:String(i),message:{id:'same'}}))]) {
    await assert.rejects(forkAtMessage({getSessionMessages:async()=>messages, forkSession:()=>assert.fail('must not fork')}, {itemId:'text:same:0'}));
  }
});
