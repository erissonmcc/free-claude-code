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
test('regeneration forks before the original user turn and drops its answer and later turns', async () => {
  const {forkBeforeTurn} = require('../../src/free_claude_code/runtime/claude_session_fork.cjs');
  const messages = [
    {type:'user',uuid:'u0',message:{role:'user',content:'earlier'}},
    {type:'assistant',uuid:'a0',message:{id:'earlier',content:[]}},
    {type:'user',uuid:'u1',message:{role:'user',content:'question'}},
    {type:'assistant',uuid:'a1',message:{id:'selected',content:[]}},
    {type:'user',uuid:'u2',message:{role:'user',content:'later'}},
  ];
  let args;
  const sdk={getSessionMessages:async()=>messages,forkSession:async(...a)=>{args=a;return{sessionId:'new'};}};
  assert.equal((await forkBeforeTurn(sdk,{sessionId:'old',itemId:'text:selected:0',prompt:'question',cwd:'/work'})).sessionId,'new');
  assert.equal(args[1].upToMessageId,'a0');
  sdk.getSessionMessages=async()=>messages.slice(2);
  args=null;
  assert.equal((await forkBeforeTurn(sdk,{itemId:'text:selected:0',prompt:'question'})).sessionId,null);
  assert.equal(args,null);
  await assert.rejects(forkBeforeTurn(sdk,{itemId:'text:selected:0',prompt:'missing'}));
});
