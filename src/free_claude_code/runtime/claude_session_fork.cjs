// Resolve the visible block to the SDK transcript UUID; never fall back to the latest turn.
async function forkAtMessage(sdk, request) {
  const messages = await sdk.getSessionMessages(request.sessionId, {dir: request.cwd});
  const id = request.itemId || '';
  const candidates = messages.filter(entry => entry.type === 'assistant' && !entry.parent_tool_use_id &&
    (id.startsWith(`text:${entry.message?.id}:`) || id.startsWith(`text:${entry.uuid}:`)));
  if (candidates.length !== 1) throw new Error('Não foi possível localizar esta resposta no histórico nativo. A conversa original foi preservada.');
  return sdk.forkSession(request.sessionId, {dir: request.cwd, upToMessageId: candidates[0].uuid, title: request.title});
}
module.exports = {forkAtMessage};

async function forkBeforeTurn(sdk, request) {
  const messages = (await sdk.getSessionMessages(request.sessionId, {dir: request.cwd})).filter(entry => !entry.parent_tool_use_id);
  const selected = messages.map((entry,index) => ({entry,index})).filter(({entry}) => entry.type === 'assistant' &&
    (request.itemId.startsWith(`text:${entry.message?.id}:`) || request.itemId.startsWith(`text:${entry.uuid}:`)));
  if (selected.length !== 1) throw new Error('A resposta selecionada não foi localizada no histórico nativo.');
  const prompt = request.prompt.startsWith('/fast ') ? request.prompt.slice(6).trim() : request.prompt;
  const text = entry => typeof entry.message?.content === 'string' ? entry.message.content :
    (entry.message?.content || []).filter(block=>block.type === 'text').map(block=>block.text).join('');
  let user = selected[0].index - 1;
  while (user >= 0 && !(messages[user].type === 'user' && text(messages[user]) === prompt)) user--;
  if (user < 0) throw new Error('A pergunta original não foi localizada. O histórico foi preservado.');
  if (user === 0) return {sessionId:null}; // First turn: begin a fresh native session.
  return sdk.forkSession(request.sessionId, {dir:request.cwd, upToMessageId:messages[user-1].uuid, title:request.title});
}
module.exports.forkBeforeTurn = forkBeforeTurn;
