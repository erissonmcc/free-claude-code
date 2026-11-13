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
