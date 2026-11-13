const test = require('node:test');
const assert = require('node:assert/strict');
const {playwrightServer} = require('../../src/free_claude_code/runtime/claude_playwright.cjs');

test('browser starts lazily, reuses its context, and closes at turn completion', async () => {
  let getter, launches = 0, closes = 0, serverCloses = 0;
  const context = {on() {}, async close() {closes++;}};
  const server = await playwrightServer({home:'/data', sessionId:'session-one',
    createConnection: async (config, getContext) => {getter=getContext; return {async close() {serverCloses++;}};},
    chromium: {async launchPersistentContext(profile, options) {
      launches++;
      assert.equal(profile, '/data/.fcc/playwright/session-one');
      assert.equal(options.headless, true);
      return context;
    }},
  });
  assert.equal(launches, 0);
  assert.equal(server.config.type, 'sdk');
  assert.equal(await getter(), context);
  assert.equal(await getter(), context);
  assert.equal(launches, 1);
  await server.close();
  assert.equal(closes, 1);
  assert.equal(serverCloses, 1);
});
test('session IDs cannot escape the browser profile directory', async () => {
  await assert.rejects(playwrightServer({home:'/data', sessionId:'../../other'}));
});
