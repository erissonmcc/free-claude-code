const test = require('node:test');
const assert = require('node:assert/strict');
const {playwrightServer} = require('../../src/free_claude_code/runtime/claude_playwright.cjs');

test('direct Playwright runs locally with a separate profile per conversation', () => {
  const first = playwrightServer({enabled:true, cli:'/packages/playwright/cli.js', home:'/data', sessionId:'session-one'});
  const second = playwrightServer({enabled:true, cli:'/packages/playwright/cli.js', home:'/data', sessionId:'session-two'});
  assert.equal(first.command, process.execPath);
  assert.equal(first.args[0], '/packages/playwright/cli.js');
  assert(first.args.includes('--headless'));
  assert(first.args.includes('--no-sandbox'));
  assert(first.args.includes('/data/.fcc/playwright/session-one'));
  assert(second.args.includes('/data/.fcc/playwright/session-two'));
  assert(!first.args.some(value => value.includes('cdp-endpoint')));
});
test('an installation without browser support does not register Playwright', () => {
  assert.equal(playwrightServer({enabled:false}), undefined);
});
test('session IDs cannot escape the browser profile directory', () => {
  assert.throws(() => playwrightServer({enabled:true, cli:'/cli.js', home:'/data', sessionId:'../../other'}));
});
