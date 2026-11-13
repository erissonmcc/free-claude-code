// Exercise the actual in-process MCP and browser before switching production traffic.
const http = require('node:http');
const os = require('node:os');
const fs = require('node:fs/promises');
const path = require('node:path');
const assert = require('node:assert/strict');
const {playwrightServer} = require('../../src/free_claude_code/runtime/claude_playwright.cjs');

(async () => {
  const home = await fs.mkdtemp(path.join(os.tmpdir(), 'fcc-browser-check-'));
  const pageServer = http.createServer((req, res) => res.end('<title>FCC Playwright ready</title><h1>Browser check</h1>'));
  await new Promise(resolve => pageServer.listen(0, '127.0.0.1', resolve));
  const browser = await playwrightServer({
    home, sessionId:'build-check',
    createConnection: require('@playwright/mcp').createConnection,
    chromium: require('playwright').chromium,
  });
  const pending = new Map();
  let nextId = 0;
  const transport = {
    async start() {}, async close() {},
    async send(message) {
      const resolve = pending.get(message.id);
      if (resolve) {pending.delete(message.id); resolve(message);}
    },
  };
  const rpc = async (method, params) => {
    const id = ++nextId;
    const response = new Promise(resolve => pending.set(id, resolve));
    transport.onmessage({jsonrpc:'2.0', id, method, params});
    const result = await response;
    if (result.error) throw new Error(JSON.stringify(result.error));
    if (result.result.isError) throw new Error(JSON.stringify(result.result.content));
    return result.result;
  };
  const timeout = setTimeout(() => {console.error('MCP browser verification timed out.'); process.exit(1);}, 60000);
  try {
    await browser.config.instance.connect(transport);
    await rpc('initialize', {protocolVersion:'2025-03-26', capabilities:{}, clientInfo:{name:'fcc-build',version:'1.0.0'}});
    transport.onmessage({jsonrpc:'2.0', method:'notifications/initialized'});
    const tools = await rpc('tools/list');
    assert(tools.tools.some(tool => tool.name === 'browser_navigate'));
    await rpc('tools/call', {name:'browser_navigate', arguments:{url:`http://127.0.0.1:${pageServer.address().port}`}});
    const screenshot = await rpc('tools/call', {name:'browser_take_screenshot', arguments:{type:'png'}});
    assert(screenshot.content.some(block => block.type === 'image'));
    const snapshot = await rpc('tools/call', {name:'browser_snapshot', arguments:{}});
    assert(JSON.stringify(snapshot).includes('FCC Playwright ready'));
    console.log(`MCP verified: ${tools.tools.length} tools; navigation, screenshot and snapshot succeeded.`);
  } finally {
    clearTimeout(timeout);
    await browser.close();
    pageServer.closeAllConnections();
    await new Promise(resolve => pageServer.close(resolve));
    await fs.rm(home, {recursive:true, force:true});
  }
})().catch(error => {console.error(error); process.exitCode = 1;});
