const path = require('node:path');

function playwrightServer({enabled, cli, home, sessionId}) {
  if (!enabled) return undefined;
  if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(sessionId)) {
    throw new Error('Invalid session ID for the Playwright browser profile.');
  }
  return {
    type: 'stdio',
    command: process.execPath,
    args: [
      cli, '--browser', 'chromium', '--headless', '--no-sandbox',
      '--caps', 'vision',
      '--user-data-dir', path.join(home, '.fcc', 'playwright', sessionId),
      '--output-dir', path.join(home, '.fcc', 'playwright', sessionId, 'artifacts'),
    ],
  };
}

module.exports = {playwrightServer};
