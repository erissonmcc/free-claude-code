const path = require('node:path');

async function playwrightServer({home, sessionId, createConnection, chromium}) {
  if (typeof sessionId !== 'string' || !/^[a-zA-Z0-9_-]+$/.test(sessionId)) {
    throw new Error('Invalid session ID for the Playwright browser profile.');
  }
  const profile = path.join(home, '.fcc', 'playwright', sessionId);
  let context;
  const instance = await createConnection({
    capabilities: ['core', 'vision'],
    outputDir: path.join(profile, 'artifacts'),
  }, async () => {
    if (!context) {
      context = await chromium.launchPersistentContext(profile, {
        headless: true,
        args: ['--no-sandbox', '--disable-dev-shm-usage', '--no-zygote', '--single-process', '--disable-gpu'],
      });
      context.on('close', () => { context = undefined; });
    }
    return context;
  });
  return {
    config: {type: 'sdk', name: 'playwright', instance},
    close: async () => {
      if (context) await context.close();
      await instance.close();
    },
  };
}

module.exports = {playwrightServer};
