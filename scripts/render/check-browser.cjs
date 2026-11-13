// Fail deployment before traffic switches if Chromium cannot run in this image.
const {chromium} = require('playwright');
(async () => {
  const browser = await chromium.launch({headless:true, args:['--no-sandbox', '--disable-dev-shm-usage']});
  try {
    const page = await browser.newPage();
    await page.goto('data:text/html,<title>FCC Playwright ready</title>');
    if (await page.title() !== 'FCC Playwright ready') throw new Error('Chromium verification failed.');
    console.log('Chromium launched and rendered successfully.');
  } finally {await browser.close();}
})().catch(error => {console.error(error); process.exitCode = 1;});
