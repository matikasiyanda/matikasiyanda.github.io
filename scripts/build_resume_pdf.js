// Render /resume/ to assets/files/siyanda-matika-resume.pdf using the page's print styles.
// Usage: serve the built site on http://127.0.0.1:4077, then
//   NODE_PATH=<dir with puppeteer-core> node scripts/build_resume_pdf.js [site-url] [chrome-path]
const puppeteer = require('puppeteer-core');
const path = require('path');
const site = process.argv[2] || 'http://127.0.0.1:4077';
const chrome = process.argv[3] || '/usr/bin/google-chrome';
(async () => {
  const browser = await puppeteer.launch({ executablePath: chrome, args: ['--no-sandbox'], headless: 'new' });
  const page = await browser.newPage();
  await page.goto(site + '/resume/', { waitUntil: 'networkidle0' });
  await page.emulateMediaType('print');
  await page.evaluateHandle('document.fonts.ready');
  const out = path.join(__dirname, '..', 'assets', 'files', 'siyanda-matika-resume.pdf');
  await page.pdf({ path: out, format: 'A4', printBackground: false, preferCSSPageSize: true, displayHeaderFooter: false, tagged: true });
  await browser.close();
  console.log('wrote', out);
})();
