/** Real browser-to-AASIST smoke test. Supply an authorized recording path.
 * No mocked API, accuracy claim, or retained recording is produced.
 * npm install --prefix .cache/browser playwright-core@1.56.1
 * Set VOXGUARD_PLAYWRIGHT_MODULE to its absolute package directory if needed.
 * node tests/browser.mjs <recording> [base-url] [screenshot-output]
 */
import { createRequire } from 'node:module';
import assert from 'node:assert/strict';
import { existsSync, mkdirSync } from 'node:fs';
import { dirname, resolve } from 'node:path';
const require = createRequire(import.meta.url);
const { chromium } = require(process.env.VOXGUARD_PLAYWRIGHT_MODULE || 'playwright-core');
const [recording, base = 'http://127.0.0.1:5173', screenshot = 'artifacts/browser-smoke.png'] = process.argv.slice(2);
assert(recording && existsSync(recording), 'Supply an existing, authorized audio recording');
const executablePath = process.env.VOXGUARD_CHROMIUM || undefined;
const browser = await chromium.launch({ headless: true, executablePath, args: ['--no-sandbox'] });
try {
  const page = await browser.newPage({ viewport: { width: 1440, height: 1000 } });
  await page.goto(base, { waitUntil: 'networkidle' });
  await page.locator('input[type=file]').setInputFiles({ name: 'bad.txt', mimeType: 'text/plain', buffer: Buffer.from('not audio') });
  await page.getByRole('alert').filter({ hasText: 'Choose a WAV' }).waitFor();
  await page.locator('input[type=file]').setInputFiles(resolve(recording));
  const responsePromise = page.waitForResponse(r => r.url().endsWith('/api/analyze') && r.request().method() === 'POST', { timeout: 180000 });
  await page.getByRole('button', { name: /^Analyze recording/ }).click();
  const response = await responsePromise;
  assert.equal(response.status(), 200, await response.text());
  const result = await response.json();
  assert(['human', 'ai', 'uncertain'].includes(result.classification));
  assert(Number.isFinite(result.score) && result.score >= 0 && result.score <= 1);
  assert.equal(result.model.loaded, true);
  assert.equal(result.model.name, 'AASIST');
  assert(result.segments.length > 0);
  await page.locator('.score-row strong').filter({ hasText: result.score.toFixed(3) }).waitFor();
  assert.equal(await page.getByLabel('Uploaded recording playback').count(), 1);
  const download = page.waitForEvent('download');
  await page.getByRole('button', { name: 'Download analysis JSON' }).click();
  assert((await download).suggestedFilename().endsWith('.json'));
  mkdirSync(dirname(resolve(screenshot)), { recursive: true });
  await page.screenshot({ path: screenshot, fullPage: true });
  await page.setViewportSize({ width: 390, height: 844 });
  assert(await page.evaluate(() => document.documentElement.scrollWidth <= window.innerWidth + 1), 'Mobile layout overflows horizontally');
  console.log(JSON.stringify({ real_browser_upload: 'passed', model: result.model.name, classification: result.classification, score: result.score, duration_seconds: result.duration_seconds, inference_latency_ms: result.latency_ms, download: 'passed', mobile_layout: 'passed', screenshot }, null, 2));
} finally {
  await browser.close();
}
