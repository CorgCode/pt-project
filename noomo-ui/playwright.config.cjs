const { randomBytes } = require('node:crypto');
const { defineConfig } = require('@playwright/test');

const address = 'http://127.0.0.1:18845';
process.env.PT_LAB_TEST_PASSWORD ??= randomBytes(24).toString('hex');

module.exports = defineConfig({
  testDir: './tools/browser',
  outputDir: './test-results',
  workers: 1,
  use: { baseURL: address, browserName: 'chromium', channel: 'chromium', reducedMotion: 'reduce', trace: 'retain-on-failure' },
  webServer: {
    command: 'python3 tools/browser/fixture.py',
    url: `${address}/api/health`,
    timeout: 30_000,
  },
});
