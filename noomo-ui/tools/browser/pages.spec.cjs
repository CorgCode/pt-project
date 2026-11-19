const { expect, test } = require('@playwright/test');
test.use({ launchOptions: { args: ['--use-gl=angle', '--use-angle=swiftshader', '--enable-unsafe-swiftshader'] } });

function watchErrors(page) {
  const errors = [];
  page.on('pageerror', error => errors.push(error.message));
  page.on('response', response => {
    if (response.status() >= 400) errors.push(`${response.status()} ${response.url()}`);
  });
  return errors;
}

for (const width of [320, 1440]) {
  test(`model and workspace work at ${width}px`, async ({ page }, testInfo) => {
    const errors = watchErrors(page);
    await page.setViewportSize({ width, height: 900 });
    await page.emulateMedia({ reducedMotion: 'reduce' });
    await page.goto('/model/');
    const contents = page.getByRole('navigation', { name: 'На этой странице' });
    await expect(contents.getByRole('link')).toHaveCount(11);
    await contents.getByRole('link', { name: 'Результаты на LANL' }).click();
    await expect(page).toHaveURL(/#results$/);
    await expect(page.locator('#results h2')).toBeInViewport();
    await page.screenshot({ path: testInfo.outputPath('model-results.png') });
    for (const image of await page.locator('.model-figure img, .source-figure img').all()) {
      await image.evaluate(node => node.loading = 'eager');
      await expect.poll(() => image.evaluate(node => node.naturalWidth)).toBeGreaterThan(0);
    }
    await page.locator('header a[href="/lab/"]').click();
    await expect(page).toHaveURL(/\/lab\/$/);
    await expect(page.locator('#stat-hosts')).toHaveText('15');
    await page.locator('nav [data-view="contacts"]').click();
    await expect(page.locator('#contacts-title')).toBeVisible();
    await page.locator('nav [data-view="changes"]').click();
    await page.locator('#change-scenario').click();
    await expect(page.locator('#change-detail')).toContainText('service-4');
    const viewportWidth = await page.evaluate(() => document.documentElement.scrollWidth);
    expect(viewportWidth).toBeLessThanOrEqual(width);
    expect(errors).toEqual([]);
  });
}

test('the model page keeps its reading position while diagrams load', async ({ page }) => {
  await page.setViewportSize({ width: 1440, height: 900 });
  await page.addInitScript(() => {
    window.scrollResets = 0;
    const scrollTo = window.scrollTo.bind(window);
    window.scrollTo = (...args) => {
      const top = typeof args[0] === 'object' ? args[0]?.top : args[1];
      if (top === 0 && window.scrollY > 100) window.scrollResets++;
      return scrollTo(...args);
    };
  });
  await page.goto('/model/');
  const diagram = page.locator('.model-figure img[loading="lazy"]').last();
  await page.mouse.wheel(0, 2200);
  await expect.poll(() => page.evaluate(() => window.scrollY)).toBeGreaterThan(1800);
  const readingPosition = await page.evaluate(() => window.scrollY);
  await diagram.evaluate(async (image) => {
    image.loading = 'eager';
    await image.decode();
    // Let any layout work triggered by the image finish.
    await new Promise(resolve => setTimeout(resolve, 250));
  });
  expect(await page.evaluate(() => window.scrollResets)).toBe(0);
  expect(await page.evaluate(() => window.scrollY)).toBeGreaterThanOrEqual(readingPosition);
  const contents = page.getByRole('navigation', { name: 'На этой странице' });
  await contents.getByRole('link', { name: 'Результаты на LANL' }).click();
  await expect(page).toHaveURL(/#results$/);
  await expect(page.locator('#results h2')).toBeInViewport();
});

test('legacy routes point to the current content and preserve useful deep links', async ({ page }) => {
  await page.emulateMedia({ reducedMotion: 'reduce' });
  for (const [old, next] of [
    ['/work/', /\/lab\/$/],
    ['/contact/', /\/model\/$/],
    ['/product/', url => url.pathname === '/' && !url.hash],
    ['/product/#results', /\/model\/#results$/],
  ]) {
    await page.goto(old);
    await expect(page).toHaveURL(next);
  }
  await expect(page.locator('#results h2')).toBeInViewport();
});

test('mobile chart labels fit inside their SVG view boxes', async ({ page }) => {
  for (const name of ['score-comparison-mobile.svg', 'recency-mobile.svg']) {
    await page.goto(`/model/assets/${name}`);
    const clipped = await page.evaluate(() => {
      const svg = document.documentElement;
      const view = svg.viewBox.baseVal;
      const inverse = svg.getCTM().inverse();
      return [...svg.querySelectorAll('g[id^="text_"]')].filter(group => {
        const { x, y, width, height } = group.getBBox();
        const matrix = inverse.multiply(group.getCTM());
        return [new DOMPoint(x, y), new DOMPoint(x + width, y), new DOMPoint(x, y + height), new DOMPoint(x + width, y + height)]
          .map(point => point.matrixTransform(matrix))
          .some(point => point.x < view.x - 0.6 || point.x > view.x + view.width + 0.6 ||
            point.y < view.y - 0.6 || point.y > view.y + view.height + 0.6);
      }).map(group => group.id);
    });
    expect(clipped, name).toEqual([]);
  }
});

test.describe('native WebGL scene', () => {
  test('a click starts the scene and all six chapters reach the lab link', async ({ page }, testInfo) => {
    test.setTimeout(90_000);
    const errors = watchErrors(page);
    await page.setViewportSize({ width: 1100, height: 730 });
    await page.goto('/');
    const start = page.getByRole('button', { name: 'Начать исследование сети' });
    await expect(start).toBeVisible({ timeout: 45_000 });
    await expect(page.getByRole('link', { name: 'Без 3D' })).toHaveCount(0);
    await start.click();
    const enter = page.getByRole('button', { name: 'К сценам' });
    await expect(enter).toBeVisible({ timeout: 20_000 });
    await enter.click();
    await expect(enter).toHaveCount(0);
    const active = page.locator('.pt-product-copy:not([hidden])');
    const chapters = [
      [2000, 'Понять, что происходит в сети'],
      [4000, 'Поведение видно по связям'],
      [6000, 'От одного устройства — к похожим'],
      [8000, 'Контекст для оператора SOC'],
      [10000, 'Попробуйте на своих данных'],
      [11650, 'Откройте рабочее место'],
    ];
    for (const [y, title] of chapters) {
      // A scene transition briefly locks scroll. Retry until its scroll controller releases the page.
      await expect(async () => {
        await page.evaluate(y => window.scrollTo(0, y), y);
        await expect(active).toContainText(title, { timeout: 2500 });
      }).toPass({ timeout: 12_000 });
      await expect(active).toHaveCount(1);
    }
    const lab = page.locator('.pt-end-actions a[href="/lab/"]');
    await expect(lab).toBeVisible();
    await page.screenshot({ path: testInfo.outputPath('scene-final.png') });
    await page.evaluate(() => window.scrollTo(0, 2000));
    await expect(lab).toBeHidden();
    await expect(active).toContainText(chapters[0][1]);
    await page.evaluate(() => window.scrollTo(0, 11650));
    await lab.click();
    await expect(page).toHaveURL(/\/lab\/$/);
    expect(errors).toEqual([]);
  });
  test('the start button responds to a touch tap', async ({ browser, baseURL }) => {
    test.setTimeout(70_000);
    const page = await browser.newPage({
      baseURL, viewport: { width: 390, height: 844 }, isMobile: true, hasTouch: true,
      reducedMotion: 'reduce',
    });
    try {
      await page.goto('/');
      const start = page.getByRole('button', { name: 'Начать исследование сети' });
      await expect(start).toBeVisible({ timeout: 45_000 });
      await start.tap();
      await expect(page.getByRole('button', { name: 'К сценам' })).toBeVisible({ timeout: 20_000 });
    } finally {
      await page.close();
    }
  });
});
