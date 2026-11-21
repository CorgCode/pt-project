const { expect, test } = require('@playwright/test');

async function signIn(page) {
  await page.getByLabel('Имя пользователя').fill('browser-test');
  await page.getByLabel('Пароль').fill(process.env.PT_LAB_TEST_PASSWORD);
  await page.getByRole('button', { name: 'Войти', exact: true }).click();
  await expect(page.locator('body')).toHaveAttribute('data-workspace-mode', 'account');
  await expect(page).toHaveURL(/\/lab\/$/);
}

test('a visitor opens the lab, saves a two-period investigation, and signs out', async ({ page }, testInfo) => {
  const scriptErrors = [];
  page.on('pageerror', error => scriptErrors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/lab/');
  await expect(page).toHaveURL(/\/lab\/login\/$/);
  await expect(page.locator('#host-search')).toHaveCount(0);
  await page.getByRole('link', { name: 'Создать аккаунт' }).click();
  await page.getByLabel('Имя пользователя').fill('new-analyst');
  await page.getByLabel('Пароль').fill(process.env.PT_LAB_TEST_PASSWORD);
  await page.getByRole('button', { name: 'Зарегистрироваться' }).click();
  await expect(page).toHaveURL(/\/lab\/$/);
  await expect(page.locator('#directory-count')).toHaveText('15');
  await expect(page.locator('#search-help')).toBeVisible();
  await expect(page.locator('#workspace-sidebar')).toBeVisible();
  const search = page.locator('#host-search');
  await search.focus();
  expect(await search.evaluate(input => getComputedStyle(input).outlineStyle)).toBe('none');
  expect(await page.locator('.search').evaluate(field => getComputedStyle(field).outlineStyle)).toBe('solid');
  await search.fill('Рабочий ПК 02');
  await page.locator('#host-list button').click();
  await expect(page.locator('#selected-id')).toHaveText('Рабочий ПК 02');
  await expect(page.locator('#selected-technical-id')).toHaveText('device-02');
  await page.locator('#mode-traffic').click();
  await expect(page.locator('#map-help')).toContainText('с кем общалось');
  await page.getByRole('button', { name: 'Скрыть подсказки' }).click();
  await expect(page.locator('#search-help')).toBeHidden();
  await page.locator('#host-search').clear();
  const actions = page.getByText('Действия с набором', { exact: true });
  await expect(page.getByRole('button', { name: 'Загрузить CSV' })).toBeHidden();
  await actions.click();
  await expect(page.getByRole('button', { name: 'Загрузить CSV' })).toBeVisible();
  await page.getByRole('button', { name: 'Загрузить CSV' }).press('Escape');
  await expect(actions).toBeFocused();
  await page.getByRole('button', { name: 'Свернуть меню' }).click();
  await expect(page.locator('#workspace-sidebar')).toBeHidden();
  await page.getByRole('button', { name: 'Показать меню' }).click();
  await expect(page.locator('#workspace-sidebar')).toBeVisible();
  await expect(page.locator('#host-search')).toBeVisible();
  await expect(page.locator('nav [data-view="groups"]')).toBeHidden();
  await expect(page.locator('#workspace-name')).toBeHidden();
  await expect(page.locator('#group-filter')).toBeHidden();
  await page.getByText('Фильтры и сортировка', { exact: true }).click();
  await expect(page.locator('#group-filter')).toBeVisible();
  await page.getByText('Фильтры и сортировка', { exact: true }).click();
  await page.screenshot({ path: testInfo.outputPath('devices.png') });
  await page.locator('nav [data-view="changes"]').click();
  await page.getByRole('button', { name: 'Открыть учебный разбор' }).click();
  const detail = page.locator('#change-detail');
  await expect(detail).toContainText('device-07');
  await expect(detail).toContainText('service-4');
  await expect(page.locator('.incident-marker')).toContainText('10:42 UTC');
  await expect(page.locator('#incident-timeline li')).toHaveCount(6);
  await expect(page.locator('#incident-timeline .incident-start')).toContainText('10:40');
  await expect(page.locator('#change-list .change-row').first()).toContainText('device-07');
  await detail.getByRole('button', { name: 'Проверить контакты device-07' }).click();
  await expect(page.locator('#contact-table-body')).toContainText('service-4');
  await page.locator('nav [data-view="changes"]').click();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: testInfo.outputPath('comparison.png'), fullPage: true });

  await page.locator('nav [data-view="case"]').click();
  await page.locator('#operator-note').fill('Проверить обращение к service-4');
  await page.locator('#save-note').click();
  await expect(page.locator('#save-state')).toHaveText('Сохранено на компьютере');
  await page.reload();
  await expect(page.locator('#search-help')).toBeHidden();
  await expect(page.locator('#workspace-name')).toHaveValue('Учебный инцидент: новый контакт');
  await expect(page.locator('#directory-count')).toHaveText('16');
  await page.locator('#host-search').fill('device-07');
  await page.locator('#host-list button').click();
  await page.locator('#to-case').click();
  await expect(page.locator('#view-title')).toBeFocused();
  await expect(page.locator('#operator-note')).toHaveValue('Проверить обращение к service-4');

  await page.getByRole('button', { name: 'Выйти' }).click();
  await expect(page).toHaveURL(/\/lab\/login\/$/);
  await expect(page.locator('#host-search')).toHaveCount(0);
  await expect(page.getByRole('button', { name: 'Войти', exact: true })).toBeVisible();
  await page.goto('/lab/workspace/');
  await expect(page).toHaveURL(/\/lab\/login\/$/);
  expect(scriptErrors).toEqual([]);
});

test('mobile viewport keeps login and navigation reachable', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/lab/');
  await page.screenshot({ path: testInfo.outputPath('devices-mobile.png') });
  const signInButton = page.getByRole('button', { name: 'Войти', exact: true });
  await expect(signInButton).toBeInViewport({ ratio: 1 });
  await page.screenshot({ path: testInfo.outputPath('login-mobile.png'), fullPage: true });
  await signIn(page);
  await expect(page.locator('#search-help')).toBeVisible();
  await page.locator('#host-search').fill('device-02');
  await page.locator('#host-list button').click();
  await page.locator('#mode-traffic').click();
  await expect(page.locator('#map-help')).toBeVisible();
  await page.getByRole('button', { name: 'Скрыть подсказки' }).click();
  const nav = page.getByRole('navigation', { name: 'Разделы программы' });
  await nav.getByText('Ещё разделы').click();
  for (const button of await nav.getByRole('button').all()) {
    const bounds = await button.boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(391);
    expect(await button.evaluate(node => node.scrollWidth)).toBeLessThanOrEqual(await button.evaluate(node => node.clientWidth));
  }
  await nav.getByText('Ещё разделы').click();
  await page.locator('nav [data-view="changes"]').click();
  await page.getByRole('button', { name: 'Открыть учебный разбор' }).click();
  await expect(page.locator('#change-detail')).toContainText('service-4');
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: testInfo.outputPath('comparison-mobile.png'), fullPage: true });
});

test('contacts, groups, watchlist and reports work with the demo', async ({ page }) => {
  await page.goto('/lab/');
  await signIn(page);
  await page.getByText('Ещё действия', { exact: true }).click();
  await page.locator('#toggle-watch').click();
  const nav = page.getByRole('navigation', { name: 'Разделы программы' });
  await nav.getByText('Ещё разделы').click();
  await nav.locator('[data-view="watch"]').click();
  await expect(page.locator('#watch-list')).toContainText('device-01');

  await nav.locator('[data-view="groups"]').click();
  await expect(page.locator('#group-list button').first()).toBeVisible();
  await nav.locator('[data-view="contacts"]').click();
  await expect(page.locator('#contact-table-body tr').first()).toBeVisible();
  const csv = page.waitForEvent('download');
  await page.locator('#export-contacts').click();
  expect((await csv).suggestedFilename()).toMatch(/\.csv$/);

  await nav.locator('[data-view="case"]').click();
  const report = page.waitForEvent('download');
  await page.locator('#export-readable').click();
  expect((await report).suggestedFilename()).toMatch(/\.html$/);
});
