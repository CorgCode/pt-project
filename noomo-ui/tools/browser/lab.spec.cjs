const { expect, test } = require('@playwright/test');

async function signIn(page) {
  const dialog = page.getByRole('dialog', { name: 'Вход в рабочее место' });
  await dialog.getByLabel('Имя пользователя').fill('browser-test');
  await dialog.getByLabel('Пароль').fill(process.env.PT_LAB_TEST_PASSWORD);
  await dialog.getByRole('button', { name: 'Войти', exact: true }).click();
  await expect(dialog).toBeHidden();
  await expect(page.locator('body')).toHaveAttribute('data-workspace-mode', 'account');
  await expect(page).toHaveURL(/\/lab\/$/);
}

test('a visitor opens the lab, saves a two-period investigation, and signs out', async ({ page }, testInfo) => {
  const scriptErrors = [];
  page.on('pageerror', error => scriptErrors.push(error.message));
  await page.setViewportSize({ width: 1440, height: 1000 });
  await page.goto('/lab/');
  await expect(page.locator('body')).toHaveAttribute('data-workspace-mode', 'guest');
  await expect(page.locator('#stat-hosts')).toHaveText('16');
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
  const tutorial = page.locator('#tutorial-step');
  await expect(tutorial).toContainText('device-07');
  await page.locator('#host-search').fill('device-07');
  await page.locator('#host-list button').click();
  await expect(tutorial).toContainText('Реальные контакты');
  await page.locator('#mode-traffic').click();
  await expect(tutorial).toContainText('service-4');
  await page.locator('#to-case').click();
  await expect(tutorial).toContainText('Запишите наблюдение');
  await page.locator('#operator-note').fill('Проверить обращение к service-4');
  await expect(tutorial).toContainText('Разбор завершён');
  await page.getByRole('button', { name: 'Готово', exact: true }).click();
  await page.locator('nav [data-view="changes"]').click();
  const detail = page.locator('#change-detail');
  await expect(detail).toContainText('device-07');
  await expect(detail).toContainText('service-4');
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: testInfo.outputPath('comparison.png'), fullPage: true });

  await page.locator('nav [data-view="case"]').click();
  await page.locator('#operator-note').fill('Проверить обращение к service-4');
  await page.locator('#save-note').click();
  await expect(page.getByRole('dialog', { name: 'Вход в рабочее место' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('login.png') });
  await page.getByRole('button', { name: 'Создать аккаунт' }).click();
  const registration = page.getByRole('dialog', { name: 'Регистрация' });
  await registration.getByLabel('Имя пользователя').fill('new-analyst');
  await registration.getByLabel('Пароль').fill(process.env.PT_LAB_TEST_PASSWORD);
  await registration.getByRole('button', { name: 'Зарегистрироваться' }).click();
  await expect(registration).toBeHidden();
  await expect(page.locator('#save-state')).toHaveText('Сохранено на компьютере');
  await page.reload();
  await expect(page.locator('#tutorial')).toBeHidden();
  await expect(page.locator('#workspace-name')).toHaveValue('Учебный разбор: новый контакт');
  await expect(page.locator('#stat-hosts')).toHaveText('16');
  await page.locator('#host-search').fill('device-07');
  await page.locator('#host-list button').click();
  await page.locator('#to-case').click();
  await expect(page.locator('#view-title')).toBeFocused();
  await expect(page.locator('#operator-note')).toHaveValue('Проверить обращение к service-4');

  await page.getByRole('button', { name: 'Выйти' }).click();
  await expect(page).toHaveURL(/\/lab\/$/);
  await expect(page.locator('body')).toHaveAttribute('data-workspace-mode', 'guest');
  await expect(page.getByRole('button', { name: 'Войти', exact: true })).toBeVisible();
  await page.goto('/lab/workspace/');
  await expect(page).toHaveURL(/\/lab\/$/);
  expect(scriptErrors).toEqual([]);
});

test('mobile viewport keeps login and navigation reachable', async ({ page }, testInfo) => {
  await page.setViewportSize({ width: 390, height: 844 });
  await page.goto('/lab/');
  await page.screenshot({ path: testInfo.outputPath('devices-mobile.png') });
  await page.locator('#site-login').click();
  const signInButton = page.locator('#account-form').getByRole('button', { name: 'Войти', exact: true });
  await expect(signInButton).toBeInViewport({ ratio: 1 });
  await page.screenshot({ path: testInfo.outputPath('login-mobile.png'), fullPage: true });
  await signIn(page);
  const nav = page.getByRole('navigation', { name: 'Разделы программы' });
  const navBounds = await nav.boundingBox();
  expect(navBounds.y + navBounds.height).toBeCloseTo(844, 0);
  await nav.getByText('Ещё разделы').click();
  for (const button of await nav.getByRole('button').all()) {
    const bounds = await button.boundingBox();
    expect(bounds.height).toBeGreaterThanOrEqual(44);
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
