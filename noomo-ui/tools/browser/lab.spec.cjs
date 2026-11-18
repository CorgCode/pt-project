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
  await expect(page.locator('#stat-hosts')).toHaveText('15');
  await page.getByText('Как провести разбор', { exact: true }).click();
  await page.getByRole('button', { name: 'К сравнению' }).click();
  await page.getByRole('button', { name: 'Открыть учебный разбор' }).click();
  const detail = page.locator('#change-detail');
  await expect(detail).toContainText('device-07');
  await expect(detail).toContainText('service-4');
  await page.locator('#quick-start summary').click();
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: testInfo.outputPath('comparison.png'), fullPage: true });

  await page.locator('nav [data-view="case"]').click();
  await page.locator('#operator-note').fill('Проверить обращение к service-4');
  await page.locator('#save-note').click();
  await expect(page.getByRole('dialog', { name: 'Вход в рабочее место' })).toBeVisible();
  await page.screenshot({ path: testInfo.outputPath('login.png') });
  await signIn(page);
  await expect(page.locator('#save-state')).toHaveText('Сохранено на компьютере');
  await page.reload();
  await expect(page.locator('#workspace-name')).toHaveValue('Учебный разбор: новый контакт');
  await expect(page.locator('#stat-hosts')).toHaveText('16');
  await page.locator('#host-search').fill('device-07');
  await page.locator('#host-list button').click();
  await page.locator('#to-case').click();
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
  await page.locator('#site-login').click();
  const signInButton = page.locator('#account-form').getByRole('button', { name: 'Войти', exact: true });
  await expect(signInButton).toBeInViewport({ ratio: 1 });
  await page.screenshot({ path: testInfo.outputPath('login-mobile.png'), fullPage: true });
  await signIn(page);
  const nav = page.getByRole('navigation', { name: 'Разделы программы' });
  for (const button of await nav.getByRole('button').all()) {
    const bounds = await button.boundingBox();
    expect(bounds.x).toBeGreaterThanOrEqual(0);
    expect(bounds.x + bounds.width).toBeLessThanOrEqual(391);
    expect(await button.evaluate(node => node.scrollWidth)).toBeLessThanOrEqual(await button.evaluate(node => node.clientWidth));
  }
  await page.locator('nav [data-view="changes"]').click();
  await page.getByRole('button', { name: 'Открыть учебный разбор' }).click();
  await expect(page.locator('#change-detail')).toContainText('service-4');
  await page.evaluate(() => window.scrollTo({ top: 0, behavior: 'instant' }));
  await page.screenshot({ path: testInfo.outputPath('comparison-mobile.png'), fullPage: true });
});
