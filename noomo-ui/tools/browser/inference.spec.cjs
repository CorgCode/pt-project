const { expect, test } = require('@playwright/test');

test('uploaded traffic is embedded, visualized and restored from PostgreSQL', async ({ page }) => {
  test.skip(!process.env.PT_TEST_DATABASE_URL, 'requires disposable PostgreSQL and the real inference service');
  await page.goto('/lab/');
  await page.getByLabel('Имя пользователя').fill('browser-test');
  await page.getByLabel('Пароль').fill(process.env.PT_LAB_TEST_PASSWORD);
  await page.getByRole('button', { name: 'Войти', exact: true }).click();
  await expect(page.locator('body')).toHaveAttribute('data-ml-enabled', 'true');
  const csv = 'src,dst,bytes,packets,proto,dst_port\nalpha,beta,800,10,tcp,443\nalpha,gamma,150,2,udp,53\nbeta,gamma,900,5,tcp,80\n';
  const result = page.waitForResponse(r => r.url().endsWith('/api/inference/embeddings'));
  await page.locator('#current-file').setInputFiles({
    name: 'observed.csv', mimeType: 'text/csv', buffer: Buffer.from(csv),
  });
  const response = await result;
  expect(response.status()).toBe(200);
  const prediction = await response.json();
  expect(prediction.schema).toBe('hostquery/1');
  expect(prediction.dataset.map(row => row.id)).toEqual(['alpha', 'beta', 'gamma']);
  expect(prediction.dataset[0].v.some(value => value > 0)).toBe(true);
  await expect(page.locator('#embedding-model')).toContainText('traffic-profile-v1');
  await expect(page.locator('#network')).toContainText('beta');
  await page.getByText('Действия с набором', { exact: true }).click();
  const saved = page.waitForResponse(r => r.url().endsWith('/api/workspaces') && r.request().method() === 'POST');
  await page.locator('#save-workspace').click();
  const saveResponse = await saved;
  expect(saveResponse.status()).toBe(201);
  const { id } = await saveResponse.json();
  const opened = page.waitForResponse(r => r.url().endsWith('/api/workspaces/' + id));
  await page.reload();
  const openResponse = await opened;
  expect((await openResponse.json()).inference.dataset).toEqual(prediction.dataset);
  await expect(page.locator('#embedding-model')).toContainText('traffic-profile-v1');
  await expect(page.locator('#network')).toContainText('beta');
});
