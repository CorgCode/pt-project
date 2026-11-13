<!-- Hallmark · macrostructure: separate-model-and-lab · theme: violet-glass · genre: atmospheric
pre-emit critique: P3 H3 E3 S4 R3 V3 · slop test: visual design rejected; redesign required -->

# Черновики отдельных страниц

Ранние варианты `/model/` и `/lab/` были отклонены пользователем. `/lab/` остаётся черновиком. `/model/` переделана на обычный шаблон кейса Noomo: оригинальные стили и фон, видео, описание модели и таблицы. Собственная 3D-сцена отключена. Главная не менялась. Визуальная проверка WebGL в браузере остаётся ограниченной разрешениями Computer Use.

Основная страница и её модули восстановлены из `d9d3b61`. Не подключать к ней `pt-pages.css`, `pt-navigation.js` и новые страницы до доработки их визуального решения.

Черновики используют отдельные `pt-page-content.js`, `pt-page-lab.js`, `pt-page-world.js`; их изменение не затрагивает рабочую сцену. `pt-page.js` создаёт самостоятельный renderer. Import map подставляет `pt-engine.js` вместо Nuxt-библиотеки, чтобы не запускать исходное приложение на отдельных страницах.

Лаборатория обрабатывает CSV и сравнивает профили локально, поддерживает 1024 признака. Она не запускает обученную модель. Проверки: `node tools/check-pages.cjs` после `npm ci`. Эти проверки подтверждают состояние, расчёты и геометрию, а не визуальное качество в браузере.

## Preview

- Genre: atmospheric.
- Structure: main scene / annotated model / laboratory.
- Theme: violet glass, локальные Onest и Oswald.
- Visual review: отклонено пользователем; требуется переделка.
- Ограничение проверки: управление Chrome недоступно из-за отсутствия разрешений Computer Use.
- Источник описания модели: `scoreboard/submissions/multiday_codebook/README.md` и `multiday_codebook_plus/README.md`.

Локальные Three.js и RGBELoader перенесены из `ui-prototype/vendor` вместе с MIT-лицензией. Chart.js уже находится в `vendor` с собственной лицензией. Происхождение исходной главной страницы описано в основном README.
