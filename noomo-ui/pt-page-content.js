/* Hallmark · macrostructure: scene-to-field-guide · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: structural checks pass; visual review pending
 */
(() => {
  const content = `
  <header class="pt-guide-intro">
    <p>МФТИ × Positive Technologies</p>
    <h2>От сетевых записей —<br>к понятной картине.</h2>
    <p>Дальше — весь продукт: какую задачу решаем, как устроена наша модель, чему её учим и что уже можно попробовать. Без подготовки в машинном обучении.</p>
    <nav aria-label="Подробно о продукте"><a href="#pt-detail-problem">Задача</a><a href="#pt-detail-model">Модель</a><a href="#pt-detail-results">Результаты</a><a href="#pt-detail-demo">Попробовать</a></nav>
  </header>
  <section id="pt-detail-problem" class="pt-guide-section">
    <h2>Адрес виден.<br>Поведение — ещё нет.</h2>
    <div class="pt-guide-prose"><p>В корпоративной сети общаются рабочие компьютеры, серверы и сервисы. Запись соединения показывает, кто кому передал данные. Но один IP-адрес не объясняет, что это за устройство и привычно ли оно себя ведёт.</p><p>Хост — просто устройство в сети. SOC — команда, которая следит за её безопасностью. Наш продукт помогает специалисту SOC исследовать хост через его взаимодействия: с кем он общается, какими сервисами пользуется, сколько передаёт данных и когда активен.</p></div>
    <figure class="pt-flow-figure"><div><strong>Сетевые записи</strong><span>Адреса, время, порты, объёмы</span></div><b aria-hidden="true">→</b><div><strong>Профиль поведения</strong><span>Числовое описание хоста</span></div><b aria-hidden="true">→</b><div><strong>Понятное сравнение</strong><span>Похожие устройства и различия</span></div><figcaption>Мы исследуем сетевые характеристики. Реальные роли устройств заранее не размечены.</figcaption></figure>
  </section>
  <section id="pt-detail-product" class="pt-guide-section">
    <h2>Что оператор<br>сможет сделать.</h2>
    <div class="pt-product-journey">
      <article><span>Выбрать устройство</span><h3>«На кого оно похоже?»</h3><p>Получить ближайшие профили и перейти к их исследованию. Вместо просмотра всех устройств вручную — начать с короткого списка соседей.</p><small>Поиск по векторам работает в демонстрации ниже.</small></article>
      <article><span>Посмотреть окружение</span><h3>«Что объединяет эту группу?»</h3><p>Собрать хосты с похожим поведением и изучить общие сервисы, собеседников и режим активности. Группа помогает сформулировать гипотезу о назначении устройств.</p><small>Кластеризация и объяснение соседства — следующие этапы.</small></article>
      <article><span>Сравнить во времени</span><h3>«Почему поведение изменилось?»</h3><p>Заметить необычный переход хоста из одной группы в другую. Например, устройство стало общаться с новым окружением. Это повод проверить изменение, а не готовый вывод об атаке.</p><small>Отслеживание переходов — следующий этап сервиса.</small></article>
    </div>
  </section>
  <section id="pt-detail-model" class="pt-guide-section">
    <h2>Как устроена<br>наша модель.</h2>
    <div class="pt-guide-prose"><p>Представьте профиль слушателя в музыкальном сервисе. Чтобы найти похожие вкусы, можно сравнивать не имена людей, а то, что они слушают. Здесь похожая идея: вместо музыки — сетевые привычки компьютера.</p><p>Модель превращает эти привычки в <strong>вектор — список чисел</strong>. Его ещё называют embedding, или векторным представлением. Отдельное число обычно не означает «сервер» или «почтовый клиент». Полезен весь профиль целиком: по нему можно измерить, насколько близки два хоста.</p></div>
    <h3>Сначала собираем признаки.</h3>
    <div class="pt-feature-ledger"><div><strong>С кем общается</strong><p>Какие устройства находятся в сетевом окружении хоста и как часто повторяются контакты.</p></div><div><strong>Как общается</strong><p>Порты, протоколы и направления соединений: отправляет запросы или принимает их.</p></div><div><strong>Сколько передаёт</strong><p>Число соединений, пакетов и байтов. Единичный контакт отличается от регулярного обмена.</p></div><div><strong>Когда активен</strong><p>Признаки собираются по временным окнам, чтобы описывать поведение за период, а не одну запись.</p></div></div>
    <figure class="pt-model-diagram"><div class="pt-model-input"><span>Записи LANL</span><strong>Окно трафика</strong><span>Признаки хоста</span></div><span class="pt-model-arrow" aria-hidden="true">↓</span><div class="pt-model-core"><strong>Кодировщик трафика</strong><p>Нейросеть переводит признаки в общее пространство профилей.</p></div><span class="pt-model-arrow" aria-hidden="true">↓</span><div class="pt-vector-ribbon" aria-label="Схематический вектор"><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><i></i><span>896 чисел</span></div><figcaption>Схема объясняет путь данных. Полоски иллюстрируют вектор, а не реальные значения и не точное расположение слоёв сети.</figcaption></figure>
    <h3>Почему одного окна недостаточно?</h3>
    <div class="pt-guide-prose"><p>В отдельный час устройство может почти молчать или выполнять непривычную задачу. По одному такому фрагменту легко получить неполный портрет. Поэтому многодневная модель использует историю: для уже известных хостов дообучаются отдельные профили, которые собирают информацию из нескольких дней.</p><p>Чтобы сопоставить с ними более позднее окно трафика, используется адаптер Ridge — обучаемое линейное преобразование. Простыми словами, он приводит описание текущей активности к тому же «языку», на котором записана история хоста.</p></div>
    <h3>Многодневная модель: три части.</h3>
    <p class="pt-guide-prose">Исследовательское название — multiday_codebook_plus. Нажмите на часть профиля, чтобы увидеть, зачем она нужна.</p>
    <div class="pt-model-tabs" role="group" aria-label="Состав вектора"><button data-model-part="history" aria-pressed="true"><b>896</b><span>История хоста</span></button><button data-model-part="contacts" aria-pressed="false"><b>64</b><span>Новые контакты</span></button><button data-model-part="graph" aria-pressed="false"><b>64</b><span>Структура сети</span></button></div>
    <div class="pt-model-explanation" aria-live="polite"><h4 id="pt-model-part-title">Устойчивый портрет по истории</h4><p id="pt-model-part-copy">Основные 896 чисел — профиль известного хоста и адаптация позднего трафика к этому пространству. Эта часть помогает узнавать устройство по поведению в разные дни.</p></div>
    <figure class="pt-vector-total"><div><span class="pt-vector-history">История · 896</span><span>Контакты · 64</span><span>Граф · 64</span></div><figcaption><strong>1 024 числа</strong> — итоговый профиль хоста. Части объединяются, чтобы сравнение учитывало несколько сторон поведения.</figcaption></figure>
    <h3>Почему выбрали такую архитектуру?</h3>
    <div class="pt-guide-prose"><p>История помогает меньше зависеть от случайного часа активности. Отдельный блок контактов добавляет сигнал для задачи новых собеседников. Графовая часть описывает место хоста в структуре связей: важны не только объёмы трафика, но и окружение.</p><p>Это рабочая исследовательская гипотеза, которую мы проверяем сравнением моделей. Улучшение нельзя полностью приписать архитектуре: многодневный подход также использует больше истории, чем исходный кодировщик.</p></div>
  </section>
  <section id="pt-detail-training" class="pt-guide-section">
    <h2>Как модель учится<br>без подписанных ролей.</h2>
    <div class="pt-guide-prose"><p>В данных нет готового ответа «это бухгалтерский компьютер, а это сервер базы данных». Поэтому проверяем более конкретные задачи, ответ на которые можно получить из самих сетевых записей.</p><p>Можно ли узнать тот же хост в другой день? Отличить будущую связь от отсутствующей? Поставить нового собеседника выше в списке кандидатов? По этим сигналам сравниваем представления, даже когда реальные роли устройств неизвестны.</p></div>
    <ol class="pt-training-steps"><li><strong>Базовый кодировщик</strong><p>Начальный кодировщик обучается на 32 часовых окнах из дней 02–09. Он получает сетевые признаки и учится строить сопоставимые числовые представления.</p></li><li><strong>История известных хостов</strong><p>Многодневные профили обучаются по истории дней 03–12. Это отдельные обучаемые представления известных устройств, а не просто новые названия их адресов.</p></li><li><strong>Дополнительные сигналы</strong><p>К профилю добавляются блок новых собеседников и графовый блок. Затем позднее окно сопоставляется с профилем через адаптер.</p></li></ol>
    <figure class="pt-training-split"><div><strong>02–12</strong><span>Обучение</span></div><div><strong>13</strong><span>Подбор настроек</span></div><div><strong>14–16</strong><span>Финальная проверка</span></div><figcaption>Номера дней в эксперименте. Будущие тестовые окна не используются для обучения и выбора настроек.</figcaption></figure>
    <p class="pt-guide-prose">Разделение по времени ближе к задаче оператора: сегодня у нас есть история, а завтра придут новые записи. При этом результаты на известных хостах не доказывают такое же качество для совершенно нового устройства — это нужно проверять отдельно.</p>
  </section>
  <section id="pt-detail-results" class="pt-guide-section">
    <h2>Что получилось.<br>И как читать цифры.</h2>
    <p class="pt-guide-prose">Сравнение из таблицы экспериментов от 4 ноября 2026 года. Общий балл объединяет несколько задач. Он нужен для сравнения подходов и не означает процент обнаруженных атак.</p>
    <figure class="pt-score-comparison"><div><span>Временной кодировщик</span><i style="--score:56.5%"></i><b>56,5</b></div><div><span>Многодневные профили</span><i style="--score:64.9%"></i><b>64,9</b></div><div><span>Профили + контакты + граф</span><i style="--score:70%"></i><b>70,0</b></div><figcaption>Баллы из 100 на общих тестовых окнах. Для последней модели: интервал оценки 65,6–73,4.</figcaption></figure>
    <div class="pt-metric-ledger"><article><strong>59,6%</strong><h3>Узнать тот же хост</h3><p>В среднем по тестовым окнам правильное устройство оказалось первым кандидатом. Проверяем, сохраняется ли его узнаваемый профиль в другой день.</p><small>Метрика: Identity Top-1</small></article><article><strong>0,974</strong><h3>Предсказать связь</h3><p>Показывает, насколько хорошо модель различает пары с будущим взаимодействием и без него. Значение ближе к 1 лучше; это не доля верных ответов.</p><small>Метрика: ROC-AUC</small></article><article><strong>0,318</strong><h3>Найти нового собеседника</h3><p>Оценивает место верного нового контакта в списке. Чем раньше он появляется, тем выше балл. Здесь ещё есть пространство для улучшения.</p><small>Метрика: MRR</small></article></div>
    <aside class="pt-guide-note"><h3>Что эти результаты пока не доказывают</h3><p>У нас нет разметки реальных ролей, поэтому мы не заявляем точность определения «серверов» и «рабочих станций». Поиск похожих хостов также не является детектором атак. Следующий шаг — проверять полезность групп и объяснений вместе с оператором.</p></aside>
  </section>
  <section id="pt-detail-demo" class="pt-guide-section pt-full-lab">
    <h2>Теперь попробуйте.<br>Прямо на этой странице.</h2>
    <p class="pt-guide-prose">Выберите набор данных, затем хост. Карта покажет его положение среди других профилей, а график — пять ближайших соседей. Нажмите на точку или соседа, чтобы продолжить исследование.</p>
    <div class="pt-lab-guide"><span><b>1</b> Выберите данные</span><span><b>2</b> Нажмите на хост</span><span><b>3</b> Сравните соседей</span></div>
    <div class="pt-lab-controls"><div class="lab-presets" role="group" aria-label="Набор данных"><button data-preset="project" aria-pressed="true">300 хостов</button><button data-preset="small" aria-pressed="false">24 хоста</button><button data-preset="flows" aria-pressed="false">Пример трафика</button></div><label id="upload-zone" class="pt-upload"><input id="csv-file" type="file" accept=".csv,text/csv"><strong>↑ Загрузить свой CSV</strong><span>Или перетащите файл сюда · до 3 МБ</span></label><details><summary>Формат файла и пример</summary><p>Сетевые записи: src, dst; дополнительно bytes, packets, dst_port, proto. До 10 000 строк и 500 хостов. По ним строится быстрый профиль из 20 признаков.</p><p>Готовые векторы: host_id, emb_0, emb_1… До 500 хостов и 1024 признаков. Для косинусного сравнения нужны ненулевые векторы.</p><button id="download-sample">Скачать пример CSV ↓</button></details><p id="lab-status" role="status" aria-live="polite"></p></div>
    <div class="pt-lab-workspace"><div class="pt-lab-selection"><label for="lab-host">Выбранный хост<select id="lab-host"></select></label><span id="lab-mode"></span></div><div class="pt-lab-visuals"><figure><figcaption><strong>Карта профилей</strong><span>Нажмите на точку — выберите устройство</span></figcaption><div class="pt-chart-wrap"><canvas id="lab-map-chart" role="img" aria-label="Карта профилей хостов"></canvas></div><p>Карта — упрощённая проекция на плоскость. Точное сходство вычисляется по всем признакам, а не по расстоянию на рисунке.</p></figure><figure><figcaption><strong>Пять ближайших соседей</strong><span>Сходство ближе к 1 — профили ближе</span></figcaption><div class="pt-chart-wrap"><canvas id="lab-neighbors-chart" role="img" aria-label="Косинусное сходство ближайших хостов"></canvas></div><div id="lab-neighbor-buttons"></div></figure></div><div class="pt-lab-observation"><div><h3 id="lab-insight-title"></h3><p id="lab-insight-copy" aria-live="polite"></p></div><div id="lab-summary"></div></div></div>
    <aside class="pt-guide-note"><h3>Что именно работает в этой демонстрации</h3><p>Сходство и графики рассчитываются в браузере. Набор из 300 хостов содержит тестовые поведенческие векторы проекта. Загрузка трафика строит упрощённые признаки; она не запускает и не обучает multiday_codebook_plus. Файл остаётся на вашем устройстве.</p><p>Кластеризация, объяснение причин соседства и отслеживание переходов пока не реализованы. Мы показываем текущий поиск и отдельно объясняем дальнейшую цель продукта.</p></aside>
  </section>`;
  window.__ptMountContent = function () {
    if (document.getElementById("pt-guide")) return;
    // Wait until Vue has hydrated the retained scene before adding siblings.
    const nuxt = document.getElementById("__nuxt");
    if (nuxt && !nuxt.__vue_app__) return;
    const footer = document.querySelector("#smooth-content .home-footer") || document.getElementById("pt-page-source");
    if (!footer) return;
    const guide = document.createElement("main");
    guide.id = "pt-guide";
    guide.className = "pt-guide pt-scene-source";
    guide.setAttribute("aria-label", "Полное текстовое описание продукта");
    guide.innerHTML = content;
    guide.setAttribute("aria-hidden","true");
    guide.querySelectorAll("a,button,input,select,summary").forEach(el=>el.tabIndex=-1);
    footer.before(guide);
    guide.addEventListener("click", (event) => {
      const link = event.target.closest('a[href^="#pt-detail-"]');
      if (link) {
        event.preventDefault();
        window.__ptOpenProduct("#" + link.hash.replace("#pt-detail-", ""));
      }
      event.stopPropagation();
    });
    ["pointerdown", "pointerup", "pointermove"].forEach(type => guide.addEventListener(type, e => e.stopPropagation()));
    const parts = {
      history: ["Устойчивый портрет по истории", "Основные 896 чисел — профиль известного хоста и адаптация позднего трафика к этому пространству. Эта часть помогает узнавать устройство по поведению в разные дни."],
      contacts: ["Подсказка о новых собеседниках", "Отдельные 64 числа обучаются для задачи новых контактов. Они помогают выбирать профили соседей, чьи связи полезны для прогноза будущих собеседников исходного хоста."],
      graph: ["Место в структуре сети", "Ещё 64 числа получаются из собственных векторов графа взаимодействий. Они дополняют поведение сведениями об окружении: важно не только сколько данных передаёт хост, но и с какими узлами он связан."]
    };
    guide.querySelectorAll("[data-model-part]").forEach(button => button.addEventListener("click", () => {
      guide.querySelectorAll("[data-model-part]").forEach(b => b.setAttribute("aria-pressed", String(b === button)));
      const part = parts[button.dataset.modelPart];
      document.getElementById("pt-model-part-title").textContent = part[0];
      document.getElementById("pt-model-part-copy").textContent = part[1];
    }));
    if (!document.body.dataset.page) window.__ptStartLab?.();
    window.__ptRefreshLayout?.();
  };
  window.__ptMountContent();
})();
