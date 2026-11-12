import { h as gsap } from "/_nuxt/iNjJ8EkR.js";
const steps = [
  {
    id: "problem",
    start: 850,
    end: 2150,
    label: "01 / ЗАЧЕМ ЭТО НУЖНО",
    title: "Что скрывается<br>за адресом хоста?",
    body: "Хост — это компьютер или сервер в сети. В логах виден его адрес, но не всегда понятно, чем он занимается и какое поведение для него привычно.",
    detail:
      "Мы собираем контекст его взаимодействий: с кем он общается, по каким сервисам, сколько передаёт данных и когда активен.",
    link: "#problem",
    cta: "Разобраться в задаче",
  },
  {
    id: "product",
    start: 2150,
    end: 3450,
    label: "02 / КАРТА ПРОДУКТА",
    title: "Три вопроса<br>к своей сети.",
    body: "Сервис помогает оператору SOC — специалисту, который следит за безопасностью сети, — исследовать поведение устройств.",
    kind: "product",
    link: "#product",
    cta: "Посмотреть карту продукта",
  },
  {
    id: "input",
    start: 3450,
    end: 4550,
    label: "03 / ИЗ ЧЕГО СТРОИМ ПРОФИЛЬ",
    title: "Из записей —<br>в поведение.",
    body: "Одна запись соединения мало что говорит. За окно времени мы собираем признаки активности хоста, его собеседников и используемых сервисов.",
    kind: "input",
    link: "#model",
    cta: "Посмотреть все этапы",
  },
  {
    id: "encoder",
    start: 4550,
    end: 5650,
    label: "04 / КОДИРОВЩИК ТРАФИКА",
    title: "Поведение становится<br>набором чисел.",
    body: "Нейросеть-кодировщик переводит признаки трафика в вектор. В нём 896 чисел. По таким описаниям можно сравнивать компьютеры, даже если их реальные роли заранее неизвестны.",
    kind: "encoder",
    detail:
      "Многодневная модель дообучает отдельные профили известных хостов по истории. Адаптер помогает сопоставлять с ними трафик более позднего окна.",
    link: "#model",
    cta: "Заглянуть внутрь модели",
  },
  {
    id: "architecture",
    start: 5650,
    end: 6950,
    label: "05 / MULTIDAY CODEBOOK PLUS",
    title: "Один профиль.<br>Три части.",
    body: "К истории хоста мы добавили отдельный блок для прогноза новых собеседников и информацию о структуре сетевого графа.",
    kind: "architecture",
    link: "#model",
    cta: "Подробная архитектура",
  },
  {
    id: "training",
    start: 6950,
    end: 7850,
    label: "06 / КАК ПРОВЕРЯЕМ",
    title: "Учимся на прошлом.<br>Проверяем на будущем.",
    body: "Настройки выбираются до финальной оценки. Модель фиксируется, а затем проверяется на более поздних днях, которые не использовались при обучении.",
    kind: "training",
    link: "#results",
    cta: "Методика и ограничения",
  },
  {
    id: "results",
    start: 7850,
    end: 8900,
    label: "07 / РЕЗУЛЬТАТЫ ИССЛЕДОВАНИЯ",
    title: "Качество, которое<br>можно сравнить.",
    body: "На общих тестовых окнах multiday_codebook_plus получила 70,0 из 100. Общий score объединяет несколько задач, а не показывает процент обнаруженных атак.",
    kind: "results",
    link: "#results",
    cta: "Понятно о каждой метрике",
  },
  {
    id: "demo",
    start: 8900,
    end: 10080,
    label: "08 / ПОПРОБУЙТЕ ПРЯМО ЗДЕСЬ",
    title: "Начнём<br>с одного хоста.",
    body: "Выберите пример. Мы сравним его вектор с остальными и покажем ближайший профиль. Затем можно перейти в лабораторию и загрузить свои данные.",
    kind: "demo",
    link: "#demo",
    cta: "Открыть большую лабораторию",
  },
];
let root,
  panel,
  nav,
  current = -1,
  smoother = null,
  started = false,
  exampleIndex = 0,
  found = false;
const $ = (id) => document.getElementById(id);
const metrics = {
  identity: [
    "59,6%",
    "Верный хост найден с первой попытки в среднем по тестовым окнам. Это задача узнавания устройства в другой день.",
  ],
  links: [
    "0,974",
    "ROC-AUC: качество различения будущей связи и её отсутствия. Это не 97,4% верных предсказаний.",
  ],
  contacts: [
    "0,318",
    "MRR: качество ранжирования новых собеседников. Чем раньше в списке верный контакт, тем выше оценка.",
  ],
};
function element(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function openProduct(hash) {
  window.__ptOpenProduct?.(hash);
}
function sceneScroll(y) {
  try {
    smoother = gsap.core.globals().ScrollSmoother?.get();
    if (smoother) {
      smoother.scrollTo(y, true);
      return;
    }
  } catch {}
  window.scrollTo({
    top: y,
    behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "instant"
      : "smooth",
  });
}
function illustrations(kind) {
  if (kind === "product")
    return '<div class="pt-scene-list"><div><b>01</b><p><strong>Найти похожие хосты</strong><small>Поиск доступен в демонстрации</small></p></div><div><b>02</b><p><strong>Исследовать группы</strong><small>Следующий этап сервиса</small></p></div><div><b>03</b><p><strong>Замечать необычные переходы</strong><small>Следующий этап сервиса</small></p></div></div>';
  if (kind === "input")
    return '<div class="pt-scene-features"><span>Собеседники</span><span>Порты и протоколы</span><span>Пакеты и байты</span><span>Время и длительность</span></div>';
  if (kind === "encoder")
    return '<div class="pt-encoder-picture" aria-label="Схема: признаки, кодировщик, вектор"><span>Признаки</span><b>→</b><span>Кодировщик</span><b>→</b><strong>896</strong></div>';
  if (kind === "architecture")
    return '<div class="pt-architecture-selector" role="group" aria-label="Части модели"><button data-part="0" aria-pressed="true"><b>896</b> История</button><button data-part="1" aria-pressed="false"><b>64</b> Контакты</button><button data-part="2" aria-pressed="false"><b>64</b> Граф</button></div><p class="pt-scene-detail" id="pt-part-copy">Профиль известного хоста и адаптер, который переводит поздний трафик в то же пространство. Итого все три части дают вектор из 1 024 чисел.</p>';
  if (kind === "training")
    return '<div class="pt-scene-timeline"><div><b>02–12</b><span>Обучение</span></div><i>→</i><div><b>13</b><span>Подбор</span></div><i>→</i><div><b>14–16</b><span>Проверка</span></div></div>';
  if (kind === "results")
    return (
      '<div class="pt-scene-score"><strong>70<small>/100</small></strong><span>multiday_codebook_plus<br>Таблица от 4 ноября 2026</span></div><div class="pt-scene-metrics" role="group" aria-label="Задачи оценки"><button data-metric="identity" aria-pressed="true">Узнать хост</button><button data-metric="links" aria-pressed="false">Предсказать связь</button><button data-metric="contacts" aria-pressed="false">Новый контакт</button></div><p class="pt-metric-copy"><b id="pt-metric-number">59,6%</b> <span id="pt-metric-text">' +
      metrics.identity[1] +
      "</span></p>"
    );
  if (kind === "demo") return '<div id="pt-mini-demo"></div>';
  return "";
}
function renderStep(index) {
  if (index === current) return;
  current = index;
  const step = steps[index];
  panel.dataset.step = step.id;
  panel.setAttribute("aria-label", step.title.replaceAll("<br>", " "));
  panel.innerHTML =
    '<span class="pt-story-eyebrow">' +
    step.label +
    "</span><h2>" +
    step.title +
    '</h2><p class="pt-story-body">' +
    step.body +
    "</p>" +
    illustrations(step.kind) +
    (step.detail ? '<p class="pt-scene-detail">' + step.detail + "</p>" : "") +
    '<button class="pt-story-more" data-open="' +
    step.link +
    '">' +
    step.cta +
    " <span>↗</span></button>";
  panel
    .querySelector("[data-open]")
    .addEventListener("click", () => openProduct(step.link));
  panel.querySelectorAll("[data-part]").forEach((button) =>
    button.addEventListener("click", () => {
      panel
        .querySelectorAll("[data-part]")
        .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      $("pt-part-copy").textContent = [
        "Профиль известного хоста и адаптер, который переводит поздний трафик в то же пространство. Итого все три части дают вектор из 1 024 чисел.",
        "64 числа учатся подбирать соседей, чьи контакты помогают угадывать будущих новых собеседников исходного хоста.",
        "64 числа описывают положение хоста в графе взаимодействий. Они дополняют поведенческий профиль структурой связей.",
      ][Number(button.dataset.part)];
    }),
  );
  panel.querySelectorAll("[data-metric]").forEach((button) =>
    button.addEventListener("click", () => {
      panel
        .querySelectorAll("[data-metric]")
        .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
      const m = metrics[button.dataset.metric];
      $("pt-metric-number").textContent = m[0];
      $("pt-metric-text").textContent = m[1];
    }),
  );
  if (step.kind === "demo") renderMini();
  nav
    .querySelectorAll("[data-story]")
    .forEach((button) =>
      button.setAttribute(
        "aria-current",
        String(
          button.dataset.story === step.id ||
            (button.dataset.story === "encoder" &&
              ["input", "architecture"].includes(step.id)),
        ),
      ),
    );
}
function renderMini() {
  const box = $("pt-mini-demo");
  if (!box || !window.HOSTS?.length) return;
  const host = window.HOSTS[exampleIndex % window.HOSTS.length],
    norm = (h) => Math.hypot(...h.v);
  const nearest = window.HOSTS.filter((h) => h !== host)
    .map((h) => ({
      host: h,
      score:
        host.v.reduce((sum, x, i) => sum + x * h.v[i], 0) /
        (norm(host) * norm(h)),
    }))
    .sort((a, b) => b.score - a.score)[0];
  box.replaceChildren();
  const figure = element("div", "pt-mini-network");
  figure.innerHTML =
    '<span class="pt-mini-node">Хост A</span><i class="' +
    (found ? "found" : "") +
    '"></i><span class="pt-mini-node ' +
    (found ? "selected" : "") +
    '">' +
    (found ? "Хост B" : "?") +
    "</span>";
  box.append(figure);
  const text = element(
    "p",
    "pt-mini-caption",
    found
      ? "Ближайший профиль найден. Косинусное сходство: " +
          nearest.score.toFixed(3) +
          ". Значение не означает вероятность атаки."
      : "У каждого хоста — числовой профиль. Найдём самый близкий.",
  );
  box.append(text);
  const controls = element("div", "pt-mini-controls");
  const action = element(
    "button",
    "",
    found ? "Исследовать соседа ↗" : "Найти похожего",
  );
  action.addEventListener("click", () => {
    if (found) {
      exampleIndex = window.HOSTS.indexOf(nearest.host);
      found = false;
    } else found = true;
    renderMini();
  });
  const reset = element("button", "", "Другой пример ↻");
  reset.addEventListener("click", () => {
    exampleIndex = (exampleIndex + 41) % window.HOSTS.length;
    found = false;
    renderMini();
  });
  controls.append(action, reset);
  box.append(controls);
  box.append(
    element("small", "", "Пример на тестовых behavioral-векторах проекта."),
  );
}
function mount() {
  if (root || !document.body) return;
  document.documentElement.lang = "ru";
  document.body.classList.add("pt-has-story");
  root = element("div", "pt-story-root");
  root.id = "pt-story";
  root.hidden = true;
  panel = element("section", "pt-story-panel");
  panel.setAttribute("aria-live", "polite");
  root.append(panel);
  nav = element("nav", "pt-story-nav");
  nav.setAttribute("aria-label", "Рассказ о проекте");
  [
    ["problem", "Задача", 900],
    ["encoder", "Модель", 3500],
    ["results", "Результаты", 7900],
    ["demo", "Лаборатория", 9000],
  ].forEach(([id, text, y]) => {
    const button = element("button", "", text);
    button.dataset.story = id;
    button.addEventListener("click", () => sceneScroll(y));
    nav.append(button);
  });
  root.append(nav);
  const cue = element(
    "div",
    "pt-story-cue",
    "Сетевой трафик → профиль хоста. Прокрутите, чтобы разобраться ↓",
  );
  root.append(cue);
  ["click", "pointerdown", "pointermove", "pointerup"].forEach((type) =>
    panel.addEventListener(type, (e) => e.stopPropagation()),
  );
  ["click", "pointerdown", "pointermove", "pointerup"].forEach((type) =>
    root.addEventListener(type, (e) => e.stopPropagation()),
  );
  document.body.append(root);
  renderStep(0);
}
function frame() {
  requestAnimationFrame(frame);
  if (document.hidden) return;
  mount();
  if (!root) return;
  document.body.classList.add("pt-has-story");
  const scene = document.getElementById("home-scene");
  const visible = scene && Number(getComputedStyle(scene).opacity) > 0.05;
  try {
    smoother = gsap.core.globals().ScrollSmoother?.get();
  } catch {}
  const y = smoother?.scrollTop?.() ?? window.scrollY;
  started = !!visible;
  root.hidden = !started || y > 10100;
  document.body.classList.toggle("pt-story-active", started && y <= 10100);
  if (!started) return;
  const index = steps.findIndex((step) => y >= step.start && y < step.end);
  panel.hidden = index < 0;
  if (index >= 0) {
    renderStep(index);
    const step = steps[index],
      p = (y - step.start) / (step.end - step.start),
      fade = Math.min(1, p / 0.1, (1 - p) / 0.09);
    panel.style.opacity = String(Math.max(0.06, fade));
    panel.style.translate = matchMedia("(prefers-reduced-motion: reduce)")
      .matches
      ? ""
      : `0 ${(1 - Math.min(1, p / 0.14)) * 14}px`;
  }
  nav.hidden = y < 700;
  root.querySelector(".pt-story-cue").hidden = y > 800;
}
mount();
requestAnimationFrame(frame);
