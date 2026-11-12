"use strict";
const $ = (id) => document.getElementById(id);
const motionPreference = matchMedia("(prefers-reduced-motion: reduce)");
const clamp = (x, a = 0, b = 1) => Math.min(b, Math.max(a, x));
const lerp = (a, b, t) => a + (b - a) * t;
const ease = (t) => t * t * (3 - 2 * t);
const chapters = [...document.querySelectorAll("[data-scene]")];
let pageProgress = 0,
  sceneProgress = 0,
  targetScene = 0,
  activeScene = 0;
let chapterBounds = [];
function measure() {
  chapterBounds = chapters.map((el) => ({
    el,
    top: el.getBoundingClientRect().top + scrollY,
    height: el.offsetHeight,
    scene: Number(el.dataset.scene),
  }));
  updateScroll();
}
function updateScroll() {
  updateModelPin();
  const y = scrollY + innerHeight * 0.12;
  let index = 0;
  chapterBounds.forEach((b, i) => {
    if (y >= b.top) index = i;
  });
  const current = chapterBounds[index],
    next = chapterBounds[Math.min(index + 1, chapterBounds.length - 1)];
  if (current) {
    const span = Math.max(1, next.top - current.top);
    const p = ease(clamp((y - current.top) / span));
    targetScene = current.scene + (next.scene - current.scene) * p;
    activeScene = current.scene;
  }
  pageProgress = clamp(
    scrollY / Math.max(1, document.documentElement.scrollHeight - innerHeight),
  );
  $("scroll-progress").style.transform = `scaleY(${pageProgress})`;
  $("chapter-number").textContent = String(
    Math.min(9, Math.floor(pageProgress * 9) + 1),
  ).padStart(2, "0");
  document
    .querySelector(".site-header")
    .classList.toggle("scrolled", scrollY > 60);
  const h = $("home").getBoundingClientRect();
  const heroP = clamp(-h.top / Math.max(1, h.height));
  if (!motionPreference.matches) {
    const title = document.querySelector(".hero h1");
    title.style.transform = `translateY(${-5 - heroP * 12}%) scale(${1 + heroP * 0.06})`;
    title.style.opacity = String(1 - heroP * 0.8);
  }
}
function updateModelPin() {
  const section = $("model-experience");
  if (!section) return;
  const rect = section.getBoundingClientRect();
  const local = clamp(
    -rect.top / Math.max(1, section.offsetHeight - innerHeight),
  );
  const step = Math.min(3, Math.floor(local * 4));
  document.querySelectorAll("[data-model-step]").forEach((el) => {
    const active = Number(el.dataset.modelStep) === step;
    el.classList.toggle("is-active", active);
    el.setAttribute("aria-hidden", String(!active));
  });
  document
    .querySelectorAll(".model-scene-progress i")
    .forEach((el, i) => el.classList.toggle("active", i <= step));
  $("model-scene-count").textContent =
    String(step + 1).padStart(2, "0") + " / 04";
  $("model-scene-annotation").textContent = [
    "Входные взаимодействия",
    "Числовые признаки поведения",
    "Профиль с историей",
    "Соседи в пространстве профилей",
  ][step];
}

addEventListener("scroll", updateScroll, { passive: true });
addEventListener("resize", measure);
document.fonts?.ready.then(measure);
const observer = new IntersectionObserver(
  (entries) =>
    entries.forEach((entry) => {
      if (entry.isIntersecting) entry.target.classList.add("in-view");
    }),
  { threshold: 0.13 },
);
document.querySelectorAll(".chapter").forEach((el) => observer.observe(el));
// Guided demonstration. The comparison uses the project's test vectors, not invented scores.
const hosts = window.HOSTS.map((h) => ({ ...h, norm: Math.hypot(...h.v) }));
let hostIndex = 0,
  selected = hosts[0],
  closest = [],
  demoStep = 0,
  comparedNeighbor = 0;
const cosine = (a, b) =>
  a.norm && b.norm
    ? a.v.reduce((sum, x, i) => sum + x * b.v[i], 0) / (a.norm * b.norm)
    : 0;
function chooseHost(host) {
  selected = host;
  closest = hosts
    .filter((h) => h !== host)
    .map((h) => ({ host: h, score: cosine(host, h) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  demoStep = 0;
  comparedNeighbor = 0;
  renderDemo();
}
const svgNS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs) {
  const el = document.createElementNS(svgNS, tag);
  Object.entries(attrs).forEach(([k, v]) => el.setAttribute(k, String(v)));
  return el;
}
const demoPoints = [
  { x: 280, y: 180, r: 30, label: "Хост A" },
  { x: 420, y: 105, r: 20, label: "Хост B" },
  { x: 440, y: 265, r: 15, label: "Хост C" },
  { x: 205, y: 295, r: 18, label: "Хост D" },
  { x: 110, y: 145, r: 14, label: "Хост E" },
  { x: 250, y: 55, r: 11, label: "Хост F" },
];
let draggedNode = null,
  dragMoved = false;
function updateNetworkPositions() {
  const center = demoPoints[0];
  document.querySelectorAll("#demo-lines line").forEach((line, i) => {
    const p = demoPoints[i + 1];
    line.setAttribute("x1", center.x);
    line.setAttribute("y1", center.y);
    line.setAttribute("x2", p.x);
    line.setAttribute("y2", p.y);
  });
  document.querySelectorAll("#demo-nodes .demo-node").forEach((g, i) => {
    const p = demoPoints[i];
    g.querySelector("circle").setAttribute("cx", p.x);
    g.querySelector("circle").setAttribute("cy", p.y);
    g.querySelector("text").setAttribute("x", p.x);
    g.querySelector("text").setAttribute("y", p.y + p.r + 25);
  });
}
function renderNetwork() {
  const lines = $("demo-lines"),
    nodes = $("demo-nodes");
  lines.replaceChildren();
  nodes.replaceChildren();
  const center = demoPoints[0];
  demoPoints.slice(1).forEach((p, i) =>
    lines.append(
      svgEl("line", {
        x1: center.x,
        y1: center.y,
        x2: p.x,
        y2: p.y,
        class:
          "demo-line" +
          (demoStep > 0 && i === comparedNeighbor ? " match" : ""),
      }),
    ),
  );
  demoPoints.forEach((p, i) => {
    const g = svgEl("g", {
      class: "demo-node",
      tabindex: i === 0 ? -1 : 0,
      role: "button",
      "aria-label": i === 0 ? "Выбранный хост" : `Сравнить с ${p.label}`,
    });
    g.append(
      svgEl("circle", {
        cx: p.x,
        cy: p.y,
        r: demoStep > 0 && i === comparedNeighbor + 1 ? 28 : p.r,
        opacity: demoStep === 0 && i > 0 ? 0.42 : 1,
      }),
    );
    const text = svgEl("text", {
      x: p.x,
      y: p.y + p.r + 25,
      class: i === 0 ? "main-label" : "",
    });
    text.textContent = p.label;
    g.append(text);
    g.addEventListener("pointerdown", (e) => {
      draggedNode = { index: i, x: e.clientX, y: e.clientY };
      dragMoved = false;
      try {
        g.setPointerCapture(e.pointerId);
      } catch {}
    });
    if (i > 0) {
      const compare = () => {
        if (dragMoved) return;
        comparedNeighbor = i - 1;
        demoStep = 1;
        renderDemo();
      };
      g.addEventListener("click", compare);
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          dragMoved = false;
          compare();
        }
      });
    }
    nodes.append(g);
  });
}
$("demo-network").addEventListener("pointermove", (e) => {
  if (!draggedNode) return;
  if (Math.hypot(e.clientX - draggedNode.x, e.clientY - draggedNode.y) > 5)
    dragMoved = true;
  if (!dragMoved) return;
  const svg = $("demo-network"),
    matrix = svg.getScreenCTM();
  if (!matrix) return;
  const point = svg.createSVGPoint();
  point.x = e.clientX;
  point.y = e.clientY;
  const p = point.matrixTransform(matrix.inverse());
  demoPoints[draggedNode.index].x = clamp(p.x, 40, 520);
  demoPoints[draggedNode.index].y = clamp(p.y, 35, 315);
  updateNetworkPositions();
});
["pointerup", "pointercancel", "lostpointercapture"].forEach((event) =>
  $("demo-network").addEventListener(event, () => {
    draggedNode = null;
  }),
);
function renderDemo() {
  const descriptions = [
    [
      "01 / ВЫБРАННЫЙ ХОСТ",
      "Вот наш хост.",
      "Модель уже превратила его поведение в числовой профиль. Теперь можно сравнить его с профилями других устройств.",
      "Найти похожего",
      "У каждого устройства — свой профиль",
    ],
    [
      "02 / БЛИЖАЙШИЙ СОСЕД",
      "Хост B — самый близкий.",
      "Мы сравнили векторы. Среди доступных устройств профиль хоста B оказался ближе всего к профилю A. Нажмите на B, чтобы продолжить исследование с него.",
      "Посмотреть все связи",
      "Линия выделяет наиболее похожий профиль",
    ],
    [
      "03 / ЧТО ЭТО ЗНАЧИТ",
      "Похожесть — начало вопроса.",
      "Мы нашли близкий профиль. Следующий шаг продукта — объяснить сходство по сервисам, окружению и другим сетевым признакам. Сейчас показано именно сравнение векторов.",
      "Начать заново",
      "Выберите любой узел, чтобы сменить точку отсчёта",
    ],
  ];
  const d = descriptions[demoStep];
  $("demo-step-label").textContent = d[0];
  $("demo-heading").textContent =
    demoStep === 1
      ? `${demoPoints[comparedNeighbor + 1].label} ${comparedNeighbor === 0 ? "— самый близкий." : "— выбран для сравнения."}`
      : d[1];
  $("demo-description").textContent =
    demoStep === 1
      ? "Сравниваем его профиль с профилем A. Ниже — настоящее косинусное сходство векторов. Можно выбрать другой узел, подвигать схему или сделать этот хост новой точкой отсчёта."
      : d[2];
  $("demo-action").replaceChildren(document.createTextNode(d[3]));
  const arrow = document.createElement("span");
  arrow.textContent = "↗";
  $("demo-action").append(arrow);
  $("demo-caption").textContent = d[4];
  $("host-id").textContent = selected.id;
  $("demo-result").hidden = demoStep === 0;
  $("demo-score").textContent = closest[comparedNeighbor].score.toFixed(3);
  renderNetwork();
}
$("reanchor").addEventListener("click", () =>
  chooseHost(closest[comparedNeighbor].host),
);
$("demo-action").addEventListener("click", () => {
  demoStep = (demoStep + 1) % 3;
  renderDemo();
});
$("next-example").addEventListener("click", () => {
  hostIndex = (hostIndex + 41) % hosts.length;
  chooseHost(hosts[hostIndex]);
});
function renderHostSearch() {
  const q = $("host-search").value.trim().toLowerCase();
  const matches = hosts.filter((h) => h.id.toLowerCase().includes(q));
  $("search-status").textContent = q
    ? matches.length
      ? `Найдено: ${matches.length}`
      : "Хост не найден"
    : "";
  $("host-options").replaceChildren(
    ...matches.slice(0, 4).map((host) => {
      const b = document.createElement("button");
      b.textContent = host.id;
      b.addEventListener("click", () => chooseHost(host));
      return b;
    }),
  );
}
$("host-search").addEventListener("input", renderHostSearch);
chooseHost(selected);
renderHostSearch();
// Ambient music is requested by default; browser autoplay policy may defer it until interaction.
let audio = null,
  audioEnabled = false;
function createAudio() {
  const Context = window.AudioContext || window.webkitAudioContext;
  if (!Context) return null;
  const ctx = new Context(),
    master = ctx.createGain(),
    filter = ctx.createBiquadFilter();
  master.gain.value = 0;
  filter.type = "lowpass";
  filter.frequency.value = 900;
  filter.Q.value = 0.25;
  filter.connect(master);
  master.connect(ctx.destination);
  const oscillators = [];
  [130.8128, 195.9977, 261.6256, 311.127, 391.9954].forEach((frequency, i) => {
    const oscillator = ctx.createOscillator(),
      gain = ctx.createGain();
    oscillator.type = "sine";
    oscillator.frequency.value = frequency;
    gain.gain.value = 0.035 / (1 + i * 0.35);
    oscillator.connect(gain);
    gain.connect(filter);
    const lfo = ctx.createOscillator(),
      depth = ctx.createGain();
    lfo.frequency.value = 0.035 + i * 0.008;
    depth.gain.value = 0.012 / (1 + i * 0.3);
    lfo.connect(depth);
    depth.connect(gain.gain);
    oscillator.start();
    lfo.start();
    oscillators.push(oscillator, lfo);
  });
  return { ctx, master, oscillators };
}
let audioWanted = true;
function reflectAudio() {
  const playing = audioEnabled && audioWanted;
  $("sound").setAttribute("aria-pressed", String(audioWanted));
  $("sound").dataset.playing = String(playing);
  $("sound").setAttribute(
    "aria-label",
    audioWanted ? "Выключить музыку" : "Включить музыку",
  );
  $("sound-label").textContent = playing
    ? "Звук вкл."
    : audioWanted
      ? "Звук: авто"
      : "Звук выкл.";
}
async function activateAudio() {
  if (!audioWanted) return;
  try {
    if (!audio) audio = createAudio();
    if (!audio) {
      $("audio-status").textContent = "Аудио не поддерживается";
      return;
    }
    await audio.ctx.resume();
    if (!audioWanted) return;
    if (audio.ctx.state && audio.ctx.state !== "running") return;
    audioEnabled = true;
    audio.master.gain.cancelScheduledValues(audio.ctx.currentTime);
    audio.master.gain.setTargetAtTime(0.45, audio.ctx.currentTime, 1.2);
    reflectAudio();
    $("audio-status").textContent = "Спокойная музыка включена";
  } catch {
    reflectAudio();
  }
}
$("sound").addEventListener("click", () => {
  audioWanted = !audioWanted;
  if (audioWanted) activateAudio();
  else {
    audioEnabled = false;
    if (audio)
      audio.master.gain.setTargetAtTime(0, audio.ctx.currentTime, 0.08);
    $("audio-status").textContent = "Музыка выключена";
  }
  reflectAudio();
});
function unlockAudio(e) {
  if (e.target.closest("#sound")) return;
  if (audioWanted && !audioEnabled) activateAudio();
}
document.addEventListener("pointerdown", unlockAudio, { passive: true });
document.addEventListener("keydown", unlockAudio);
document.addEventListener("visibilitychange", () => {
  if (audio)
    audio.master.gain.setTargetAtTime(
      document.hidden ? 0 : audioEnabled && audioWanted ? 0.45 : 0,
      audio.ctx.currentTime,
      0.5,
    );
});
reflectAudio();
activateAudio();

let cutawayLayer = 0;
const layerData = [
  [
    "896 чисел",
    "Профиль хоста",
    "Основная часть хранит выученное представление известного компьютера. Адаптер позволяет сопоставить с ним трафик позднего окна.",
  ],
  [
    "64 числа",
    "Прогноз новых контактов",
    "Эта часть учится подбирать соседей, чьи контакты помогают предсказывать будущих собеседников исходного хоста.",
  ],
  [
    "64 числа",
    "Структура графа",
    "Эта часть описывает положение хоста в сети связей. Она добавляет информацию, которую отдельный поведенческий профиль может не сохранить.",
  ],
];
document.querySelectorAll("[data-layer]").forEach((button) =>
  button.addEventListener("click", () => {
    cutawayLayer = Number(button.dataset.layer);
    document
      .querySelectorAll("[data-layer]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    const item = layerData[cutawayLayer];
    $("layer-number").textContent = item[0];
    $("layer-title").textContent = item[1];
    $("layer-description").textContent = item[2];
  }),
);
const metricData = [
  [
    "УЗНАВАНИЕ С ПЕРВОЙ ПОПЫТКИ",
    "59,6%",
    "Identity Top-1",
    "Тот же компьютер, другой день.",
    "Модель сравнивает профиль с более поздним трафиком и пытается узнать тот же хост. В среднем по тестовым окнам верный хост найден первым в 59,6% случаев.",
    "Это узнавание хоста среди кандидатов, а не обнаружение атак.",
  ],
  [
    "РАЗЛИЧЕНИЕ БУДУЩИХ СВЯЗЕЙ",
    "0,974",
    "ROC-AUC · больше — лучше",
    "Будут ли два хоста общаться?",
    "По двум профилям классификатор пытается отличить будущую связь от её отсутствия. ROC-AUC оценивает качество такого различения.",
    "0,5 соответствует случайному ранжированию, 1 — идеальному. 0,974 нельзя читать как 97,4% верных предсказаний.",
  ],
  [
    "КАЧЕСТВО ПОИСКА НОВОГО КОНТАКТА",
    "0,318",
    "MRR · больше — лучше",
    "Насколько рано найдён верный контакт?",
    "Модель ранжирует возможных новых собеседников. Чем выше в списке оказывается настоящий будущий контакт, тем выше оценка. MRR усредняет обратную позицию верного ответа.",
    "1 означает, что верный контакт всегда первый. 0,318 — не процент угаданных контактов и не постоянное третье место.",
  ],
];
document.querySelectorAll("[data-metric]").forEach((button) =>
  button.addEventListener("click", () => {
    document
      .querySelectorAll("[data-metric]")
      .forEach((b) => b.setAttribute("aria-pressed", String(b === button)));
    const item = metricData[Number(button.dataset.metric)];
    [
      "metric-name",
      "metric-value",
      "metric-unit",
      "metric-title",
      "metric-copy",
      "metric-caution",
    ].forEach((id, i) => ($(id).textContent = item[i]));
  }),
);

// One persistent, scroll-driven 3D scene, rather than independent decorative canvases.
function initWorld() {
  const canvas = $("world");
  if (!window.THREE) {
    document.body.classList.add("no-webgl");
    return;
  }
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({
      canvas,
      alpha: true,
      antialias: true,
      powerPreference: "high-performance",
    });
  } catch {
    document.body.classList.add("no-webgl");
    return;
  }
  renderer.setPixelRatio(
    Math.min(devicePixelRatio || 1, innerWidth < 769 ? 1.3 : 1.7),
  );
  renderer.setClearColor(
    new THREE.Color(
      getComputedStyle(document.documentElement)
        .getPropertyValue("--black")
        .trim(),
    ),
    0,
  );
  renderer.toneMapping = THREE.ACESFilmicToneMapping;
  renderer.toneMappingExposure = 1.25;
  const token = (name) =>
    getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const colors = {
    core: new THREE.Color(token("--scene-core")),
    edge: new THREE.Color(token("--scene-edge")),
    blue: new THREE.Color(token("--scene-blue")),
    pink: new THREE.Color(token("--scene-pink")),
    highlight: new THREE.Color(token("--scene-highlight")),
    white: new THREE.Color(token("--white")),
    dark: new THREE.Color(token("--dark")),
  };
  const scene = new THREE.Scene(),
    camera = new THREE.PerspectiveCamera(32, 1, 0.1, 100);
  camera.position.set(0, 0, 15);
  scene.add(new THREE.HemisphereLight(colors.white, colors.core, 2.3));
  const key = new THREE.DirectionalLight(colors.white, 4.5);
  key.position.set(-3, 6, 7);
  scene.add(key);
  const rim = new THREE.PointLight(colors.pink, 55, 30);
  rim.position.set(4, -1, 4);
  scene.add(rim);
  const blueLight = new THREE.PointLight(colors.blue, 45, 30);
  blueLight.position.set(-5, 2, 2);
  scene.add(blueLight);
  // Original studio environment for refractive highlights.
  const envCanvas = document.createElement("canvas");
  envCanvas.width = 1024;
  envCanvas.height = 512;
  const env = envCanvas.getContext("2d");
  const gradient = env.createLinearGradient(0, 0, 1024, 512);
  gradient.addColorStop(0, token("--scene-highlight"));
  gradient.addColorStop(0.4, token("--white"));
  gradient.addColorStop(0.57, token("--scene-blue"));
  gradient.addColorStop(0.75, token("--scene-core"));
  gradient.addColorStop(1, token("--white"));
  env.fillStyle = gradient;
  env.fillRect(0, 0, 1024, 512);
  env.fillStyle = token("--white");
  env.fillRect(80, 0, 90, 512);
  env.fillRect(550, 80, 50, 350);
  env.fillStyle = token("--scene-edge");
  env.fillRect(290, 0, 90, 512);
  const envTexture = new THREE.CanvasTexture(envCanvas);
  envTexture.mapping = THREE.EquirectangularReflectionMapping;
  const pmrem = new THREE.PMREMGenerator(renderer);
  const environment = pmrem.fromEquirectangular(envTexture);
  scene.environment = environment.texture;
  envTexture.dispose();
  pmrem.dispose();
  if (window.RGBELoader && window.STUDIO_HDR) {
    try {
      const binary = atob(window.STUDIO_HDR),
        bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
      const decoded = new window.RGBELoader().parse(bytes.buffer);
      const hdr = new THREE.DataTexture(
        decoded.data,
        decoded.width,
        decoded.height,
        THREE.RGBAFormat,
        decoded.type,
      );
      hdr.mapping = THREE.EquirectangularReflectionMapping;
      hdr.colorSpace = THREE.LinearSRGBColorSpace;
      hdr.flipY = true;
      hdr.minFilter = THREE.LinearFilter;
      hdr.magFilter = THREE.LinearFilter;
      hdr.generateMipmaps = false;
      hdr.needsUpdate = true;
      const generator = new THREE.PMREMGenerator(renderer);
      const studio = generator.fromEquirectangular(hdr);
      scene.environment = studio.texture;
      environment.dispose();
      hdr.dispose();
      generator.dispose();
    } catch (error) {
      console.warn("Using procedural studio fallback", error);
    }
  }

  const root = new THREE.Group();
  scene.add(root);
  const glassMaterial = new THREE.MeshPhysicalMaterial({
    color: colors.core,
    metalness: 0.05,
    roughness: 0.08,
    transmission: 0.92,
    thickness: 0.85,
    ior: 1.36,
    clearcoat: 1,
    clearcoatRoughness: 0.04,
    envMapIntensity: 1.9,
    transparent: true,
    opacity: 1,
  });

  // An original glass host-profile sculpture: a closed, lobed data capsule.
  function capsuleGeometry() {
    const vertices = [],
      indices = [],
      uv = [],
      radial = 112,
      bands = 56;
    for (let j = 0; j <= bands; j++) {
      const v = j / bands,
        theta = v * Math.PI;
      for (let i = 0; i <= radial; i++) {
        const u = i / radial,
          phi = u * Math.PI * 2,
          lobe = 1 + 0.07 * Math.sin(phi * 6) * Math.pow(Math.sin(theta), 3),
          radius = Math.sin(theta) * 1.2 * lobe;
        vertices.push(
          Math.cos(phi) * radius,
          Math.cos(theta) * 1.48,
          Math.sin(phi) * radius,
        );
        uv.push(u, v);
      }
    }
    for (let j = 0; j < bands; j++)
      for (let i = 0; i < radial; i++) {
        const a = j * (radial + 1) + i,
          b = a + radial + 1;
        indices.push(a, b, a + 1, b, b + 1, a + 1);
      }
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.Float32BufferAttribute(vertices, 3),
    );
    geometry.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
    geometry.setIndex(indices);
    geometry.computeVertexNormals();
    return geometry;
  }
  glassMaterial.side = THREE.DoubleSide;
  const core = new THREE.Mesh(capsuleGeometry(), glassMaterial);
  const innerShell = new THREE.Mesh(capsuleGeometry(), glassMaterial.clone());
  innerShell.scale.set(0.76, 0.8, 0.76);
  core.add(innerShell);
  const orbitalMaterial = new THREE.MeshPhysicalMaterial({
    color: colors.highlight,
    metalness: 0.7,
    roughness: 0.11,
    envMapIntensity: 2,
  });
  for (let i = 0; i < 3; i++) {
    const band = new THREE.Mesh(
      new THREE.TorusGeometry(0.93 + i * 0.09, 0.016, 8, 180),
      orbitalMaterial,
    );
    band.rotation.set(Math.PI / 2 + i * 0.45, i * 0.5, 0.25);
    core.add(band);
  }
  const organicTime = { value: 0 };
  core.name = "HostProfileShell";
  root.add(core);
  const inner = new THREE.Mesh(
    new THREE.SphereGeometry(0.44, 32, 20),
    new THREE.MeshPhysicalMaterial({
      color: colors.pink,
      metalness: 0.1,
      roughness: 0.12,
      clearcoat: 1,
      envMapIntensity: 1.8,
    }),
  );
  root.add(inner);
  const satelliteGeo = new THREE.SphereGeometry(0.11, 20, 12);
  const satelliteMat = new THREE.MeshPhysicalMaterial({
    color: colors.highlight,
    metalness: 0.16,
    roughness: 0.1,
    transmission: 0.6,
    thickness: 0.3,
    ior: 1.3,
    clearcoat: 1,
    envMapIntensity: 2,
  });
  const nodeGroup = new THREE.Group();
  root.add(nodeGroup);
  const nodes = [];
  const count = 60;
  for (let i = 0; i < count; i++) {
    const mesh = new THREE.Mesh(satelliteGeo, satelliteMat);
    nodeGroup.add(mesh);
    nodes.push(mesh);
  }
  const linePositions = new Float32Array(count * 6),
    lineGeometry = new THREE.BufferGeometry();
  lineGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(linePositions, 3),
  );
  const lineMaterial = new THREE.LineBasicMaterial({
    color: colors.edge,
    transparent: true,
    opacity: 0.38,
  });
  const lines = new THREE.LineSegments(lineGeometry, lineMaterial);
  root.add(lines);
  const rings = [];
  for (let i = 0; i < 3; i++) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(2.25 + i * 0.18, 0.009, 6, 150),
      new THREE.MeshPhysicalMaterial({
        color: colors.highlight,
        metalness: 0.7,
        roughness: 0.1,
        transparent: true,
        opacity: 0.6,
      }),
    );
    ring.rotation.set(0.6 + i * 0.7, 0.5 + i * 0.5, i * 0.8);
    root.add(ring);
    rings.push(ring);
  }

  const dissected = new THREE.Group();
  dissected.name = "ProfileCutaway";
  root.add(dissected);
  const shellMaterial = glassMaterial.clone();
  shellMaterial.color = colors.highlight.clone();
  shellMaterial.side = THREE.DoubleSide;
  shellMaterial.transmission = 0.94;
  const halfGeometry = new THREE.SphereGeometry(1.42, 56, 32, 0, Math.PI);
  const shellFront = new THREE.Mesh(halfGeometry, shellMaterial),
    shellBack = new THREE.Mesh(halfGeometry, shellMaterial);
  shellBack.rotation.y = Math.PI;
  dissected.add(shellFront, shellBack);
  const sliceGroups = [];
  const sliceMaterials = [colors.core, colors.pink, colors.blue].map(
    (color) =>
      new THREE.MeshPhysicalMaterial({
        color,
        metalness: 0.2,
        roughness: 0.15,
        clearcoat: 1,
        envMapIntensity: 2,
        transparent: true,
        opacity: 0.9,
      }),
  );
  const sliceGeometry = new THREE.BoxGeometry(0.19, 0.09, 0.19, 2, 2, 2);
  for (let layer = 0; layer < 3; layer++) {
    const group = new THREE.Group();
    group.name = ["HistoryLayer", "ContactsLayer", "GraphLayer"][layer];
    for (let i = 0; i < 20; i++) {
      const cube = new THREE.Mesh(sliceGeometry, sliceMaterials[layer]);
      cube.position.set(
        ((i % 5) - 2) * 0.28,
        0,
        (Math.floor(i / 5) - 1.5) * 0.28,
      );
      group.add(cube);
    }
    dissected.add(group);
    sliceGroups.push(group);
  }
  const shardGroup = new THREE.Group();
  root.add(shardGroup);
  const shardGeo = new THREE.OctahedronGeometry(0.13, 0);
  for (let i = 0; i < 8; i++) {
    const shard = new THREE.Mesh(shardGeo, shellMaterial);
    const a = (i * Math.PI) / 4;
    shard.position.set(Math.cos(a) * 3.2, Math.sin(a) * 2.5, Math.sin(i) * 1.2);
    shard.scale.set(0.6, 1.8, 0.55);
    shard.rotation.set(i * 0.7, i * 0.3, i * 0.9);
    shardGroup.add(shard);
  }

  const temp = new THREE.Vector3();
  const positions = [];
  for (let stage = 0; stage < 12; stage++) {
    const array = [];
    for (let i = 0; i < count; i++) {
      const phi = i * 2.399963,
        y = 1 - (2 * (i + 0.5)) / count,
        r = Math.sqrt(1 - y * y);
      let x = Math.cos(phi) * r * 2.4,
        z = Math.sin(phi) * r * 2.4,
        Y = y * 2.4;
      if (stage === 1) {
        x = Math.sin(i * 8.13) * 3.25;
        Y = Math.cos(i * 3.72) * 2.8;
        z = Math.sin(i * 1.78) * 2;
      }
      if (stage === 2 || stage === 7 || stage === 10) {
        const group = i % 3,
          angle = (group * Math.PI * 2) / 3;
        x = Math.cos(angle) * 1.7 + Math.cos(phi) * r * 0.85;
        Y = Math.sin(angle) * 1.7 + y * 0.85;
        z = Math.sin(phi) * r * 0.85;
      }
      if (stage === 3 || stage === 4) {
        const a = (i / count) * Math.PI * 2;
        x = Math.cos(a) * (1.1 + (i % 5) * 0.3);
        Y = Math.sin(a) * (1.1 + (i % 5) * 0.3);
        z = Math.sin(i * 0.4) * 0.7;
      }
      if (stage === 5) {
        x = ((i % 10) - 4.5) * 0.42;
        Y = (Math.floor(i / 10) - 2.5) * 0.48;
        z = Math.sin(i) * 0.12;
      }
      if (stage === 6) {
        x = ((i % 12) - 5.5) * 0.38;
        Y = (Math.floor(i / 12) - 2) * 0.5;
        z = Math.sin(i * 0.3) * 0.5;
      }
      if (stage === 8) {
        const group = Math.floor(i / 20);
        x = (group - 1) * 1.6 + Math.cos(phi) * r * 0.55;
        Y = y * 0.7;
        z = Math.sin(phi) * r * 0.55;
      }
      if (stage === 9) {
        x = ((i % 10) - 4.5) * 0.48;
        Y = (Math.floor(i / 10) - 2.5) * 0.42;
        z = Math.cos(i * 0.3) * 0.6;
      }
      array.push(new THREE.Vector3(x, Y, z));
    }
    positions.push(array);
  }
  let drag = false,
    dragX = 0,
    dragY = 0,
    lastX = 0,
    lastY = 0,
    mouseX = 0,
    mouseY = 0;
  document.addEventListener("pointerdown", (e) => {
    if (e.target.closest("a,button,input,summary,details,.glass")) return;

    drag = true;
    lastX = e.clientX;
    lastY = e.clientY;
  });
  addEventListener("pointerup", () => (drag = false));
  addEventListener("pointercancel", () => (drag = false));
  addEventListener(
    "pointermove",
    (e) => {
      mouseX = (e.clientX / innerWidth - 0.5) * 0.12;
      mouseY = (e.clientY / innerHeight - 0.5) * 0.1;
      if (drag) {
        dragX += (e.clientX - lastX) * 0.006;
        dragY += (e.clientY - lastY) * 0.006;
        lastX = e.clientX;
        lastY = e.clientY;
      }
    },
    { passive: true },
  );
  let lastTime = 0;
  const ambient = document.querySelector(".ambient");
  const paper = new THREE.Color(token("--paper"));
  const currentBackground = new THREE.Color();
  function frame(time) {
    requestAnimationFrame(frame);
    if (document.hidden) return;
    const dt = Math.min((time - lastTime) / 1000, 0.05) || 0.016;
    lastTime = time;
    const w = innerWidth,
      h = innerHeight;
    if (
      canvas.width !== Math.round(w * renderer.getPixelRatio()) ||
      canvas.height !== Math.round(h * renderer.getPixelRatio())
    ) {
      renderer.setSize(w, h, false);
      camera.aspect = w / h;
      camera.updateProjectionMatrix();
    }
    sceneProgress = lerp(
      sceneProgress,
      targetScene,
      motionPreference.matches ? 1 : Math.min(1, dt * 4),
    );
    const stage = Math.floor(clamp(sceneProgress, 0, 11)),
      next = Math.min(11, stage + 1),
      blend = sceneProgress - stage;
    const mobile = w < 769;
    const intro = clamp(1 - sceneProgress);
    const darkPhase = ease(clamp(1 - Math.abs(sceneProgress - 9)));
    currentBackground.copy(paper).lerp(colors.dark, darkPhase);
    ambient.style.backgroundColor = "#" + currentBackground.getHexString();
    ambient.style.backgroundImage = darkPhase > 0.5 ? "none" : "";
    const redField =
      ["home", "product"].some((id) => {
        const r = $(id).getBoundingClientRect();
        return r.top < 80 && r.bottom > 80;
      }) || document.querySelector(".closing").getBoundingClientRect().top < 80;
    document.body.style.setProperty(
      "--nav-glass",
      redField
        ? token("--nav-glass-red")
        : darkPhase > 0.5
          ? token("--nav-glass-dark")
          : token("--nav-glass-light"),
    );
    document.querySelector(".site-header").style.color = redField
      ? token("--white")
      : darkPhase > 0.5
        ? token("--dark-text")
        : token("--ink");
    const poses = [
      { s: 0, x: 0, y: 0, k: 1.12, rx: 0, ry: -0.2 },
      { s: 1, x: 2.7, y: 0.1, k: 0.94, rx: 0.12, ry: 0.4 },
      { s: 2, x: 0, y: 0.3, k: 0.9, rx: 0.1, ry: 0.8 },
      { s: 3, x: 2.6, y: 0, k: 1.05, rx: -0.12, ry: 0.4 },
      { s: 4, x: 2.75, y: 0.1, k: 0.9, rx: 0.05, ry: 0 },
      { s: 5, x: 2.6, y: 0, k: 1, rx: 0.16, ry: 0.55 },
      { s: 6, x: 2.65, y: 0, k: 1.05, rx: -0.1, ry: 0.9 },
      { s: 7, x: 2.7, y: 0, k: 0.97, rx: 0.08, ry: 0.4 },
      { s: 7.4, x: 2.65, y: 0, k: 1.15, rx: 0.1, ry: 0.9 },
      { s: 7.7, x: 2.7, y: 0, k: 1, rx: 0.04, ry: 0.4 },
      { s: 8, x: 0, y: 0, k: 0.75, rx: 0, ry: 0 },
      { s: 9, x: 0, y: 0, k: 0.6, rx: 0, ry: 0 },
      { s: 11, x: 2.6, y: 0, k: 0.9, rx: 0.1, ry: 0.4 },
    ];
    let poseIndex = 0;
    poses.forEach((p, i) => {
      if (sceneProgress >= p.s) poseIndex = i;
    });
    const poseA = poses[poseIndex],
      poseB = poses[Math.min(poses.length - 1, poseIndex + 1)],
      poseT = ease(
        clamp((sceneProgress - poseA.s) / Math.max(0.001, poseB.s - poseA.s)),
      );
    const px = lerp(poseA.x, poseB.x, poseT),
      py = lerp(poseA.y, poseB.y, poseT),
      pk = lerp(poseA.k, poseB.k, poseT);
    const targetX = mobile ? px * 0.18 : px,
      targetY = mobile ? (intro ? 0 : -1.9) : py;
    root.position.x = lerp(root.position.x, targetX, dt * 4);
    root.position.y = lerp(root.position.y, targetY, dt * 4);
    root.scale.setScalar(lerp(root.scale.x, pk * (mobile ? 0.78 : 1), dt * 4));
    root.visible = sceneProgress < 8.95;
    const auto = 0;
    root.rotation.y = lerp(
      root.rotation.y,
      dragX + mouseX + lerp(poseA.ry, poseB.ry, poseT),
      dt * 2,
    );
    root.rotation.x = lerp(
      root.rotation.x,
      dragY + mouseY + lerp(poseA.rx, poseB.rx, poseT),
      dt * 2,
    );
    organicTime.value = motionPreference.matches ? 0 : time * 0.0005;
    core.rotation.y = 0;
    core.rotation.z = 0;
    const coreScale =
      sceneProgress < 1
        ? 1
        : sceneProgress < 3
          ? 0.5
          : sceneProgress < 5
            ? 0.7
            : sceneProgress < 8
              ? 0.27
              : sceneProgress < 9
                ? 0.45
                : 0.3;
    const cutAmount =
      clamp((sceneProgress - 7.02) / 0.22) *
      clamp((8.15 - sceneProgress) / 0.22);
    dissected.visible = cutAmount > 0.01;
    dissected.scale.setScalar(0.78 + cutAmount * 0.22);
    shellFront.position.z = cutAmount * 0.95;
    shellBack.position.z = -cutAmount * 0.95;
    shellFront.position.x = cutAmount * 0.35;
    shellBack.position.x = -cutAmount * 0.35;
    sliceGroups.forEach((group, i) => {
      group.position.y = (1 - i) * (0.22 + cutAmount * 0.43);
      group.position.x = lerp(
        group.position.x,
        i === cutawayLayer ? cutAmount * 0.3 : 0,
        dt * 5,
      );
      group.rotation.y = sceneProgress * 0.12;
      const scale = i === cutawayLayer ? 1.13 : 1;
      group.scale.setScalar(lerp(group.scale.x, scale, dt * 4));
    });
    core.visible = cutAmount < 0.95;
    inner.visible = cutAmount < 0.95;
    core.scale.setScalar(
      lerp(core.scale.x, coreScale * (1 - cutAmount * 0.85), dt * 3),
    );
    shardGroup.visible = sceneProgress < 3;
    shardGroup.rotation.y = auto * 0.25;

    inner.scale.setScalar(
      lerp(inner.scale.x, sceneProgress < 1 ? 1 : 0.55, dt * 3),
    );
    lineMaterial.opacity = lerp(0.08, 0.35, clamp(sceneProgress));
    nodeGroup.rotation.y = 0;
    nodes.forEach((node, i) => {
      temp.copy(positions[stage][i]).lerp(positions[next][i], blend);
      if (!motionPreference.matches) {
        temp.y += Math.sin(time * 0.0006 + i) * 0.025;
      }
      node.position.copy(temp);
      node.scale.setScalar(
        (i % 11 === 0 ? 1.8 : 1) * (sceneProgress < 1 ? 0.65 : 1),
      );
      const k = i * 6;
      linePositions[k] = temp.x;
      linePositions[k + 1] = temp.y;
      linePositions[k + 2] = temp.z;
      const peer = nodes[(i + 7) % count].position;
      linePositions[k + 3] = peer.x;
      linePositions[k + 4] = peer.y;
      linePositions[k + 5] = peer.z;
    });
    lineGeometry.attributes.position.needsUpdate = true;
    rings.forEach((ring, i) => {
      ring.rotation.z = sceneProgress * 0.2 + i * 0.8;
      ring.material.opacity = lerp(
        ring.material.opacity,
        sceneProgress < 1 ? 0.65 : sceneProgress > 7 ? 0.08 : 0.22,
        dt * 3,
      );
    });
    renderer.render(scene, camera);
  }
  requestAnimationFrame(frame);
  canvas.addEventListener("webglcontextlost", (e) => {
    e.preventDefault();
    document.body.classList.add("no-webgl");
  });
}
measure();
initWorld();
