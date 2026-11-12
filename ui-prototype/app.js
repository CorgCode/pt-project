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
  demoStep = 0;
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
function renderNetwork() {
  const lines = $("demo-lines"),
    nodes = $("demo-nodes");
  lines.replaceChildren();
  nodes.replaceChildren();
  demoPoints.slice(1).forEach((p, i) => {
    lines.append(
      svgEl("line", {
        x1: 280,
        y1: 180,
        x2: p.x,
        y2: p.y,
        class: "demo-line" + (demoStep > 0 && i === 0 ? " match" : ""),
      }),
    );
  });
  demoPoints.forEach((p, i) => {
    const g = svgEl("g", {
      class: "demo-node",
      tabindex: i === 0 ? -1 : 0,
      role: "button",
      "aria-label": i === 0 ? "Выбранный хост" : `Исследовать ${p.label}`,
    });
    const circle = svgEl("circle", {
      cx: p.x,
      cy: p.y,
      r: demoStep > 0 && i === 1 ? 28 : p.r,
      opacity: demoStep === 0 && i > 0 ? 0.42 : 1,
    });
    g.append(circle);
    const text = svgEl("text", {
      x: p.x,
      y: p.y + p.r + 25,
      class: i === 0 ? "main-label" : "",
    });
    text.textContent = p.label;
    g.append(text);
    if (i > 0) {
      const select = () => chooseHost(closest[i - 1].host);
      g.addEventListener("click", select);
      g.addEventListener("keydown", (e) => {
        if (e.key === "Enter" || e.key === " ") {
          e.preventDefault();
          select();
        }
      });
    }
    nodes.append(g);
  });
}
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
  $("demo-heading").textContent = d[1];
  $("demo-description").textContent = d[2];
  $("demo-action").replaceChildren(document.createTextNode(d[3]));
  const arrow = document.createElement("span");
  arrow.textContent = "↗";
  $("demo-action").append(arrow);
  $("demo-caption").textContent = d[4];
  $("host-id").textContent = selected.id;
  $("demo-result").hidden = demoStep === 0;
  $("demo-score").textContent = closest[0].score.toFixed(3);
  renderNetwork();
}
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
// Quiet original ambient sound. Starts only after an explicit click.
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
$("sound").addEventListener("click", async () => {
  try {
    if (!audio) audio = createAudio();
    if (!audio) {
      $("audio-status").textContent = "Аудио не поддерживается этим браузером";
      return;
    }
    await audio.ctx.resume();
    audioEnabled = !audioEnabled;
    audio.master.gain.cancelScheduledValues(audio.ctx.currentTime);
    audio.master.gain.setTargetAtTime(
      audioEnabled ? 0.45 : 0,
      audio.ctx.currentTime,
      audioEnabled ? 1.2 : 0.08,
    );
    $("sound").setAttribute("aria-pressed", String(audioEnabled));
    $("sound").setAttribute(
      "aria-label",
      audioEnabled ? "Выключить музыку" : "Включить спокойную музыку",
    );
    $("sound-label").textContent = audioEnabled ? "Звук вкл." : "Звук выкл.";
    $("audio-status").textContent = audioEnabled
      ? "Музыка включена"
      : "Музыка выключена";
  } catch {
    $("audio-status").textContent = "Не удалось включить звук";
  }
});
document.addEventListener("visibilitychange", () => {
  if (audio)
    audio.master.gain.setTargetAtTime(
      document.hidden ? 0 : audioEnabled ? 0.45 : 0,
      audio.ctx.currentTime,
      0.5,
    );
});
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
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 1.7));
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
  const core = new THREE.Mesh(
    new THREE.TorusKnotGeometry(1.08, 0.35, 150, 20, 2, 3),
    glassMaterial,
  );
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
  const satelliteGeo = new THREE.IcosahedronGeometry(0.11, 2);
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
    if (scrollY > innerHeight * 1.1) return;
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
    document.body.style.setProperty(
      "--nav-glass",
      darkPhase > 0.5 ? token("--nav-glass-dark") : token("--nav-glass-light"),
    );
    document.querySelector(".site-header").style.color =
      darkPhase > 0.5 ? token("--dark-text") : token("--ink");
    const rightStage = sceneProgress > 0.5 && sceneProgress < 7.8;
    let targetX = rightStage ? (mobile ? 0.6 : 3.1) : 0;
    let targetY = mobile ? (intro ? -0.1 : -1.9) : 0;
    if (sceneProgress > 7.8) targetY = mobile ? -1.3 : -0.5;
    root.position.x = lerp(root.position.x, targetX, dt * 3);
    root.position.y = lerp(root.position.y, targetY, dt * 3);
    const baseScale = mobile ? (intro ? 0.87 : 0.64) : intro ? 1.12 : 0.94;
    root.scale.setScalar(lerp(root.scale.x, baseScale, dt * 3));
    const auto = motionPreference.matches ? 0 : time * 0.00006;
    root.rotation.y = lerp(root.rotation.y, dragX + mouseX + auto, dt * 2);
    root.rotation.x = lerp(
      root.rotation.x,
      dragY + mouseY + Math.sin(sceneProgress * 0.4) * 0.12,
      dt * 2,
    );
    core.rotation.y = motionPreference.matches ? 0 : time * 0.00013;
    core.rotation.z = sceneProgress * 0.2;
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
    core.scale.setScalar(lerp(core.scale.x, coreScale, dt * 3));
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
