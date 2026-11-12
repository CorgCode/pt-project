window.__ptStartLab = function () {
if (window.__ptLabStarted) return;
window.__ptLabStarted = true;
const $ = (id) => document.getElementById(id);
const hosts = window.HOSTS.map(h => ({...h, norm: Math.hypot(...h.v)}));
const cosine = (a,b) => a.v.reduce((sum,x,i) => sum+x*b.v[i],0)/(a.norm*b.norm);
const measure = () => window.__ptRefreshLayout?.();
// Browser-local analysis. Uploaded data never leaves this page.
let labDataset = hosts.map((h) => ({ id: h.id, v: [...h.v], norm: h.norm }));
let labSelected = labDataset[0],
  labProjection = [],
  labCharts = { map: null, neighbors: null },
  labStats = null,
  labKind = "project";
const sampleFlowCSV =
  "src,dst,bytes,packets,dst_port,proto\n" +
  Array.from({ length: 80 }, (_, i) => {
    const src = "device-" + String((i % 8) + 1).padStart(2, "0"),
      dst = "service-" + ((i % 3) + 1);
    return `${src},${dst},${((i % 7) + 1) * 240},${(i % 9) + 1},${[443, 53, 22][i % 3]},${i % 3 === 1 ? "UDP" : "TCP"}`;
  }).join("\n");
function csvRows(text) {
  const first = text.replace(/^\uFEFF/, "").split(/\r?\n/, 1)[0];
  const delimiter = [";", ",", "\t"].sort(
    (a, b) => first.split(b).length - first.split(a).length,
  )[0];
  const rows = [];
  let row = [],
    cell = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (quoted || cell.trim() === "") quoted = !quoted;
      else throw Error("Кавычка внутри незаключённого поля CSV.");
    } else if (ch === delimiter && !quoted) {
      row.push(cell.trim());
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell.trim());
      if (row.some((x) => x !== "")) rows.push(row);
      row = [];
      cell = "";
    } else cell += ch;
    if (rows.length > 10000)
      throw Error("Для браузерного примера допустимо до 10 000 строк.");
  }
  if (quoted) throw Error("В CSV не закрыты кавычки.");
  row.push(cell.trim());
  if (row.some((x) => x !== "")) rows.push(row);
  if (rows.length < 2)
    throw Error("Нужны заголовок и хотя бы одна строка данных.");
  rows[0][0] = rows[0][0].replace(/^\uFEFF/, "");
  const columns = rows[0].map((x) => x.trim().toLowerCase());
  if (new Set(columns).size !== columns.length)
    throw Error("Имена столбцов повторяются.");
  for (const r of rows.slice(1))
    if (r.length !== columns.length)
      throw Error("Число полей в строках не совпадает с заголовком.");
  return { columns, rows: rows.slice(1) };
}
function parseLabCSV(text) {
  const parsed = csvRows(text),
    { columns, rows } = parsed;
  const hi = columns.indexOf("host_id");
  if (hi >= 0) {
    const dims = columns
      .map((name, i) => ({ name, i }))
      .filter((x) => /^emb_\d+$/.test(x.name))
      .sort((a, b) => Number(a.name.slice(4)) - Number(b.name.slice(4)));
    if (!dims.length || dims.length > 256)
      throw Error("Ожидаются от 1 до 256 столбцов emb_0, emb_1 и далее.");
    if (rows.length > 500)
      throw Error("Для интерактивного примера допустимо до 500 хостов.");
    const ids = new Set();
    const dataset = rows.map((row) => {
      const id = row[hi];
      if (!id || ids.has(id))
        throw Error("ID хостов должны быть заполнены и уникальны.");
      ids.add(id);
      const v = dims.map((d) => {
        if (row[d.i] === "") throw Error("В векторе есть пустое значение.");
        const x = Number(row[d.i]);
        if (!Number.isFinite(x))
          throw Error("Векторы должны содержать только конечные числа.");
        return x;
      });
      const norm = Math.hypot(...v);
      if (!norm || !Number.isFinite(norm))
        throw Error(
          "Нулевые или слишком большие векторы нельзя сравнить по косинусу.",
        );
      return { id, v, norm };
    });
    if (dataset.length < 2)
      throw Error("Для сравнения нужны хотя бы два хоста.");
    return {
      dataset,
      stats: { hosts: dataset.length, dimensions: dims.length },
      kind: "vectors",
    };
  }
  const si = ["src", "source", "src_host"]
      .map((k) => columns.indexOf(k))
      .find((i) => i >= 0),
    di = ["dst", "target", "dst_host"]
      .map((k) => columns.indexOf(k))
      .find((i) => i >= 0);
  if (si === undefined || di === undefined)
    throw Error("Нужны столбцы src и dst либо host_id и emb_0, emb_1…");
  const bi = columns.indexOf("bytes"),
    pi = columns.indexOf("packets"),
    porti = columns.indexOf("dst_port"),
    proti = columns.indexOf("proto");
  const map = new Map();
  const get = (id) => {
    if (!id) throw Error("src и dst не должны быть пустыми.");
    if (!map.has(id))
      map.set(id, { id, v: new Array(20).fill(0), peers: new Set() });
    if (map.size > 500) throw Error("В примере поддерживается до 500 хостов.");
    return map.get(id);
  };
  let totalBytes = 0;
  const numeric = (row, index) => {
    if (index < 0) return 0;
    const x = Number(row[index]);
    if (!Number.isFinite(x) || x < 0)
      throw Error("bytes и packets должны быть неотрицательными числами.");
    return x;
  };
  for (const row of rows) {
    const a = get(row[si]),
      b = get(row[di]),
      bytes = numeric(row, bi),
      packets = numeric(row, pi);
    a.v[0]++;
    b.v[1]++;
    a.v[2] += bytes;
    b.v[3] += bytes;
    a.v[4] += packets;
    b.v[5] += packets;
    a.peers.add(b.id);
    b.peers.add(a.id);
    const service =
      (proti >= 0 ? row[proti] : "") + ":" + (porti >= 0 ? row[porti] : "");
    let hash = 0;
    for (const ch of service) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    a.v[8 + (hash % 6)]++;
    b.v[14 + (hash % 6)]++;
    totalBytes += bytes;
  }
  const dataset = [...map.values()].map((h) => {
    h.v[6] = h.peers.size;
    h.v[7] = h.v[0] + h.v[1];
    h.v = h.v.map(Math.log1p);
    return { id: h.id, v: h.v };
  });
  if (dataset.length < 2) throw Error("Для сравнения нужны хотя бы два хоста.");
  const scales = dataset[0].v.map((_, i) =>
    Math.max(1, ...dataset.map((h) => h.v[i])),
  );
  dataset.forEach((h) => {
    h.v = h.v.map((x, i) => x / scales[i]);
    h.norm = Math.hypot(...h.v);
  });
  return {
    dataset,
    stats: {
      hosts: dataset.length,
      flows: rows.length,
      bytes: totalBytes,
      dimensions: 20,
    },
    kind: "flows",
  };
}
function projectProfiles(dataset) {
  const n = dataset.length,
    d = dataset[0].v.length,
    mean = new Array(d).fill(0);
  dataset.forEach((h) => h.v.forEach((x, j) => (mean[j] += x / n)));
  const x = dataset.map((h) => h.v.map((v, j) => v - mean[j]));
  function axis(previous) {
    let v = Array.from({ length: d }, (_, j) =>
      Math.sin(j * 1.7 + (previous ? 1.1 : 0.4)),
    );
    for (let iter = 0; iter < 18; iter++) {
      let next = new Array(d).fill(0);
      for (const row of x) {
        const dot = row.reduce((s, z, j) => s + z * v[j], 0);
        row.forEach((z, j) => (next[j] += z * dot));
      }
      if (previous) {
        const dot = next.reduce((s, z, j) => s + z * previous[j], 0);
        next = next.map((z, j) => z - dot * previous[j]);
      }
      const norm = Math.hypot(...next);
      if (norm < 1e-12) return new Array(d).fill(0);
      v = next.map((z) => z / norm);
    }
    return v;
  }
  const a = axis(null),
    b = axis(a);
  return x.map((row, i) => ({
    x: row.reduce((s, z, j) => s + z * a[j], 0),
    y: row.reduce((s, z, j) => s + z * b[j], 0),
    hostId: dataset[i].id,
  }));
}
function labToken(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim();
}
function chooseLabHost(id) {
  labSelected = labDataset.find((h) => h.id === id) || labDataset[0];
  $("lab-host").value = labSelected.id;
  renderLab();
}
window.__ptSelectHost = chooseLabHost;
function installLab(data, label) {
  labDataset = data.dataset;
  labStats = data.stats;
  labKind = data.kind;
  labProjection = projectProfiles(labDataset);
  $("lab-host").replaceChildren(
    ...labDataset.map((h) => {
      const option = document.createElement("option");
      option.value = h.id;
      option.textContent = h.id;
      return option;
    }),
  );
  $("lab-status").textContent = `${label} · ${labDataset.length} хостов`;
  $("lab-mode").textContent =
    data.kind === "flows"
      ? "Быстрый профиль по взаимодействиям"
      : "Сравнение готовых векторов";
  chooseLabHost(labDataset[0].id);
  measure();
}
function renderLab() {
  const near = labDataset
    .filter((h) => h !== labSelected)
    .map((host) => ({ host, score: cosine(labSelected, host) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, 5);
  $("lab-neighbor-buttons").replaceChildren(
    ...near.map((n) => {
      const b = document.createElement("button");
      b.textContent = n.host.id + " ↗";
      b.addEventListener("click", () => chooseLabHost(n.host.id));
      return b;
    }),
  );
  $("lab-insight-title").textContent = `Ближе всего — ${near[0].host.id}`;
  $("lab-insight-copy").textContent =
    `Сходство с выбранным хостом: ${near[0].score.toFixed(3)}. Мы сравнили все доступные профили. Выберите точку на карте или одного из соседей, чтобы продолжить исследование.`;
  const stats = labStats || {
    hosts: labDataset.length,
    dimensions: labDataset[0].v.length,
  };
  const summary = [
    [stats.hosts, "хостов в наборе"],
    [stats.dimensions, "числовых признаков"],
  ];
  if (stats.flows) summary.push([stats.flows, "сетевых записей"]);
  $("lab-summary").replaceChildren(
    ...summary.map(([value, label]) => {
      const div = document.createElement("div"),
        n = document.createElement("strong"),
        l = document.createElement("small");
      n.textContent = String(value);
      l.textContent = label;
      div.append(n, l);
      return div;
    }),
  );
  window.__ptLabState = {dataset:labDataset, selected:labSelected, projection:labProjection, near, stats, kind:labKind};
  window.dispatchEvent(new CustomEvent("pt:lab-update", {detail:window.__ptLabState}));
  if (!window.Chart) return;
  Chart.defaults.font.family = labToken("--font-body").replaceAll("'", "");
  Chart.defaults.color = labToken("--muted");
  const nearIds = new Set(near.map((n) => n.host.id));
  const base = labProjection.filter(
    (p) => p.hostId !== labSelected.id && !nearIds.has(p.hostId),
  );
  const neighborPoints = labProjection.filter((p) => nearIds.has(p.hostId));
  const current = labProjection.filter((p) => p.hostId === labSelected.id);
  const datasets = [
    {
      label: "Другие хосты",
      data: base,
      pointRadius: 3,
      pointHoverRadius: 6,
      backgroundColor: labToken("--bar-muted"),
    },
    {
      label: "Ближайшие",
      data: neighborPoints,
      pointRadius: 6,
      pointHoverRadius: 9,
      backgroundColor: labToken("--accent"),
    },
    {
      label: "Выбранный",
      data: current,
      pointRadius: 9,
      pointHoverRadius: 11,
      backgroundColor: labToken("--ink"),
    },
  ];
  const animation = matchMedia("(prefers-reduced-motion: reduce)").matches ? false : {duration: 650};
  if (!labCharts.map) {
    labCharts.map = new Chart($("lab-map-chart"), {
      type: "scatter",
      data: { datasets },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => ctx.raw.hostId } },
        },
        scales: { x: { display: false }, y: { display: false } },
        onClick: (event, elements) => {
          if (elements.length) {
            const p =
              labCharts.map.data.datasets[elements[0].datasetIndex].data[
                elements[0].index
              ];
            chooseLabHost(p.hostId);
          }
        },
      },
    });
  } else {
    labCharts.map.data.datasets = datasets;
    labCharts.map.update();
  }
  const bars = {
    labels: near.map((n) => n.host.id),
    datasets: [
      {
        data: near.map((n) => n.score),
        backgroundColor: near.map((_, i) =>
          labToken(i === 0 ? "--accent" : "--bar-muted"),
        ),
        borderRadius: 3,
        barThickness: 13,
      },
    ],
  };
  if (!labCharts.neighbors) {
    labCharts.neighbors = new Chart($("lab-neighbors-chart"), {
      type: "bar",
      data: bars,
      options: {
        indexAxis: "y",
        responsive: true,
        maintainAspectRatio: false,
        animation,
        plugins: {
          legend: { display: false },
          tooltip: { callbacks: { label: (ctx) => ctx.raw.toFixed(3) } },
        },
        scales: {
          x: {
            min: -1,
            max: 1,
            grid: { color: labToken("--line") },
            ticks: { font: { size: 9 } },
          },
          y: { grid: { display: false }, ticks: { font: { size: 10 } } },
        },
      },
    });
  } else {
    labCharts.neighbors.data = bars;
    labCharts.neighbors.update();
  }
}
function markPreset(value) {
  document
    .querySelectorAll("[data-preset]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.preset === value)),
    );
}
document.querySelectorAll("[data-preset]").forEach((b) =>
  b.addEventListener("click", () => {
    markPreset(b.dataset.preset);
    if (b.dataset.preset === "flows") {
      installLab(parseLabCSV(sampleFlowCSV), "Учебные взаимодействия");
      return;
    }
    const dataset = hosts
      .slice(0, b.dataset.preset === "small" ? 24 : hosts.length)
      .map((h) => ({ id: h.id, v: [...h.v], norm: h.norm }));
    installLab(
      {
        dataset,
        stats: { hosts: dataset.length, dimensions: dataset[0].v.length },
        kind: "project",
      },
      b.dataset.preset === "small"
        ? "Первые 24 хоста проекта"
        : "Набор проекта",
    );
  }),
);
$("lab-host").addEventListener("change", () =>
  chooseLabHost($("lab-host").value),
);
async function loadLabFile(file) {
  if (!file) return;
  if (file.size > 3 * 1024 * 1024) {
    $("lab-status").textContent =
      "Файл слишком большой. Поддерживается CSV до 3 МБ.";
    return;
  }
  try {
    $("lab-status").textContent = "Читаем файл…";
    const text = await file.text();
    const parsed = parseLabCSV(text);
    installLab(parsed, file.name);
    markPreset("uploaded");
  } catch (e) {
    $("lab-status").textContent = "Не удалось прочитать файл: " + e.message;
  }
}
$("csv-file").addEventListener("change", (e) => loadLabFile(e.target.files[0]));
const drop = $("upload-zone");
drop.addEventListener("dragover", (e) => {
  e.preventDefault();
  drop.classList.add("drag-over");
});
drop.addEventListener("dragleave", () => drop.classList.remove("drag-over"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("drag-over");
  loadLabFile(e.dataTransfer.files[0]);
});
$("download-sample").addEventListener("click", () => {
  const url = URL.createObjectURL(
      new Blob([sampleFlowCSV], { type: "text/csv;charset=utf-8" }),
    ),
    a = document.createElement("a");
  a.href = url;
  a.download = "example-network.csv";
  a.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
});
installLab(
  {
    dataset: labDataset,
    stats: { hosts: labDataset.length, dimensions: labDataset[0].v.length },
    kind: "project",
  },
  "Набор проекта",
);

};
