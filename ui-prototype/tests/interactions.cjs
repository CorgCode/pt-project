const { JSDOM } = require("jsdom"),
  fs = require("fs"),
  assert = require("node:assert/strict");
const root = require("node:path").resolve(__dirname, "..") + "/";
const html = fs
  .readFileSync(root + "page.html", "utf8")
  .replace(/__(?:CSS|FONTS|THREE|RGBE|CHART|HDR|DATA|APP|LAB)__/g, "");
const dom = new JSDOM(html, {
    runScripts: "outside-only",
    url: "https://example.test/",
  }),
  w = dom.window;
w.matchMedia = () => ({ matches: true });
w.IntersectionObserver = class {
  observe() {}
  unobserve() {}
};
w.Chart = class {
  static defaults = { font: {} };
  constructor(canvas, config) {
    this.data = config.data;
    this.options = config.options;
  }
  update() {}
};
let audioCreated = 0;
w.AudioContext = class {
  constructor() {
    audioCreated++;
    this.currentTime = 0;
    this.destination = {};
    this.state = "running";
  }
  createGain() {
    return {
      gain: { value: 0, cancelScheduledValues() {}, setTargetAtTime() {} },
      connect() {},
    };
  }
  createBiquadFilter() {
    return { frequency: {}, Q: {}, connect() {} };
  }
  createOscillator() {
    return { frequency: {}, connect() {}, start() {} };
  }
  resume() {
    return Promise.resolve();
  }
};
w.eval(fs.readFileSync(root + "data.js", "utf8"));
w.eval(
  fs.readFileSync(root + "app.js", "utf8") +
    "\n" +
    fs.readFileSync(root + "lab.js", "utf8") +
    "\nwindow.__qa={parseLabCSV,projectProfiles,loadLabFile,getDataset:()=>labDataset,getCharts:()=>labCharts};",
);
const $ = (id) => w.document.getElementById(id);
(async () => {
  await new Promise((r) => setTimeout(r, 0));
  assert.equal(audioCreated, 1);
  assert.equal($("sound").getAttribute("aria-pressed"), "true");
  assert.equal($("sound").dataset.playing, "true");
  $("sound").click();
  assert.equal($("sound").getAttribute("aria-pressed"), "false");
  $("sound").click();
  await new Promise((r) => setTimeout(r, 0));
  assert.equal($("sound").dataset.playing, "true");
  assert.equal(audioCreated, 1);
  assert.equal($("demo-result").hidden, true);
  $("demo-action").click();
  assert.match($("demo-heading").textContent, /самый близкий/);
  const source = w.HOSTS[0],
    norm = (h) => Math.hypot(...h.v),
    cos = (b) =>
      source.v.reduce((s, x, i) => s + x * b.v[i], 0) /
      (norm(source) * norm(b));
  const nearest = w.HOSTS.slice(1).sort((a, b) => cos(b) - cos(a))[0];
  assert.equal($("demo-score").textContent, cos(nearest).toFixed(3));
  $("demo-nodes").children[2].dispatchEvent(
    new w.MouseEvent("click", { bubbles: true }),
  );
  assert.match($("demo-heading").textContent, /Хост C/);
  assert.equal($("host-id").textContent, source.id);
  $("reanchor").click();
  assert.notEqual($("host-id").textContent, source.id);
  assert.equal($("demo-result").hidden, true);
  w.document.querySelector('[data-layer="2"]').click();
  assert.equal($("layer-title").textContent, "Структура графа");
  w.document.querySelector('[data-metric="1"]').click();
  assert.equal($("metric-value").textContent, "0,974");
  assert.match($("metric-caution").textContent, /нельзя читать/);
  assert.equal($("lab-host").options.length, 300);
  assert.equal(w.__qa.getCharts().map.data.datasets.length, 3);
  w.document.querySelector('[data-preset="small"]').click();
  assert.equal($("lab-host").options.length, 24);
  w.document.querySelector('[data-preset="flows"]').click();
  assert.equal($("lab-host").options.length, 11);
  assert.match($("lab-mode").textContent, /Быстрый/);
  assert(
    w.__qa.getDataset().every((h) => Number.isFinite(h.norm) && h.norm > 0),
  );
  const parse = w.__qa.parseLabCSV;
  let imported = parse('host_id;emb_0;emb_1\n"a;b";1;2\nc;2;1');
  assert.equal(imported.dataset[0].id, "a;b");
  assert.equal(imported.dataset.length, 2);
  assert.throws(() => parse("host_id,emb_0\na,0\nb,1"), /Нулевые/);
  assert.throws(() => parse("host_id,emb_0\na,1\na,2"), /уникальны/);
  assert.throws(() => parse("host_id,emb_0\na,NaN\nb,1"), /конечные/);
  assert.throws(() => parse("src,dst\na,b,c"), /Число полей/);
  assert.equal(parse("src,dst\na,b").dataset.length, 2);
  assert.throws(() => parse("src,dst,bytes\na,b,-1"), /неотрицательными/);
  const pca = w.__qa.projectProfiles(imported.dataset);
  assert.equal(pca.length, 2);
  assert(pca.every((p) => Number.isFinite(p.x) && Number.isFinite(p.y)));
  await w.__qa.loadLabFile({
    name: "own.csv",
    size: 50,
    text: async () => "host_id,emb_0,emb_1\nalpha,1,0\nbeta,1,1\ngamma,0,1",
  });
  assert.equal($("lab-host").options.length, 3);
  assert.match($("lab-status").textContent, /own.csv/);
  $("lab-host").value = "gamma";
  $("lab-host").dispatchEvent(new w.Event("change"));
  assert.equal(w.__qa.getCharts().map.data.datasets[2].data[0].hostId, "gamma");
  await w.__qa.loadLabFile({
    name: "bad.csv",
    size: 20,
    text: async () => "not,a,csv",
  });
  assert.match($("lab-status").textContent, /Не удалось/);
  assert.equal($("lab-host").options.length, 3);
  await w.__qa.loadLabFile({ size: 4 * 1024 * 1024 });
  assert.match($("lab-status").textContent, /слишком большой/);
  console.log(
    "PASS: autoplay request, mute, demo comparison/reanchor, layer inspector, plain-language metrics, live charts, presets, CSV vectors/flows, projection, invalid input, upload and dataset preservation",
  );
  dom.window.close();
})().catch((e) => {
  console.error(e);
  dom.window.close();
  process.exitCode = 1;
});
