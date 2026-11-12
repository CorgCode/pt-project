const fs = require("fs"),
  assert = require("node:assert/strict"),
  path = require("node:path");
const { JSDOM } = require("jsdom");
const THREE = require("../../ui-prototype/vendor/three.min.js");
const root = path.resolve(__dirname, "..");
(async () => {
const dom = new JSDOM(fs.readFileSync(path.join(root, "index.html"), "utf8"), {
    runScripts: "outside-only",
    pretendToBeVisual: true,
    url: "http://127.0.0.1:8845/",
  }),
  w = dom.window;
w.HTMLDialogElement.prototype.showModal = function () {
  this.setAttribute("open", "");
};
w.HTMLDialogElement.prototype.close = function () {
  this.removeAttribute("open");
  this.dispatchEvent(new w.Event("close"));
};
const callbacks = [];
w.requestAnimationFrame = (fn) => callbacks.push(fn);
w.matchMedia = () => ({ matches: true });
let scroll = 0;
w.gsap = {
  core: {
    globals: () => ({
      ScrollSmoother: {
        get: () => ({
          scrollTop: () => scroll,
          scrollTo: (y) => {
            scroll = y;
          },
        }),
      },
    }),
  },
};
const scene = w.document.createElement("div");
scene.id = "home-scene";
scene.style.opacity = "1";
w.document.body.append(scene);
w.eval(fs.readFileSync(path.join(root, "pt-adapter.js"), "utf8"));
w.document.dispatchEvent(new w.Event("DOMContentLoaded"));
w.eval(fs.readFileSync(path.join(root, "product-data.js"), "utf8"));
let source = fs
  .readFileSync(path.join(root, "pt-story.js"), "utf8")
  .replace(/^import[^\n]+\n/, "");
w.eval(fs.readFileSync(path.join(root, "pt-lab.js"), "utf8"));
w.eval(fs.readFileSync(path.join(root, "pt-content.js"), "utf8"));
assert.equal(w.document.getElementById("pt-guide"), null, "wait for hydration");
w.document.getElementById("__nuxt").__vue_app__ = {};
w.__ptMountContent();
w.eval(source);
const frame = () => {
  assert(callbacks.length);
  callbacks.shift()();
};
const $ = (id) => w.document.getElementById(id);
for (const [y, name] of [
  [1000, "problem"],
  [2500, "product"],
  [3800, "input"],
  [4800, "encoder"],
  [6000, "architecture"],
  [7200, "training"],
  [8200, "results"],
  [9300, "demo"],
]) {
  scroll = y;
  frame();
  assert.equal(w.document.querySelector(".pt-story-panel").dataset.step, name);
  assert(!$("pt-story").hidden);
}
assert.match($("pt-mini-demo").textContent, /Хост A/);
w.document.querySelector(".pt-mini-controls button").click();
assert.match($("pt-mini-demo").textContent, /Косинусное сходство/);
w.document.querySelector(".pt-story-more").click();
assert.equal(scroll.id, "pt-detail-demo");
assert.equal(w.document.querySelectorAll("iframe, dialog").length, 0);
assert.equal(w.document.querySelectorAll("#pt-guide").length, 1);
assert.equal(w.document.querySelectorAll("#lab-host option").length, 300);
w.document.querySelector('[data-preset="flows"]').click();
assert.match($("lab-status").textContent, /Учебные взаимодействия/);
assert.equal(w.document.querySelectorAll("#lab-host option").length, 11);
w.document.querySelector('[data-preset="small"]').click();
assert.equal(w.document.querySelectorAll("#lab-host option").length, 24);
assert.equal(w.document.querySelectorAll("#lab-neighbor-buttons button").length, 5);
w.document.querySelector('[data-model-part="graph"]').click();
assert.match($("pt-model-part-copy").textContent, /собственных векторов/);
assert.equal(w.document.querySelectorAll(".pt-guide-section").length, 6);
assert(!/noomo/i.test(w.document.querySelector(".home-footer").textContent));
async function upload(text, size = text.length) {
  Object.defineProperty($("csv-file"), "files", {configurable:true, value:[{name:"test.csv",size,text:async()=>text}]});
  $("csv-file").dispatchEvent(new w.Event("change"));
  await new Promise(setImmediate);
}
await upload("host_id,emb_0,emb_1\na,1,0\nb,1,1\nc,0,1");
assert.equal(w.document.querySelectorAll("#lab-host option").length, 3);
assert.match($("lab-status").textContent,/test.csv/);
await upload("host_id,emb_0\na,nope\nb,1");
assert.match($("lab-status").textContent,/Не удалось/);
assert.equal(w.document.querySelectorAll("#lab-host option").length, 3, "failed upload preserves dataset");
await upload("src,dst,bytes\na,b,120\nb,c,80");
assert.match($("lab-mode").textContent,/Быстрый профиль/);
await upload("",4*1024*1024);
assert.match($("lab-status").textContent,/слишком большой/);
const link = w.document.getElementById("pt-product-link");
link.click();
assert.equal(scroll.id,"pt-detail-demo");
assert.equal(w.document.querySelectorAll("iframe,dialog").length,0);
scroll = 6000;
frame();
w.document.querySelector('[data-part="2"]').click();
assert.match($("pt-part-copy").textContent, /графе/);
scroll = 8200;
frame();
w.document.querySelector('[data-metric="links"]').click();
assert.equal($("pt-metric-number").textContent, "0,974");
assert.match($("pt-metric-text").textContent, /не 97,4%/);
w.document.querySelector('[data-story="encoder"]').click();
assert.equal(scroll, 3500);
const raw = JSON.parse(
  fs.readFileSync(path.join(root, "fonts/PTDisplay_Regular.json"), "utf8"),
);
const font = {
  generateShapes(text, size) {
    const result = [],
      scale = size / raw.resolution;
    let offset = 0;
    for (const char of text) {
      const glyph = raw.glyphs[char];
      assert(glyph, "missing glyph " + char);
      const tokens = glyph.o.split(" "),
        p = new THREE.ShapePath();
      let i = 0;
      const x = () => Number(tokens[i++]) * scale + offset,
        y = () => Number(tokens[i++]) * scale;
      while (i < tokens.length) {
        const c = tokens[i++];
        if (c === "m") p.moveTo(x(), y());
        else if (c === "l") p.lineTo(x(), y());
        else if (c === "q") {
          const ex = x(),
            ey = y(),
            cx = x(),
            cy = y();
          p.quadraticCurveTo(cx, cy, ex, ey);
        } else if (c === "b") {
          const ex = x(),
            ey = y(),
            cx = x(),
            cy = y(),
            dx = x(),
            dy = y();
          p.bezierCurveTo(cx, cy, dx, dy, ex, ey);
        }
      }
      result.push(...p.toShapes());
      offset += glyph.ha * scale;
    }
    return result;
  },
};
const group = new THREE.Group();
const expected = {
  AND: "ДАННЫЕ",
  "DELIGHTD.001": "ХОСТЫ",
  INSPIREE: "МОДЕЛЬ",
  TOO: "СВЯЗИ",
  "Noomo_Logo_for_Footer.003": "ПРОФИЛЬ СЕТИ",
  Work_1_text: "ПОИСК",
  Work_2_text: "ПРОФИЛЬ",
  Work_3_text: "МЕТРИКИ",
  Work_4_text: "ЛАБОРАТОРИЯ",
};
for (const name of Object.keys(expected)) {
  const mesh = new THREE.Mesh(
    new THREE.PlaneGeometry(10000, 6000),
    new THREE.MeshBasicMaterial(),
  );
  mesh.name = name;
  mesh.position.set(3, 4, 5);
  group.add(mesh);
}
w.__ptPrepareSceneText(group, {
  FontLoader: class {
    load(url, callback) {
      assert.equal(url, "/fonts/PTDisplay_Regular.json");
      callback(font);
    }
  },
  ShapeGeometry: THREE.ShapeGeometry,
  DoubleSide: THREE.DoubleSide,
});
group.children.forEach((mesh) => {
  assert.equal(mesh.userData.ptText, expected[mesh.name]);
  assert(mesh.geometry.attributes.position.count > 0);
  const box = mesh.geometry.boundingBox;
  assert(box.max.x - box.min.x <= 11800.01);
  assert(box.max.y - box.min.y <= 7080.01);
  assert.deepEqual(mesh.position.toArray(), [3, 4, 5]);
});
assert(
  fs
    .readFileSync(path.join(root, "_nuxt/DkR7Kq9H.js"), "utf8")
    .includes("window.__ptPrepareSceneText?.(A,"),
);
console.log(
  "PASS: 8 story stages, mini-demo, metrics, architecture selection, inline content, laboratory presets, scene navigation and 9 replaced 3D texts with preserved transforms",
);
dom.window.close();

})().catch(error => {console.error(error);process.exitCode=1;});
