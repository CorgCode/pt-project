const fs = require("node:fs"), path = require("node:path"), assert = require("node:assert/strict");
const { JSDOM } = require("jsdom");
const THREE = require("../../ui-prototype/vendor/three.min.js");
const root = path.resolve(__dirname, "..");
const dom = new JSDOM(fs.readFileSync(path.join(root,"index.html"),"utf8"), {runScripts:"outside-only",url:"http://127.0.0.1:8845"}), w = dom.window;
w.eval(fs.readFileSync(path.join(root,"pt-adapter.js"),"utf8"));
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
console.log("PASS: 9 Russian 3D labels, font geometry, bounding boxes and preserved animation transforms");
dom.window.close();
