const fs = require("fs"),
  assert = require("node:assert/strict"),
  { JSDOM } = require("jsdom");
const root = require("node:path").resolve(__dirname, "..") + "/";
const THREE = require(root + "vendor/three.min.js");
const html = fs
  .readFileSync(root + "page.html", "utf8")
  .replace("__CSS__", fs.readFileSync(root + "style.css", "utf8"))
  .replace("__FONTS__", "")
  .replace(/__(?:THREE|RGBE|CHART|HDR|DATA|APP|LAB)__/g, "");
const dom = new JSDOM(html, {
    runScripts: "outside-only",
    pretendToBeVisual: true,
  }),
  w = dom.window;
w.THREE = { ...THREE };
let renderedScene = null;
class Renderer {
  constructor({ canvas }) {
    this.canvas = canvas;
    this.ratio = 1;
  }
  setPixelRatio(n) {
    this.ratio = n;
  }
  getPixelRatio() {
    return this.ratio;
  }
  setClearColor() {}
  setSize(x, y) {
    this.canvas.width = Math.round(x * this.ratio);
    this.canvas.height = Math.round(y * this.ratio);
  }
  render(scene, camera) {
    renderedScene = scene;
    scene.updateMatrixWorld();
    assert(Number.isFinite(camera.aspect));
  }
}
w.THREE.WebGLRenderer = Renderer;
w.THREE.PMREMGenerator = class {
  fromEquirectangular() {
    return { texture: new THREE.Texture(), dispose() {} };
  }
  dispose() {}
};
w.HTMLCanvasElement.prototype.getContext = function () {
  return {
    createLinearGradient() {
      return { addColorStop() {} };
    },
    fillRect() {},
  };
};
w.matchMedia = () => ({ matches: false });
w.IntersectionObserver = class {
  observe() {}
  unobserve() {}
};
const frames = [];
w.requestAnimationFrame = (fn) => {
  frames.push(fn);
  return frames.length;
};
w.eval(fs.readFileSync(root + "vendor/rgbe.js", "utf8"));
const hdr = fs.readFileSync(root + "vendor/studio.hdr");
const result = new w.RGBELoader().parse(
  hdr.buffer.slice(hdr.byteOffset, hdr.byteOffset + hdr.byteLength),
);
assert.equal(result.width, 1024);
assert.equal(result.height, 512);
w.STUDIO_HDR = hdr.toString("base64");
w.eval(fs.readFileSync(root + "data.js", "utf8"));
w.eval(
  fs.readFileSync(root + "app.js", "utf8") +
    "\nwindow.__setScene=(value)=>{targetScene=value;sceneProgress=value};",
);
assert(frames.length > 0);
frames.shift()(1000);
assert(renderedScene);
for (const stage of [0, 1, 2, 4, 5, 6, 7.15, 7.4, 7.7, 8, 9, 10]) {
  w.__setScene(stage);
  frames.shift()(1000 + stage * 100);
  renderedScene.traverse((node) =>
    assert(node.matrixWorld.elements.every(Number.isFinite)),
  );
  if (stage === 7.4) {
    assert(renderedScene.getObjectByName("ProfileCutaway").visible);
    assert(!renderedScene.getObjectByName("HostProfileShell").visible);
  }
}

let meshes = 0,
  vertices = 0;
renderedScene.traverse((node) => {
  if (node.isMesh) {
    meshes++;
    const p = node.geometry.attributes.position;
    if (p) {
      vertices += p.count;
      assert([...p.array].every(Number.isFinite), "invalid geometry");
    }
    assert(
      node.matrixWorld.elements.every(Number.isFinite),
      "invalid transform",
    );
  }
});
assert(meshes > 100);
assert(vertices > 20000);
console.log(
  "PASS: HDR 1024×512 parsed, scene initialized, " +
    meshes +
    " meshes / " +
    vertices +
    " vertices, finite geometry and animated transforms (render stub; not visual GPU test)",
);
dom.window.close();
