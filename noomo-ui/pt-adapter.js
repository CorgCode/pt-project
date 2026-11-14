"use strict";
function openProduct(hash = "#demo") {
  window.location.assign(/model|result|train/.test(hash)?"/model/":"/lab/");
}
window.__ptOpenProduct = openProduct;
document.addEventListener("click",event=>{
 const link=event.target.closest('a');if(!link)return;
 const raw=link.getAttribute('href')||'';
 let route=null;
 if(/^\/(lab|model)\//.test(raw))route=raw;
 else if(raw.startsWith('#pt-detail-')||raw.startsWith('/product/'))route=/model|result|train/.test(raw)?'/model/':'/lab/';
 else if(raw==='/work'||raw==='/work/'||raw.startsWith('/cases/'))route='/lab/';
 else if(raw==='/contact'||raw==='/contact/'||raw.startsWith('mailto:'))route='/model/';
 if(route){event.preventDefault();event.stopImmediatePropagation();window.location.assign(route);}
},true);
function adapt() {
  window.__ptMountContent?.();
  document.querySelectorAll(".mobile-menu .bottom a").forEach((link, i) => {
    const label = ["Код проекта", "О модели", "Лаборатория", "Результаты"][i % 4];
    if (link.textContent !== label) link.textContent = label;
    link.href = ["https://github.com/CorgCode/pt-project", "/model/", "/lab/", "/model/#results"][i % 4];
  });
  document.querySelectorAll(".status p, .home-preloader .inner-circle p").forEach(p => {
    const labels = {on:"Вкл", off:"Выкл", start:"Начать"};
    const next = labels[p.textContent.trim()];
    if (next) p.textContent = next;
  });
  document.querySelectorAll(".social-links-global-parent").forEach(panel => {
    if(panel.dataset.ptGestures)return;panel.dataset.ptGestures="ready";
    ["pointerdown","pointermove","pointerup","mousedown","mouseup","touchstart","touchend"].forEach(type=>panel.addEventListener(type,event=>event.stopPropagation()));
  });
  const footer = document.querySelector(".home-footer");
  if (footer) {
    const heading = footer.querySelector("h4");
    if (heading && heading.textContent !== "Поймите свою сеть") heading.textContent = "Поймите свою сеть";
    const copy = footer.querySelector(".copy");
    if (copy && copy.textContent !== "МФТИ × Positive Technologies · Исследовательский прототип") copy.textContent = "МФТИ × Positive Technologies · Исследовательский прототип";
  }

  document
    .querySelectorAll(".right-menu a,.mobile-menu .top a")
    .forEach((link) => {
      const span = link.querySelector("span");
      if (!span) return;
      const text = span.textContent.trim();
      const names = { Work: "Лаборатория", Contact: "Модель", Home: "Продукт" };
      const href=link.getAttribute("href");if(href==="/work")link.setAttribute("href","/lab/");if(href==="/contact")link.setAttribute("href","/model/");
      if (names[text]) span.textContent = names[text];
    });
  document.querySelectorAll(".home-footer .socials a").forEach((link, i) => {
    const label = ["GitHub", "Модель", "Демонстрация"][i % 3];
    if (link.textContent !== label) link.textContent = label;
    link.href = [
      "https://github.com/CorgCode/pt-project",
      "/model/",
      "/lab/",
    ][i % 3];
  });
  const send = document.querySelector(".home-footer .send");
  if (send) {
    send.href = "/lab/";
    const p = send.querySelector(".inner-circle p");
    if (p && p.textContent.trim() !== "Открыть продукт")
      p.textContent = "Открыть продукт";
  }
  document.getElementById("pt-product-link")?.remove();
  const soundCopy = document.querySelector(".home-preloader .bottom > p");
  if (soundCopy && soundCopy.textContent.trim() !== "Для этой сцены используется звук.") soundCopy.textContent = "Для этой сцены используется звук.";
  const prompt = document.querySelectorAll(".click-and-hold .text span");
  ["Что скрывает ваша сеть?", "Нажмите и удерживайте", "Нажмите и удерживайте"].forEach((text, i) => {
    if (prompt[i] && prompt[i].textContent !== text) prompt[i].textContent = text;
  });
  const cursor = document.querySelector("#cursor .text p");
  if (cursor && /click\s*and\s*hold/i.test(cursor.textContent)) cursor.innerHTML = "Нажмите<br>и удерживайте";

}
if (document.readyState === "loading")
  document.addEventListener("DOMContentLoaded", adapt);
else adapt();
let adaptationScheduled = false;
new MutationObserver(() => {
  if (adaptationScheduled) return;
  adaptationScheduled = true;
  setTimeout(() => {
    adaptationScheduled = false;
    adapt();
  }, 200);
}).observe(document.documentElement, { childList: true, subtree: true });

// Replace the words baked into the animated GLB without changing animation targets.
window.__ptPrepareSceneText = function (scene, types) {
  const labels = {
    AND: "ДАННЫЕ",
    "DELIGHTD.001": "СЕТЬ",
    INSPIREE: "МОДЕЛЬ",
    TOO: "СВЯЗИ",
    "Noomo_Logo_for_Footer.003": "ПРОФИЛЬ СЕТИ",
    Work_1_text: "ПОИСК",
    Work_2_text: "ПРОФИЛЬ",
    Work_3_text: "МЕТРИКИ",
    Work_4_text: "ЛАБОРАТОРИЯ",
  };
  // GLTFLoader removes punctuation from node names (DELIGHTD.001 -> DELIGHTD001).
  const normalizedLabels = Object.fromEntries(Object.entries(labels).map(([name,text])=>[name.replace(/[.\[\]:/]/g,""),text]));
  scene.traverse(mesh=>{
    if(mesh.isMesh&&(labels[mesh.name]||normalizedLabels[mesh.name.replace(/[.\[\]:/]/g, "")])){
      mesh.userData.ptTextWasVisible=mesh.visible;mesh.visible=false;
    }
  });
  const loader = new types.FontLoader();
  loader.load(
    "/fonts/PTDisplay_Regular.json",
    (font) => {
      scene.traverse((mesh) => {
        const text = labels[mesh.name] || normalizedLabels[mesh.name.replace(/[.\[\]:/]/g, "")];
        if (!text || !mesh.isMesh || !mesh.geometry) return;
        const original = mesh.geometry;
        original.computeBoundingBox();
        const old = original.boundingBox;
        if (!old) return;
        const geometry = new types.ShapeGeometry(
          font.generateShapes(text, 1000),
        );
        geometry.computeBoundingBox();
        const box = geometry.boundingBox;
        const oldW = old.max.x - old.min.x,
          oldH = old.max.y - old.min.y,
          newW = box.max.x - box.min.x,
          newH = box.max.y - box.min.y;
        if (!(oldW > 0 && oldH > 0 && newW > 0 && newH > 0)) {
          geometry.dispose();
          return;
        }
        const scale = Math.min(oldW / newW, oldH / newH) * 1.18;
        const x =
            (old.min.x + old.max.x) / 2 - ((box.min.x + box.max.x) / 2) * scale,
          y =
            (old.min.y + old.max.y) / 2 - ((box.min.y + box.max.y) / 2) * scale,
          z = (old.min.z + old.max.z) / 2;
        geometry.scale(scale, scale, scale);
        geometry.translate(x, y, z);
        geometry.computeBoundingBox();
        geometry.computeBoundingSphere();
        mesh.geometry = geometry;
        mesh.userData.ptText = text;
        mesh.visible=mesh.userData.ptTextWasVisible??true;
        delete mesh.userData.ptTextWasVisible;
        const materials = Array.isArray(mesh.material)
          ? mesh.material
          : [mesh.material];
        materials.forEach((material) => {
          if (material) {
            material.side = types.DoubleSide;
            material.needsUpdate = true;
          }
        });
      });
    },
    undefined,
    (error) => console.error("PT 3D text could not load", error),
  );
};

// A transmission pass must not sample the texture currently attached for writing.
window.__ptRenderTargetScene = function (world, target) {
  const hidden=[];
  world.scene.traverse(mesh=>{
    if(!mesh.isMesh||!mesh.visible)return;
    const materials=Array.isArray(mesh.material)?mesh.material:[mesh.material];
    if(materials.some(material=>material?.buffer===target.texture)){
      hidden.push(mesh);mesh.visible=false;
    }
  });
  try{world.gl.render(world.scene,world.camera);}
  finally{hidden.forEach(mesh=>mesh.visible=true);}
};

// Resizing an open DevTools panel can interrupt the original loader timeline.
// Reveal its real start control only after the scene loader reports completion.
window.__ptSceneReady = function () {
  setTimeout(()=>{
    document.body.classList.add('pt-scene-ready');
    const start=document.querySelector('.home-preloader .done .send');
    if(start){start.setAttribute('role','button');start.setAttribute('aria-label','Начать исследование сети');start.tabIndex=0;start.addEventListener('keydown',event=>{if(event.key==='Enter'||event.key===' '){event.preventDefault();start.click();}});}
  },2800);
};
