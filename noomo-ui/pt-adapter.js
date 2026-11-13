"use strict";
function openProduct(hash = "#demo") {
  window.__ptMountContent?.();
  window.__ptGoChapter?.(hash.replace("#", ""));
}
window.__ptOpenProduct = openProduct;
document.addEventListener(
  "click",
  (event) => {
    const link = event.target.closest("a");
    if (!link) return;
    const raw = link.getAttribute("href") || "";
    let section = null,
      external = null;
    if (raw.startsWith("#pt-detail-")) section = "#" + raw.slice(11);
    else if (raw.startsWith("/product/"))
      section = raw.includes("#") ? raw.slice(raw.indexOf("#")) : "#demo";
    else if (raw === "/work" || raw === "/work/" || raw.startsWith("/cases/"))
      section = "#demo";
    else if (
      raw === "/contact" ||
      raw === "/contact/" ||
      raw.includes("noomoagency.com") ||
      raw.startsWith("mailto:")
    )
      section = "#model";
    else if (/linkedin\.com|instagram\.com|twitter\.com/.test(raw))
      external = "https://github.com/CorgCode/pt-project";
    if (section || external) {
      event.preventDefault();
      event.stopImmediatePropagation();
      if (section) openProduct(section);
      else window.location.assign(external);
    }
  },
  true,
);
function adapt() {
  window.__ptMountContent?.();
  document.querySelectorAll(".mobile-menu .bottom a").forEach((link, i) => {
    const label = ["Код проекта", "О модели", "Демонстрация", "Результаты"][i % 4];
    if (link.textContent !== label) link.textContent = label;
    link.href = ["https://github.com/CorgCode/pt-project", "#pt-detail-model", "#pt-detail-demo", "#pt-detail-results"][i % 4];
  });
  document.querySelectorAll(".status p, .home-preloader .inner-circle p").forEach(p => {
    const labels = {on:"Вкл", off:"Выкл", start:"Начать"};
    const next = labels[p.textContent.trim()];
    if (next) p.textContent = next;
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
      const names = { Work: "Продукт", Contact: "Модель", Home: "Сцена" };
      if (names[text]) span.textContent = names[text];
    });
  document.querySelectorAll(".home-footer .socials a").forEach((link, i) => {
    const label = ["GitHub", "Модель", "Демонстрация"][i % 3];
    if (link.textContent !== label) link.textContent = label;
    link.href = [
      "https://github.com/CorgCode/pt-project",
      "#pt-detail-model",
      "#pt-detail-demo",
    ][i % 3];
  });
  const send = document.querySelector(".home-footer .send");
  if (send) {
    send.href = "#pt-detail-demo";
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
    "DELIGHTD.001": "ХОСТЫ",
    INSPIREE: "МОДЕЛЬ",
    TOO: "СВЯЗИ",
    "Noomo_Logo_for_Footer.003": "ПРОФИЛЬ СЕТИ",
    Work_1_text: "ПОИСК",
    Work_2_text: "ПРОФИЛЬ",
    Work_3_text: "МЕТРИКИ",
    Work_4_text: "ЛАБОРАТОРИЯ",
  };
  const loader = new types.FontLoader();
  loader.load(
    "/fonts/PTDisplay_Regular.json",
    (font) => {
      scene.traverse((mesh) => {
        const text = labels[mesh.name];
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
