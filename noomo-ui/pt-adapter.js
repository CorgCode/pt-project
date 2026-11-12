"use strict";
let productDialog = null,
  previousFocus = null,
  pausedMedia = [];
const sections = {
  "#problem": "Задача",
  "#product": "Возможности",
  "#demo": "Лаборатория",
  "#model": "Как устроена модель",
  "#results": "Результаты",
};
function closeProduct() {
  if (productDialog?.open) productDialog.close();
}
function openProduct(hash = "#demo") {
  if (!sections[hash]) hash = "#demo";
  if (!productDialog) {
    productDialog = document.createElement("dialog");
    productDialog.id = "pt-product-dialog";
    productDialog.className = "pt-product-dialog";
    productDialog.setAttribute("aria-label", "Продукт PT Network");
    const bar = document.createElement("div");
    bar.className = "pt-product-bar";
    const title = document.createElement("strong");
    title.textContent = "PT NETWORK";
    bar.append(title);
    const nav = document.createElement("nav");
    nav.setAttribute("aria-label", "Разделы продукта");
    Object.entries(sections).forEach(([anchor, label]) => {
      const b = document.createElement("button");
      b.type = "button";
      b.dataset.section = anchor;
      b.textContent = label;
      b.addEventListener("click", () => openProduct(anchor));
      nav.append(b);
    });
    bar.append(nav);
    const close = document.createElement("button");
    close.type = "button";
    close.className = "pt-close";
    close.textContent = "Закрыть ×";
    close.setAttribute("aria-label", "Закрыть продукт и вернуться к сцене");
    close.addEventListener("click", closeProduct);
    bar.append(close);
    const frame = document.createElement("iframe");
    frame.id = "pt-product-frame";
    frame.title = "PT Network — модель и интерактивная лаборатория";
    productDialog.append(bar, frame);
    document.body.append(productDialog);
    productDialog.addEventListener("close", () => {
      frame.removeAttribute("src");
      document.documentElement.style.overflow = "";
      pausedMedia.forEach((media) => media.play().catch(() => {}));
      pausedMedia = [];
      previousFocus?.focus?.();
    });
  }
  const frame = productDialog.querySelector("iframe");
  const url = "/product/?embedded=1" + hash;
  if (!productDialog.open) {
    previousFocus = document.activeElement;
    pausedMedia = [...document.querySelectorAll("audio,video")].filter(
      (media) => !media.paused,
    );
    pausedMedia.forEach((media) => media.pause());
    document.documentElement.style.overflow = "hidden";
    productDialog.showModal();
  }
  if (frame.getAttribute("src") !== url) frame.setAttribute("src", url);
  productDialog
    .querySelectorAll("[data-section]")
    .forEach((button) =>
      button.setAttribute(
        "aria-pressed",
        String(button.dataset.section === hash),
      ),
    );
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
    if (raw.startsWith("/product/"))
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
      "/product/#model",
      "/product/#demo",
    ][i % 3];
  });
  const send = document.querySelector(".home-footer .send");
  if (send) {
    send.href = "/product/#demo";
    const p = send.querySelector(".inner-circle p");
    if (p && p.textContent.trim() !== "Открыть продукт")
      p.textContent = "Открыть продукт";
  }
  if (!document.getElementById("pt-product-link")) {
    const link = document.createElement("a");
    link.id = "pt-product-link";
    link.className = "pt-product-link";
    link.href = "/product/#demo";
    link.textContent = "Открыть лабораторию";
    const arrow = document.createElement("span");
    arrow.textContent = "↗";
    link.append(arrow);
    document.body.append(link);
  }
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
    "Noomo_Logo_for_Footer.003": "PT NETWORK",
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
        const scale = Math.min(oldW / newW, oldH / newH);
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
