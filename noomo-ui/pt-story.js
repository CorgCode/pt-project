/* Hallmark · macrostructure: spatial-product-tour · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: structural checks pass; browser review pending
 */
import { h as gsap, __tla as libraryReady } from "/_nuxt/iNjJ8EkR.js";
await libraryReady;
let root, pages = [], current = -1, scrollLength = 12000, layoutTimer, hashApplied = false;
const start = 850, stride = 850;
const reduce = () => matchMedia("(prefers-reduced-motion: reduce)").matches;
const clean = el => {
  const clone = el.cloneNode(true);
  clone.querySelectorAll("br").forEach(br => br.replaceWith(" "));
  return clone.textContent.trim().replace(/\s+/g, " ");
};
function splitText(text, limit) {
  const chunks = []; let part = "";
  text.split(/(?<=[.!?])\s+/).forEach(sentence => {
    if (part && (part + sentence).length > limit) {chunks.push(part); part = "";}
    // Long sentences are split at word boundaries, never cut off.
    for (const word of sentence.split(" ")) {
      if (part && (part + word).length > limit) {chunks.push(part.trim()); part = "";}
      part += word + " ";
    }
  });
  if (part.trim()) chunks.push(part.trim());
  return chunks;
}
function buildPages() {
  const source = document.getElementById("pt-guide");
  if (!source) return;
  const limit = innerWidth < 760 ? Math.min(360,Math.max(100,Math.floor((innerHeight*.50-180)/24)*30)) : 600;
  const result = [];
  source.querySelectorAll(".pt-guide-section").forEach(section => {
    const id = section.id.replace("pt-detail-", "");
    let title = clean(section.querySelector("h2")), body = [], bodyLength = 0;
    const flush = () => {
      if (!body.length) return;
      result.push({id, title, paragraphs:body, kind:kindFor(id,title)});
      body = []; bodyLength = 0;
    };
    section.querySelectorAll("h3,h4,p,figcaption,small").forEach(el => {
      if (el.closest(".pt-lab-controls,.pt-lab-workspace,.pt-lab-guide")) return;
      if (/^H[34]$/.test(el.tagName)) {flush(); title = clean(el); return;}
      let text = clean(el);
      if (!text || el.id === "pt-model-part-copy") return;
      if (el.parentElement.classList.contains("pt-metric-ledger")) return;
      const article = el.closest(".pt-metric-ledger article");
      if (article && el.tagName === "P") text = clean(article.querySelector("strong")) + " — " + text;
      splitText(text,limit).forEach(chunk => {
        if (bodyLength + chunk.length > limit) flush();
        body.push(chunk); bodyLength += chunk.length;
      });
    });
    flush();
  });
  // The laboratory has a dedicated interactive stop, not just explanatory text.
  const at = result.findIndex(p => p.id === "demo");
  if (at >= 0) result[at].interactive = true;
  pages = result;
  scrollLength = start + pages.length * stride;
  const runway = document.querySelector(".home-page");
  if (runway) runway.style.setProperty("height", `${scrollLength + innerHeight}px`, "important");
  window.__ptPages = pages;
  window.__ptRefreshLayout();
}
function kindFor(id,title) {
  if (id === "model") {
    if (/признак/.test(title)) return "input";
    if (/окна|архитектур/.test(title)) return "history";
    if (/три части|портрет|структур|собеседник/.test(title)) return "architecture";
    return "encoder";
  }
  return id;
}
function scrollToPage(index) {
  if (!pages.length) return;
  const i = Math.max(0,Math.min(pages.length-1,index));
  const y = start + i*stride + stride*.4;
  const smoother = gsap.core.globals().ScrollSmoother?.get();
  if (smoother) smoother.scrollTo(y, !reduce());
  else window.scrollTo({top:y,behavior:reduce()?"instant":"smooth"});
}
window.__ptGoChapter = id => {
  mount();
  const key = id === "encoder" ? "model" : id;
  const index = pages.findIndex(page => page.id === key);
  if (index >= 0) scrollToPage(index);
};
window.__ptRefreshLayout = () => {
  clearTimeout(layoutTimer);
  layoutTimer = setTimeout(() => gsap.core.globals().ScrollTrigger?.refresh(),100);
};
function button(text, action) {
  const b=document.createElement("button");b.type="button";b.textContent=text;b.addEventListener("click",action);return b;
}
function mount() {
  window.__ptMountContent?.();
  if (!pages.length && document.getElementById("pt-guide")) buildPages();
  if (root || !document.body) return;
  document.documentElement.lang="ru";
  document.body.classList.add("pt-world-tour");
  root=document.createElement("div");root.className="pt-world-ui";root.id="pt-world-ui";root.hidden=true;
  const nav=document.createElement("nav");nav.className="pt-world-nav";nav.setAttribute("aria-label","Разделы 3D-рассказа");
  [["problem","Задача"],["model","Модель"],["results","Результаты"],["demo","Попробовать"]].forEach(([id,label])=>{
    const b=button(label,()=>window.__ptGoChapter(id));b.dataset.chapter=id;nav.append(b);
  });
  const controls=document.createElement("div");controls.className="pt-world-controls";
  const prev=button("← Назад",()=>scrollToPage(current-1));prev.id="pt-world-prev";
  const count=document.createElement("span");count.id="pt-world-count";
  const next=button("Дальше →",()=>scrollToPage(current+1));next.id="pt-world-next";
  controls.append(prev,count,next);
  const lab=document.createElement("div");lab.className="pt-world-lab";lab.id="pt-world-lab";lab.hidden=true;
  [["project","300 хостов"],["small","24 хоста"],["flows","Пример трафика"]].forEach(([id,label])=>{
    const b=button(label,()=>document.querySelector(`[data-preset="${id}"]`)?.click());b.dataset.worldPreset=id;lab.append(b);
  });
  lab.append(button("↑ Загрузить CSV",()=>document.getElementById("csv-file")?.click()));
  lab.append(button("Пример CSV ↓",()=>document.getElementById("download-sample")?.click()));
  const hostLabel=document.createElement("label");hostLabel.textContent="Хост ";
  const host=document.createElement("select");host.id="pt-world-host";host.setAttribute("aria-label","Выбрать хост в 3D");
  host.addEventListener("change",()=>window.__ptSelectHost?.(host.value));hostLabel.append(host);lab.append(hostLabel);
  const status=document.createElement("span");status.id="pt-world-status";status.setAttribute("role","status");lab.append(status);
  const accessible=document.createElement("section");accessible.id="pt-world-readable";accessible.className="pt-world-readable";accessible.hidden=true;
  accessible.setAttribute("aria-label","Текущий текст сцены");
  const text=button("Текст",()=>{accessible.hidden=!accessible.hidden;});text.className="pt-world-text-toggle";
  const help=document.createElement("p");help.className="pt-world-hint";help.textContent="Прокрутка — дальше · объекты справа можно вращать";
  const reset=button("Вид ↺",()=>window.__ptResetView?.());reset.className="pt-world-view-reset";
  root.append(nav,controls,lab,accessible,text,help,reset);
  ["click","pointerdown","pointermove","pointerup"].forEach(type=>root.addEventListener(type,e=>e.stopPropagation()));
  document.body.append(root);
  if (window.__ptLabState) window.dispatchEvent(new CustomEvent("pt:lab-update",{detail:window.__ptLabState}));
}
function show(index,p) {
  const page=pages[index];
  if (current!==index) {
    current=index;
    document.getElementById("pt-world-count").textContent=`${index+1} / ${pages.length}`;
    document.getElementById("pt-world-prev").disabled=index===0;
    document.getElementById("pt-world-next").disabled=index===pages.length-1;
    document.getElementById("pt-world-lab").hidden=page.id!=="demo";
    root.querySelectorAll("[data-chapter]").forEach(b=>b.setAttribute("aria-current",String(b.dataset.chapter===page.id)));
    const readable=document.getElementById("pt-world-readable");readable.replaceChildren();
    const title=document.createElement("h2");title.textContent=page.title;readable.append(title);
    page.paragraphs.forEach(text=>{const para=document.createElement("p");para.textContent=text;readable.append(para);});
  }
  window.__ptWorldState={page,index,total:pages.length,progress:p,visible:true};
}
function frame() {
  requestAnimationFrame(frame);
  if(document.hidden)return;
  mount();
  const scene=document.getElementById("home-scene");
  const ready=scene&&Number(getComputedStyle(scene).opacity)>.05&&pages.length;
  const smoother=gsap.core.globals().ScrollSmoother?.get();
  const y=smoother?.scrollTop?.()??window.scrollY;
  if (ready && !hashApplied) {
    hashApplied=true;
    if(location.hash.startsWith("#pt-detail-"))window.__ptGoChapter(location.hash.slice(11));
  }
  const active=!!ready&&y>=start&&y<=scrollLength+1;
  if(root)root.hidden=!active;
  document.body.classList.toggle("pt-world-active",active);
  if(active){const raw=(y-start)/stride,index=Math.min(pages.length-1,Math.floor(raw));show(index,Math.min(1,raw-index));}
  else window.__ptWorldState={visible:false};
}
window.addEventListener("pt:inspect",event=>{
  const readable=document.getElementById("pt-world-readable");if(!readable)return;
  readable.replaceChildren();
  const h=document.createElement("h2");h.textContent=event.detail.title;readable.append(h);
  event.detail.paragraphs.forEach(t=>{const p=document.createElement("p");p.textContent=t;readable.append(p);});
});
window.addEventListener("pt:lab-update",event=>{
  const state=event.detail,select=document.getElementById("pt-world-host");
  if(select){
    if(select.options.length!==state.dataset.length||select.options[0]?.value!==state.dataset[0].id){
      select.replaceChildren(...state.dataset.map(h=>{const o=document.createElement("option");o.value=h.id;o.textContent=h.id;return o;}));
    }
    select.value=state.selected.id;
  }
  document.querySelectorAll("[data-world-preset]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.worldPreset===(state.kind==="flows"?"flows":state.dataset.length===24?"small":state.kind==="project"?"project":"uploaded"))));
});
window.addEventListener("resize",()=>{
  const section=pages[current]?.id;
  clearTimeout(window.__ptResizeTimer);
  window.__ptResizeTimer=setTimeout(()=>{buildPages();current=-1;if(section)window.__ptGoChapter(section);window.__ptWorldResize?.();},180);
});
document.addEventListener("keydown",e=>{
  if(root?.hidden||/INPUT|SELECT|TEXTAREA|BUTTON/.test(e.target.tagName))return;
  if(e.key==="ArrowRight"){e.preventDefault();scrollToPage(current+1);}
  if(e.key==="ArrowLeft"){e.preventDefault();scrollToPage(current-1);}
});
// The hidden CSV status is mirrored for upload validation, without a second interface.
window.__ptStatusObserver = new MutationObserver(()=>{
  const status=document.getElementById("pt-world-status"),source=document.getElementById("lab-status");
  if(status&&source&&status.textContent!==source.textContent)status.textContent=source.textContent;
});
window.__ptStatusObserver.observe(document.documentElement,{childList:true,subtree:true});
mount();requestAnimationFrame(frame);
