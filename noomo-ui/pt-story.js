/* Hallmark · macrostructure: spatial-product-tour · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: structural checks pass; browser review pending
 */
import { h as gsap, __tla as libraryReady } from "/_nuxt/iNjJ8EkR.js";
await libraryReady;
let mode="main";
const mainPages = [
  {id:"problem",kind:"device",title:"Зачем изучать сеть",paragraphs:["В сети компании работают компьютеры, серверы и служебные системы. Они постоянно обмениваются данными.","Чтобы разобраться в работе сети, важно понимать, какие устройства ведут себя похоже и что их объединяет."]},
  {id:"problem",kind:"traffic",title:"Адреса недостаточно",paragraphs:["По адресу компьютера не видно, чем он занят и какое место занимает в сети.","Взаимодействия дают больше контекста: с кем он общается, какие сервисы использует и когда активен. Из этого складывается картина его поведения."]},
  {id:"model",kind:"features",title:"Что мы предлагаем?",paragraphs:["Наша ML-модель сама находит сходства в сетевом поведении устройств.","Вы выбираете компьютер — сервис показывает, на какие устройства он похож. Для этого не нужно заранее знать их назначение."]},
  {id:"product",kind:"search",title:"Чем вам будет полезен наш продукт?",paragraphs:["Вместо поиска по всей сети вы получаете устройства с похожим поведением. Можно изучать их вместе и сравнивать окружение."],note:"Здесь — пример на учебных данных."},
  {id:"product",kind:"device",title:"Видеть сеть целиком",paragraphs:["Наша цель — показывать группы похожих устройств и замечать изменения в их поведении."],note:"Сейчас в прототипе работает поиск похожих устройств."},
  {id:"demo",kind:"device",title:"От устройства — к окружению",paragraphs:["Сначала изучаете одно устройство, затем — похожие на него. Так можно разбирать сеть по частям, а не просматривать все адреса подряд."]}
];
const guidedPages = [
  {id:"problem",kind:"device",title:"Это хост. Обычное устройство.",paragraphs:["Хост — компьютер или сервер в сети. Адрес мы видим. А вот его назначение и привычки по одному адресу не понять."]},
  {id:"problem",kind:"traffic",title:"Посмотрим, с кем он общается.",paragraphs:["Пакеты идут от устройства к сервисам и обратно. Из этих взаимодействий складывается история его поведения."],note:"Условный пример трафика, не реальные записи выбранного хоста."},
  {id:"model",kind:"features",title:"Собираем сетевые привычки.",paragraphs:["С кем общается? Какие сервисы использует? Сколько передаёт? Когда активен? Модель получает признаки этих взаимодействий."]},
  {id:"model",kind:"encoder",title:"Привычки превращаются в числовой профиль.",paragraphs:["Кодировщик — нейросеть, которая превращает признаки трафика в 896 чисел. По таким профилям можно сравнить устройства."],note:"Наглядная схема, не точное число слоёв сети."},
  {id:"model",kind:"architecture",title:"Добавляем историю и окружение.",paragraphs:["896 чисел описывают профиль хоста. Ещё 64 помогают предсказывать новые контакты, и 64 — учитывать структуру сети. Нажмите на часть модели."]},
  {id:"training",kind:"training",title:"Проверяем на будущих днях.",paragraphs:["Учимся на днях 02–12, выбираем настройки на 13-м, проверяем на 14–16-м. Будущие тестовые записи не используются при обучении."]},
  {id:"demo",kind:"search",title:"А теперь — найдём похожего.",paragraphs:["Выбрали один хост. Сравним его профиль со всеми остальными. Нажмите «Найти похожего» — расчёт выполняется здесь, в браузере."],note:"Тестовые векторы проекта. Обученная многодневная модель здесь не запускается."},
  {id:"results",kind:"results",title:"Что получилось в исследовании.",paragraphs:["Общий балл вырос с 56,5 до 70,0 из 100. Это сравнение нескольких задач на тестовых окнах, а не процент обнаруженных атак."],note:"Таблица от 4 ноября 2026. Метрики и ограничения — в режиме «Подробно»."},
  {id:"product",kind:"device",title:"Меньше ручного поиска. Больше контекста.",paragraphs:["Сейчас можно искать похожие профили. Дальше — объединять хосты в группы, объяснять соседство и замечать необычные изменения поведения."]},
  {id:"demo",kind:"graph-final",title:"Ваша сеть. Исследуйте её.",paragraphs:["Выберите узел — увидите ближайшие профили. Тяните узел, чтобы раздвинуть граф. Тяните свободное место, чтобы повернуть всю сцену."],note:"Линии показывают близость профилей, а не реальные сетевые соединения."}
];
let root, pages = [], current = -1, scrollLength = 12000, layoutTimer, hashApplied = false;
const start = 850;
const pageStride = () => mode==="main" ? 2000 : 850;
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
  pages = mode==="main" ? mainPages.map(p=>({...p,guided:true,simple:true})) : mode==="guided" ? guidedPages.map(p=>({...p,guided:true})) : result;
  document.body.classList.toggle("pt-guided-mode",mode!=="details");
  scrollLength = Math.max(11200, start + pages.length * pageStride());
  const runway = document.querySelector(".home-page");
  if (runway) runway.style.setProperty("height", `${scrollLength + innerHeight}px`, "important");
  window.__ptPages = pages;
  window.__ptStoryLayout = {start,stride:pageStride(),scrollLength};
  window.__ptRefreshLayout();
}
window.__ptSetTourMode = value => {
  const section=pages[current]?.id;
  mode=["main","details","guided"].includes(value)?value:"main";current=-1;buildPages();
  document.querySelectorAll("[data-tour-mode]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.tourMode===mode)));
  window.__ptWorldResize?.();
  if(section)window.__ptGoChapter(section);
};
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
  const y = start + i*pageStride() + pageStride()*.4;
  const smoother = gsap.core.globals().ScrollSmoother?.get();
  if (smoother) smoother.scrollTo(y, !reduce());
  else window.scrollTo({top:y,behavior:reduce()?"instant":"smooth"});
}
window.__ptGoGraph = () => {const index=pages.findIndex(p=>p.kind==="graph-final");if(index>=0)scrollToPage(index);};
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
  const modes=document.createElement("div");modes.className="pt-tour-modes";
  [["guided","Понятный обзор"],["details","Подробно"]].forEach(([id,label])=>{const b=button(label,()=>window.__ptSetTourMode(id));b.dataset.tourMode=id;b.setAttribute("aria-pressed",String(id===mode));modes.append(b);});
  const action=button("Найти похожего",()=>window.__ptFindSimilar?.());action.id="pt-find-similar";action.hidden=true;
  const dataButton=button("Данные и загрузка ▾",()=>lab.classList.toggle("pt-data-open"));dataButton.id="pt-data-toggle";lab.prepend(dataButton);
  const zoom=document.createElement("div");zoom.className="pt-world-zoom";
  zoom.append(button("−",()=>window.__ptZoomView?.(.85)),button("+",()=>window.__ptZoomView?.(1.15)));
  const links=document.createElement("nav");links.className="pt-page-links";links.setAttribute("aria-label","Продолжить исследование");links.hidden=true;
  for(const [href,label] of [["/model/","Как устроена модель →"],["/lab/","Лаборатория →"]]){const a=document.createElement("a");a.href=href;a.textContent=label;links.append(a);}
  const side=document.createElement("section");side.className="pt-main-copy";side.setAttribute("aria-label","О проекте");side.hidden=true;
  root.append(nav,controls,lab,accessible,text,help,reset,modes,action,zoom,links,side);
  ["click","pointerdown","pointermove","pointerup"].forEach(type=>root.addEventListener(type,e=>e.stopPropagation()));
  document.body.append(root);
  if (window.__ptLabState) window.dispatchEvent(new CustomEvent("pt:lab-update",{detail:window.__ptLabState}));
}
let copyAnimations=[];
function renderSideCopy(page) {
  const side=root.querySelector(".pt-main-copy");
  side.hidden=!page.simple;if(!page.simple)return;
  copyAnimations.forEach(a=>{try{a.commitStyles?.();}catch{}a.cancel();});copyAnimations=[];
  const previous=side.querySelector(".pt-copy-current");
  side.querySelector(".pt-copy-outgoing")?.remove();
  if(previous){
    previous.className="pt-copy-outgoing";previous.setAttribute("aria-hidden","true");
    if(!reduce()&&previous.animate){
      const a=previous.animate([{opacity:1,transform:"translate3d(0,0,0)"},{opacity:0,transform:"translate3d(0,-24px,0)"}],{duration:700,easing:"cubic-bezier(.3,0,.3,1)",fill:"forwards"});a.onfinish=()=>previous.remove();copyAnimations.push(a);
    }else previous.remove();
  }
  const content=document.createElement("div");content.className="pt-copy-current";
  const heading=document.createElement("h2");heading.textContent=page.title;content.append(heading);
  page.paragraphs.forEach(text=>{const paragraph=document.createElement("p");paragraph.textContent=text;content.append(paragraph);});
  if(page.note){const note=document.createElement("small");note.textContent=page.note;content.append(note);}
  side.append(content);
  if(!reduce())Array.from(content.children).forEach((element,i)=>{
    if(!element.animate)return;
    const a=element.animate([{opacity:0,transform:"translate3d(0,32px,0)"},{opacity:1,transform:"translate3d(0,0,0)"}],{duration:1600,delay:180+i*140,easing:"cubic-bezier(.16,1,.3,1)",fill:"both"});copyAnimations.push(a);
  });
}
function show(index,p) {
  const page=pages[index];
  if (current!==index) {
    current=index;
    renderSideCopy(page);
    root.querySelector(".pt-page-links").hidden=true;
    document.getElementById("pt-world-count").textContent=`${index+1} / ${pages.length}`;
    document.getElementById("pt-world-prev").disabled=index===0;
    document.getElementById("pt-world-next").disabled=index===pages.length-1;
    document.getElementById("pt-world-lab").hidden=page.id!=="demo";
    document.getElementById("pt-find-similar").hidden=page.kind!=="search";
    if(page.kind==="search"){window.__ptSearchFound=false;document.getElementById("pt-find-similar").textContent="Найти похожего";}
    document.body.classList.toggle("pt-final-graph",page.kind==="graph-final");
    document.body.classList.toggle("pt-original-scene",!!page.simple&&page.kind!=="graph-final");
    root.querySelectorAll("[data-chapter]").forEach(b=>b.setAttribute("aria-current",String(b.dataset.chapter===page.id)));
    const readable=document.getElementById("pt-world-readable");readable.replaceChildren();
    const title=document.createElement("h2");title.textContent=page.title;readable.append(title);
    page.paragraphs.forEach(text=>{const para=document.createElement("p");para.textContent=text;readable.append(para);});
  }
  window.__ptWorldState={page,index,total:pages.length,progress:p,visible:true};
  if(mode==="main" && page.kind==="search" && p>.3 && !window.__ptSearchFound)window.__ptFindSimilar?.();
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
  const active=!!ready&&y>=start;
  if(root)root.hidden=!active;
  document.body.classList.toggle("pt-world-active",active);
  if(active){const raw=(y-start)/pageStride(),index=Math.min(pages.length-1,Math.floor(raw));show(index,Math.min(1,raw-index));}
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
