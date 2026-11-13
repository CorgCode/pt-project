/* Hallmark · macrostructure: model-anatomy / network-workbench · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: browser review pending
 */
import "/pt-page-world.js?v=9";
if(document.readyState==="loading")await new Promise(resolve=>document.addEventListener("DOMContentLoaded",resolve,{once:true}));
window.__ptMountContent?.();
const $=id=>document.getElementById(id),page=document.body.dataset.page;
let sceneIndex=0,context=null,lastTime=0,rendering=false;
const frameJobs=[];
const pageState=(kind,index=0)=>({visible:true,page:{id:page==="lab"?"demo":kind==="training"?"training":kind==="results"?"results":"model",kind,title:"",paragraphs:[],guided:true},index,total:6,progress:.5});
function setScene(kind,index){sceneIndex=index;window.__ptWorldState=pageState(kind,index);}
function move(id,slot){const el=$(id);if(el)$(slot).append(el);return el;}
if(page==="lab"){
 const controls=document.querySelector(".pt-lab-controls");$("lab-controls-slot").append(controls);
 controls.querySelector(".lab-presets").className="lab-data-buttons";
 $("upload-zone").className="lab-upload";
 document.querySelector(".pt-lab-selection").querySelector("label").firstChild.textContent="Хост для сравнения";
 $("lab-selection-slot").append(document.querySelector(".pt-lab-selection"));
 move("lab-neighbor-buttons","lab-neighbor-slot");
 for(const id of ["lab-insight-title","lab-insight-copy","lab-summary"])move(id,"lab-observation-slot");
 move("lab-map-chart","lab-map-slot");move("lab-neighbors-chart","lab-bars-slot");
 controls.querySelectorAll("button,input,select,summary").forEach(el=>el.removeAttribute("tabindex"));
 document.querySelectorAll(".lab-inspector button,.lab-inspector select").forEach(el=>el.removeAttribute("tabindex"));
 window.__ptStartLab?.();
 // Only generated fixture selection crosses pages. User CSV is never persisted.
 try{
  const handoff=JSON.parse(sessionStorage.getItem("pt-lab-handoff")||"null");sessionStorage.removeItem("pt-lab-handoff");
  if(handoff?.kind==="project")window.__ptSelectHost?.(handoff.id);
 }catch{}
 setScene("graph-final",0);window.__ptGraphRevealed=true;
 const caption=document.querySelector(".stage-caption>span");caption.textContent="Узел — выбрать или тянуть · фон — вращать";
 const zoom=document.createElement("div");zoom.className="lab-view-controls";
 for(const [text,scale] of [["−",.85],["+",1.15]]){const b=document.createElement("button");b.type="button";b.textContent=text;b.setAttribute("aria-label",scale<1?"Отдалить граф":"Приблизить граф");b.onclick=()=>window.__ptZoomView?.(scale);zoom.append(b);}
 document.querySelector(".stage-caption").append(zoom);
 function updateLabels(){
  const state=window.__ptLabState;if(!state)return;
  const kind=state.kind==="flows"?"flows":state.kind==="project"?(state.dataset.length===24?"small":"project"):"uploaded";
  document.querySelectorAll("[data-preset]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.preset===kind)));
  const buttons=$("lab-neighbor-buttons").querySelectorAll("button");
  buttons.forEach((b,i)=>{const score=document.createElement("strong");score.textContent=state.near[i].score.toFixed(3);b.append(score);});
 }
 window.addEventListener("pt:lab-update",updateLabels);updateLabels();
 $("lab-mode").style.display="block";
 // Native chart fallback remains useful when GPU acceleration is unavailable.
}else{
 window.__ptStartLab?.();
 setScene("features",0);
 const sections=[...document.querySelectorAll("[data-scene]")];
 const sectionObserver=new IntersectionObserver(entries=>{
  const visible=entries.filter(e=>e.isIntersecting).sort((a,b)=>a.boundingClientRect.top-b.boundingClientRect.top);
  if(visible.length){const section=visible[0].target;setScene(section.dataset.scene,sections.indexOf(section));}
 },{rootMargin:"-15% 0px -45% 0px",threshold:0});
 sections.forEach(s=>sectionObserver.observe(s));
 const descriptions={
  history:["Устойчивый профиль хоста","Основные 896 чисел описывают известный хост по истории. Поздний трафик сопоставляется с ними через адаптер. Так случайный час меньше влияет на сравнение."],
  contacts:["Будущие собеседники","64 числа учатся находить соседей, чьё окружение полезно для прогноза новых контактов. Голова обучается на днях 03–12 с ограничением отклонения от соседств основного профиля."],
  graph:["Место в структуре сети","64 собственных вектора матрицы взаимодействий дополняют поведенческий профиль сведениями о связях. Вес блока — 0,1: он почти не влияет на косинусное сравнение, но помогает классификатору связей."]
 };
 document.querySelectorAll("[data-page-part]").forEach(button=>button.addEventListener("click",()=>{
  document.querySelectorAll("[data-page-part]").forEach(b=>b.setAttribute("aria-pressed",String(b===button)));
  const [title,copy]=descriptions[button.dataset.pagePart];$("page-part-title").textContent=title;$("page-part-copy").textContent=copy;
  setScene("architecture",3);
  const source=document.querySelector(`[data-model-part="${button.dataset.pagePart}"]`);source?.click();
  // Use the exact same mesh action as a direct hit in the 3D scene.
  window.requestAnimationFrame(()=>window.__ptWorldDebug?.hitTargets[["history","contacts","graph"].indexOf(button.dataset.pagePart)]?.userData.ptAction?.());
 }));
 window.addEventListener("pt:inspect",event=>{
  $("page-part-title").textContent=event.detail.title;$("page-part-copy").textContent=event.detail.paragraphs.join(" ");
  const active=document.querySelector("#pt-guide [data-model-part][aria-pressed=true]")?.dataset.modelPart;
  document.querySelectorAll("[data-page-part]").forEach(b=>b.setAttribute("aria-pressed",String(b.dataset.pagePart===active)));
 });
}
// Semantic content remains visible if motion or WebGL is unavailable.
const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
if(!reduced&&"IntersectionObserver" in window){
 const observer=new IntersectionObserver(entries=>entries.forEach(e=>{if(e.isIntersecting){e.target.classList.add("is-visible");observer.unobserve(e.target);}}),{threshold:.05});
 document.body.classList.add("page-motion");document.querySelectorAll(".page-reveal").forEach(s=>observer.observe(s));
}
document.querySelectorAll("[data-reset-view]").forEach(b=>b.onclick=()=>window.__ptResetView?.());
function fallback(message){
 document.querySelector(".stage-fallback").hidden=false;document.querySelector(".stage-status").textContent=message;
}
function render(time){
 if(!rendering)return;
 requestAnimationFrame(render);
 if(document.hidden||!context)return;
 const dt=lastTime?Math.min((time-lastTime)/1000,.05):.016;lastTime=time;
 window.__ptWorldFrame({...context,delta:dt});
 if(window.__ptWorldError){fallback("Не удалось показать 3D. Данные и описание доступны.");rendering=false;return;}
 context.renderer.render(context.scene,context.camera);
 for(const job of frameJobs.splice(0))job();
}
try{
 const T=window.THREE,stage=$("pt-stage");
 const renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:"high-performance"});
 renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.8));renderer.setClearColor(0,0);
 renderer.toneMapping=T.ACESFilmicToneMapping;renderer.toneMappingExposure=1.15;renderer.outputColorSpace=T.SRGBColorSpace;
 stage.append(renderer.domElement);renderer.domElement.setAttribute("aria-label",page==="lab"?"3D-граф похожих хостов":"3D-схема модели");
 const scene=new T.Scene(),camera=new T.PerspectiveCamera(40,1,.1,100);camera.position.set(0,0,8);scene.add(camera);
 scene.add(new T.HemisphereLight(0xf4e5ff,0x50365f,2.5));
 for(const [x,y,z,intensity] of [[3,4,5,3],[-3,1,2,2],[1,-2,-2,1.5]]){
  const light=new T.DirectionalLight(0xffffff,intensity);light.position.set(x,y,z);scene.add(light);
 }
 context={scene,camera,renderer};
 const resize=()=>{const {width,height}=stage.getBoundingClientRect();if(!width||!height)return;renderer.setSize(width,height,false);camera.aspect=width/height;camera.updateProjectionMatrix();window.__ptWorldResize?.();};
 new ResizeObserver(resize).observe(stage);resize();
 if(T.RGBELoader){new T.RGBELoader().load('/hdri/photo_studio_01_1k.hdr',texture=>{
  const pmrem=new T.PMREMGenerator(renderer),env=pmrem.fromEquirectangular(texture);scene.environment=env.texture;texture.dispose();pmrem.dispose();
 },undefined,()=>{/* Existing lights keep the diagram legible without the HDR. */});}
 renderer.domElement.addEventListener("webglcontextlost",event=>{event.preventDefault();rendering=false;fallback("3D-контекст потерян. Обновите страницу, чтобы восстановить сцену.");});
 document.querySelector(".stage-status").textContent=page==="lab"?"Граф сходства · до 24 узлов":"Архитектура в объёме · можно вращать";
 window.__ptPageDebug=context;rendering=true;requestAnimationFrame(render);
}catch(error){console.error("PT page renderer:",error);fallback("3D недоступно. Описание и графики работают без него.");}
