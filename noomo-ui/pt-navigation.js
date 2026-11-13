/* Hallmark · macrostructure: connected-product · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: browser review pending
 */
(() => {
 const nav=document.createElement("nav");nav.className="pt-site-nav";nav.setAttribute("aria-label","Навигация по продукту");
 const brand=document.createElement("a");brand.href="/";brand.className="pt-brand";brand.textContent="PT / NETWORK";nav.append(brand);
 [["/","Продукт"],["/model/","Модель"],["/lab/","Лаборатория"]].forEach(([href,label])=>{
  const a=document.createElement("a");a.href=href;a.textContent=label;
  if(location.pathname.replace(/\/$/,"")===href.replace(/\/$/,""))a.setAttribute("aria-current","page");nav.append(a);
 });document.body.append(nav);
 document.addEventListener("click",event=>{
  const a=event.target.closest("a");if(!a||event.defaultPrevented||event.button!==0||event.metaKey||event.ctrlKey||event.shiftKey||event.altKey||a.target||a.hasAttribute("download"))return;
  const url=new URL(a.href,location.href);
  if(url.origin!==location.origin||url.pathname===location.pathname||!/^\/(model\/?|lab\/?|)$/.test(url.pathname))return;
  if(url.pathname.startsWith("/lab")&&window.__ptLabState){
   const s=window.__ptLabState;
   try{sessionStorage.setItem("pt-lab-handoff",JSON.stringify({id:s.selected.id,kind:s.kind}));}catch{}
  }
  if(matchMedia("(prefers-reduced-motion: reduce)").matches||document.startViewTransition)return;
  event.preventDefault();document.body.classList.add("pt-route-leaving");
  setTimeout(()=>location.assign(url.href),380);
 });
 window.addEventListener("pageshow",()=>document.body.classList.remove("pt-route-leaving"));
})();
