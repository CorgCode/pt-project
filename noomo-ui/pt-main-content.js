/* Hallmark · macrostructure: original-scene-scroll · theme: original-violet · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R5 V4 · slop test: browser verification pending
 */
window.__ptMainPages = [
  {id:"problem",kind:"device",title:"Зачем изучать сеть",paragraphs:["В сети компании работают компьютеры, серверы и служебные системы. Они постоянно обмениваются данными.","Чтобы разобраться в работе сети, важно понимать, какие устройства ведут себя похоже и что их объединяет."]},
  {id:"problem",kind:"traffic",title:"Адреса недостаточно",paragraphs:["По адресу компьютера не видно, чем он занят и какое место занимает в сети.","Взаимодействия дают больше контекста: с кем он общается, какие сервисы использует и когда активен. Из этого складывается картина его поведения."]},
  {id:"model",kind:"features",title:"Что мы предлагаем?",paragraphs:["Наша ML-модель сама находит сходства в сетевом поведении устройств.","Вы выбираете компьютер — сервис показывает, на какие устройства он похож. Для этого не нужно заранее знать их назначение."]},
  {id:"product",kind:"search",title:"Чем вам будет полезен наш продукт?",paragraphs:["Вместо поиска по всей сети вы получаете устройства с похожим поведением. Можно изучать их вместе и сравнивать окружение."],note:"Здесь — пример на учебных данных."},
  {id:"product",kind:"device",title:"Видеть сеть целиком",paragraphs:["Наша цель — показывать группы похожих устройств и замечать изменения в их поведении."],note:"Сейчас в прототипе работает поиск похожих устройств."},
  {id:"demo",kind:"device",title:"От устройства — к окружению",paragraphs:["Сначала изучаете одно устройство, затем — похожие на него. Так можно разбирать сеть по частям, а не просматривать все адреса подряд."]}
];
window.__ptConfigureOriginalText = function ({gsap,ScrollTrigger,layer,left,right,bottom}) {
  document.body.classList.add("pt-native-story");
  const start=850,stride=2000,pages=window.__ptMainPages;
  const runway=document.querySelector(".home-page");
  if(runway)runway.style.setProperty("height",`${Math.max(13000,start+pages.length*stride)+innerHeight}px`,"important");
  bottom.forEach(el=>{el.hidden=true;});
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  const blocks=[];
  pages.forEach((page,index)=>{
    const block=index===0?left:index===1?right:(index%2?right:left).cloneNode(false);
    if(index>1)left.parentElement.append(block);
    block.classList.add("pt-product-copy");block.replaceChildren();
    const paragraph=document.createElement("p"),lines=[];
    [page.title,...page.paragraphs,...(page.note?[page.note]:[])].forEach((text,i)=>{
      const mask=document.createElement("span");mask.className="pt-copy-mask";
      const line=document.createElement("span");line.className=i===0?"pt-copy-line pt-copy-heading":i>page.paragraphs.length?"pt-copy-line pt-copy-note":"pt-copy-line";
      line.textContent=text;mask.append(line);paragraph.append(mask);lines.push(line);
    });block.append(paragraph);blocks.push(block);
    gsap.set(block,{opacity:1});
    const sequence=gsap.timeline({scrollTrigger:{trigger:"body",start:`top+=${start+index*stride} top`,end:`top+=${start+(index+1)*stride} top`,scrub:reduced?true:1.2,invalidateOnRefresh:true}});
    sequence.fromTo(lines,{yPercent:reduced?0:110,opacity:0},{yPercent:0,opacity:1,stagger:.06,duration:.28,ease:"power2.out"},0);
    sequence.to({}, {duration:.35},.45);
    if(index<pages.length-1)sequence.to(lines,{yPercent:reduced?0:-25,opacity:0,stagger:.025,duration:.2,ease:"power1.in"},.8);
  });
  window.__ptNativeCopy={blocks,start,stride,pages};
  window.__ptGoChapter=id=>{
    const index=pages.findIndex(p=>p.id===id);if(index<0)return;
    const y=start+index*stride+stride*.4,smoother=gsap.core.globals().ScrollSmoother?.get();
    if(smoother)smoother.scrollTo(y,!reduced);else window.scrollTo({top:y,behavior:reduced?"instant":"smooth"});
  };
  requestAnimationFrame(()=>ScrollTrigger.refresh());
};
