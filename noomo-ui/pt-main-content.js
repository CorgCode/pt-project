/* Hallmark · macrostructure: original-scene-scroll · theme: original-violet · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R5 V4 · slop test: browser verification pending
 */
window.__ptMainPages = [
  {id:"problem",kind:"device",title:"Понять, что происходит в сети",paragraphs:["В компании работают сотни устройств. По списку адресов трудно понять, кто чем занят.","Оператору SOC нужен контекст: как ведёт себя устройство и на кого оно похоже."]},
  {id:"problem",kind:"traffic",title:"Поведение видно по связям",paragraphs:["С кем устройство общается, какими сервисами пользуется, сколько данных передаёт и когда активно.","Мы собираем эти признаки в числовой профиль — компактное описание сетевого поведения."]},
  {id:"model",kind:"features",title:"От одного устройства — к похожим",paragraphs:["Вы выбираете компьютер. Сервис находит близкие профили и помогает сравнить окружение.","Роли устройств могут быть неизвестны: начинаем с того, что видно в сетевых данных."]},
  {id:"product",kind:"search",title:"Контекст для оператора SOC",paragraphs:["Кого проверить рядом с выбранным устройством? С кем оно обменивалось данными? Что общего у двух профилей?"],note:"Похожесть помогает исследовать сеть. Она не является вердиктом об атаке."},
  {id:"product",kind:"device",title:"Попробуйте на своих данных",paragraphs:["В лаборатории работают загрузка CSV, поиск соседей, группы и сравнение двух окон трафика.","Наблюдавшиеся контакты показаны отдельно от числовой похожести."],note:"Учебные данные отмечены. Ваш CSV обрабатывается локально."},
  {id:"demo",kind:"device",title:"Откройте лабораторию",paragraphs:["Начните с понятного учебного примера или загрузите свои записи. Выберите устройство — и исследуйте его окружение."],note:"Архитектура, обучение и реальные результаты исследования — на странице модели."}
];
window.__ptConfigureOriginalText = function ({gsap,ScrollTrigger,layer,left,right,bottom}) {
  document.body.classList.add("pt-native-story");
  const start=850,stride=2000,pages=window.__ptMainPages;
  const end=start+(pages.length-1+.4)*stride;
  const runway=document.querySelector(".home-page");
  if(runway)runway.style.setProperty("height",`${Math.max(10500,end+innerHeight)}px`,"important");
  document.querySelector(".home-footer")?.style.setProperty("display","none","important");
  bottom.forEach(el=>{el.hidden=true;});
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  const blocks=[];
  let actions=document.querySelector('.pt-end-actions');
  if(!actions){actions=document.createElement('nav');actions.className='pt-end-actions';actions.setAttribute('aria-label','Продолжить знакомство с продуктом');actions.innerHTML='<a href="/lab/">Попробовать на данных ↗</a><a href="/model/">Как устроена модель ↗</a>';actions.hidden=true;document.body.append(actions);}
  let sequence;
  const syncVisibility=()=>{
    const time=sequence?.time()||0;
    const active=time>0?Math.min(pages.length-1,Math.floor(time)):-1;
    actions.hidden=active!==pages.length-1;
    blocks.forEach((block,index)=>{
      block.hidden=index!==active;
      block.setAttribute("aria-hidden",String(index!==active));
    });
  };
  sequence=gsap.timeline({onUpdate:syncVisibility,scrollTrigger:{trigger:"body",start:`top+=${start} top`,end:`top+=${end} top`,scrub:reduced?true:1.2,invalidateOnRefresh:true}});
  pages.forEach((page,index)=>{
    const block=index===0?left:index===1?right:(index%2?right:left).cloneNode(false);
    if(index>1)left.parentElement.append(block);
    block.classList.add("pt-product-copy");block.dataset.ptCopy=index;block.hidden=true;block.replaceChildren();
    const paragraph=document.createElement("p"),lines=[];
    [page.title,...page.paragraphs,...(page.note?[page.note]:[])].forEach((text,i)=>{
      const mask=document.createElement("span");mask.className="pt-copy-mask";
      const line=document.createElement("span");line.className=i===0?"pt-copy-line pt-copy-heading":i>page.paragraphs.length?"pt-copy-line pt-copy-note":"pt-copy-line";
      line.textContent=text;mask.append(line);paragraph.append(mask);lines.push(line);
    });block.append(paragraph);blocks.push(block);
    gsap.set(block,{opacity:1});
    gsap.set(lines,{yPercent:reduced?0:110,opacity:0});
    sequence.fromTo(lines,{yPercent:reduced?0:110,opacity:0},{yPercent:0,opacity:1,stagger:.04,duration:.24,ease:"power2.out",immediateRender:false},index+.04);
    if(index<pages.length-1)sequence.to(lines,{yPercent:reduced?0:-25,opacity:0,stagger:.02,duration:.14,ease:"power1.in"},index+.78);
    sequence.to({}, {duration:.02},index+(index===pages.length-1?.38:.98));
  });
  syncVisibility();
  window.__ptNativeCopy={blocks,start,stride,end,pages,sequence,syncVisibility};
  window.__ptGoChapter=id=>{
    const index=pages.findIndex(p=>p.id===id);if(index<0)return;
    const y=start+index*stride+stride*.4,smoother=gsap.core.globals().ScrollSmoother?.get();
    if(smoother)smoother.scrollTo(y,!reduced);else window.scrollTo({top:y,behavior:reduced?"instant":"smooth"});
  };
  requestAnimationFrame(()=>ScrollTrigger.refresh());
};
