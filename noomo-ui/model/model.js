/* Noomo case motion is retained; research figures and music use local project assets. */
(() => {
 const audio=document.getElementById('model-music'),button=document.getElementById('music-toggle');
 if(audio&&button){
  let wanted=true,starting=false;
  try{wanted=sessionStorage.getItem('pt-music')!=='off';}catch{}
  audio.volume=.12;
  function sync(){
   const playing=!audio.paused&&!audio.ended;
   button.setAttribute('aria-pressed',String(playing));button.setAttribute('aria-label',playing?'Выключить музыку':'Включить музыку');
   button.title=playing?'Выключить музыку':button.dataset.blocked?'Нажмите, чтобы включить музыку':'Включить музыку';
  }
  async function play(){
   if(!wanted||starting)return;starting=true;
   try{await audio.play();delete button.dataset.blocked;}catch{button.dataset.blocked='true';button.title='Нажмите, чтобы включить музыку';}
   finally{starting=false;sync();}
  }
  button.addEventListener('click',()=>{
   wanted=audio.paused;
   try{sessionStorage.setItem('pt-music',wanted?'on':'off');}catch{}
   if(wanted)play();else audio.pause();sync();
  });
  ['play','pause','ended','error'].forEach(type=>audio.addEventListener(type,sync));
  const unlock=event=>{if(!event.target.closest?.('#music-toggle'))play();};
  document.addEventListener('pointerdown',unlock,{once:true});document.addEventListener('keydown',unlock,{once:true});
  sync();if(wanted)play();
 }
 const gsap=window.gsap,reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const headerBack=document.querySelector('header .back');
 const updateHeader=()=>{if(headerBack)headerBack.style.opacity=scrollY>30?'1':'0';};
 window.addEventListener('scroll',updateHeader,{passive:true});updateHeader();
 if(!gsap||reduced||!window.IntersectionObserver)return;

 const reveals=new Map();
 const observer=new IntersectionObserver(entries=>{
  for(const entry of entries){
   if(!entry.isIntersecting)continue;
   reveals.get(entry.target)?.play();
   reveals.delete(entry.target);
   observer.unobserve(entry.target);
  }
 },{rootMargin:'0px 0px -40px 0px'});
 const reveal=(element,target,from,to)=>{
  reveals.set(element,gsap.fromTo(target,from,{...to,paused:true}));
  observer.observe(element);
 };

 // Preserve semantic headings while allowing each word to rise from its own mask.
 document.querySelectorAll('h1,.model-section h2,.model-section h3,.pt-site-journey h2').forEach(heading=>{
  if(!heading.textContent.trim())return;
  heading.setAttribute('aria-label',heading.textContent.trim());
  const nodes=[...heading.childNodes];heading.replaceChildren();
  nodes.forEach(node=>{
   if(node.nodeName==='BR'){heading.append(document.createElement('br'));return;}
   (node.textContent.match(/\S+|\s+/g)||[]).forEach(word=>{
    if(/^\s+$/.test(word)){heading.append(' ');return;}
    const mask=document.createElement('span'),inner=document.createElement('span');
    mask.className='word-mask';mask.setAttribute('aria-hidden','true');inner.className='word-inner';inner.textContent=word;mask.append(inner);heading.append(mask);
   });
  });
  reveal(heading,heading.querySelectorAll('.word-inner'),{yPercent:110,rotateX:-18,opacity:0},{yPercent:0,rotateX:0,opacity:1,duration:1.15,stagger:.055,ease:'power3.out'});
 });
 if(document.querySelector('.case-hero .info-block .show-load'))gsap.fromTo('.case-hero .info-block .show-load',{opacity:0,y:24},{opacity:1,y:0,duration:1.25,stagger:.14,ease:'power3.out'});
 document.querySelectorAll('.model-section > p,.model-section > dl,.model-section > table,.model-section > ol,.model-section > details').forEach(element=>{
  reveal(element,element,{y:28,opacity:0},{y:0,opacity:1,duration:1.15,ease:'power2.out'});
 });

 const cover=document.querySelector('.case-hero .cover-block');
 const figures=[...document.querySelectorAll('.model-figure,.source-figure')];
 const images=[];
 for(const figure of figures){
  reveal(figure,figure,{y:48,opacity:0},{y:0,opacity:1,duration:1.5,ease:'power3.out'});
  const img=figure.querySelector('img');
  if(img&&!figure.classList.contains('model-overview'))images.push({figure,img});
 }

 let frame=0;
 const clamp=progress=>Math.max(0,Math.min(1,progress));
 function updateScales(){
  frame=0;
  if(cover){
   const bounds=cover.getBoundingClientRect();
   const center=(bounds.top+bounds.bottom)/2;
   const progress=clamp((innerHeight-center)/(innerHeight+cover.offsetHeight/2));
   gsap.set(cover,{scale:1-.18*progress});
  }
  for(const {figure,img} of images){
   const {top,height}=figure.getBoundingClientRect();
   if(top>innerHeight*2||top+height < -innerHeight)continue;
   const progress=clamp((innerHeight-top)/((innerHeight+height)/2));
   gsap.set(img,{scale:.96+.04*progress});
  }
 }
 function scheduleScales(){
  if(!frame)frame=requestAnimationFrame(updateScales);
 }
 window.addEventListener('scroll',scheduleScales,{passive:true});
 window.addEventListener('resize',scheduleScales);
 scheduleScales();
})();
