/* Local music and one-time reveals for the model page. */
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
 document.querySelectorAll('h1,.model-section h2,.model-section h3,.model-closing h2').forEach(heading=>{
  if(!heading.textContent.trim())return;
  heading.setAttribute('aria-label',heading.textContent.trim());
  const nodes=[...heading.childNodes];heading.replaceChildren();
  nodes.forEach(node=>{
   if(node.nodeName==='BR'||node.nodeName==='EM'){heading.append(node);return;}
   (node.textContent.match(/\S+|\s+/g)||[]).forEach(word=>{
    if(/^\s+$/.test(word)){heading.append(' ');return;}
    const mask=document.createElement('span'),inner=document.createElement('span');
    mask.className='word-mask';mask.setAttribute('aria-hidden','true');inner.className='word-inner';inner.textContent=word;mask.append(inner);heading.append(mask);
   });
  });
  reveal(heading,heading.querySelectorAll('.word-inner'),{yPercent:110,rotateX:-18,opacity:0},{yPercent:0,rotateX:0,opacity:1,duration:1.15,stagger:.055,ease:'power3.out'});
 });
 document.querySelectorAll('.model-figure,.source-figure').forEach(figure=>{
  reveal(figure,figure,{y:24,opacity:0},{y:0,opacity:1,duration:1.05,ease:'power3.out'});
 });
})();
