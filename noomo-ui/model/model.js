/* Noomo case motion is retained; research figures and music use local project assets. */
(() => {
 const audio=document.getElementById('model-music'),button=document.getElementById('music-toggle');
 if(audio&&button){
  let wanted=true,starting=false;
  audio.volume=.12;
  function sync(){
   const playing=!audio.paused&&!audio.ended;
   button.setAttribute('aria-pressed',String(playing));button.setAttribute('aria-label',playing?'Выключить музыку':'Включить музыку');
   button.title=playing?'Выключить музыку':'Включить музыку';
  }
  async function play(){
   if(!wanted||starting)return;starting=true;
   try{await audio.play();delete button.dataset.blocked;}catch{button.dataset.blocked='true';button.title='Нажмите, чтобы включить музыку';}
   finally{starting=false;sync();}
  }
  button.addEventListener('click',()=>{
   wanted=audio.paused;
   if(wanted)play();else audio.pause();sync();
  });
  ['play','pause','ended','error'].forEach(type=>audio.addEventListener(type,sync));
  const unlock=event=>{if(!event.target.closest?.('#music-toggle'))play();};
  document.addEventListener('pointerdown',unlock,{once:true});document.addEventListener('keydown',unlock,{once:true});
  sync();if(wanted)play();
 }
 const gsap=window.gsap,ScrollTrigger=window.ScrollTrigger,reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const headerBack=document.querySelector('header .back');
 const updateHeader=()=>{if(headerBack)headerBack.style.opacity=scrollY>30?'1':'0';};
 window.addEventListener('scroll',updateHeader,{passive:true});updateHeader();
 if(!gsap||!ScrollTrigger)return;
 gsap.registerPlugin(ScrollTrigger);
 if(!reduced){
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
   gsap.fromTo(heading.querySelectorAll('.word-inner'),{yPercent:110,rotateX:-18,opacity:0},{yPercent:0,rotateX:0,opacity:1,duration:1.15,stagger:.055,ease:'power3.out',scrollTrigger:{trigger:heading,start:'top 91%',once:true}});
  });
  if(document.querySelector('.case-hero .info-block .show-load'))gsap.fromTo('.case-hero .info-block .show-load',{opacity:0,y:24},{opacity:1,y:0,duration:1.25,stagger:.14,ease:'power3.out'});
  if(document.querySelector('.case-hero .cover-block'))gsap.to('.case-hero .cover-block',{scale:.82,ease:'power1.inOut',scrollTrigger:{trigger:'.case-hero .cover-block',start:'center bottom',end:'bottom top',scrub:1.2}});
  document.querySelectorAll('.model-section > p,.model-section > dl,.model-section > table,.model-section > ol,.model-section > details').forEach(element=>{
   gsap.fromTo(element,{y:28,opacity:0},{y:0,opacity:1,duration:1.15,ease:'power2.out',scrollTrigger:{trigger:element,start:'top 94%',once:true}});
  });
  document.querySelectorAll('.model-figure,.source-figure').forEach(figure=>{
   gsap.fromTo(figure,{y:48,opacity:0},{y:0,opacity:1,duration:1.5,ease:'power3.out',scrollTrigger:{trigger:figure,start:'top 92%',once:true}});
   const img=figure.querySelector('img');
   if(img&&!figure.classList.contains('model-overview'))gsap.fromTo(img,{scale:.96},{scale:1,ease:'none',scrollTrigger:{trigger:figure,start:'top bottom',end:'center center',scrub:1.4}});
  });
 }
 let refreshTimer;
 const refresh=()=>{clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>ScrollTrigger.refresh(),80);};
 document.querySelectorAll('.model-figure img,.source-figure img').forEach(img=>{if(!img.complete)img.addEventListener('load',refresh,{once:true});});
 document.fonts?.ready.then(refresh);
})();
