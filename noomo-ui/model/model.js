/* Noomo case motion is retained; research figures and music use local project assets. */
(() => {
 const audio=document.getElementById('model-music'),button=document.getElementById('music-toggle');
 if(audio&&button){
  let wanted=true,starting=false;
  try{wanted=sessionStorage.getItem('pt-model-music')!=='off';}catch{}
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
   try{sessionStorage.setItem('pt-model-music',wanted?'on':'off');}catch{}
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
  gsap.fromTo('.case-hero .show-load',{opacity:0,y:30},{opacity:1,y:0,duration:.8,stagger:.09,ease:'power2.out'});
  gsap.to('.case-hero .cover-block',{scale:.82,ease:'power1.inOut',scrollTrigger:{trigger:'.case-hero .cover-block',start:'center bottom',end:'bottom top',scrub:true}});
  document.querySelectorAll('.model-section').forEach(section=>gsap.fromTo(section,{y:30,opacity:0},{y:0,opacity:1,duration:.8,ease:'power2.out',scrollTrigger:{trigger:section,start:'top 90%',toggleActions:'play none none reverse'}}));
 }
 let refreshTimer;
 const refresh=()=>{clearTimeout(refreshTimer);refreshTimer=setTimeout(()=>ScrollTrigger.refresh(),80);};
 document.querySelectorAll('.model-figure img,.source-figure img').forEach(img=>{if(!img.complete)img.addEventListener('load',refresh,{once:true});});
 document.fonts?.ready.then(refresh);
})();
