/* Original Noomo case motion: cover scale on scroll; the page contains no generated 3D objects. */
(() => {
 const gsap=window.gsap,ScrollTrigger=window.ScrollTrigger;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 if(!gsap||!ScrollTrigger)return;
 gsap.registerPlugin(ScrollTrigger);
 if(!reduced){
  gsap.fromTo('.case-hero .show-load',{opacity:0,y:30},{opacity:1,y:0,duration:.8,stagger:.09,ease:'power2.out'});
  // Transferred from Noomo's caseHero component (nextCase.vue.548e9dbe.js).
  gsap.to('.case-hero .cover-block',{scale:.82,ease:'power1.inOut',scrollTrigger:{trigger:'.case-hero .cover-block',start:'center bottom',end:'bottom top',scrub:true}});
  document.querySelectorAll('.model-section').forEach(section=>gsap.fromTo(section,{y:30,opacity:0},{y:0,opacity:1,duration:.8,ease:'power2.out',scrollTrigger:{trigger:section,start:'top 90%',toggleActions:'play none none reverse'}}));
 }
 const headerBack=document.querySelector('header .back');
 const updateHeader=()=>{if(headerBack)headerBack.style.opacity=scrollY>30?'1':'0';};
 window.addEventListener('scroll',updateHeader,{passive:true});updateHeader();
 const video=document.querySelector('video');
 if(video){
  if(reduced){video.autoplay=false;video.pause();}
  else if('IntersectionObserver' in window)new IntersectionObserver(entries=>entries.forEach(entry=>{if(entry.isIntersecting)video.play().catch(()=>{});else video.pause();}),{threshold:.05}).observe(video);
 }
 document.fonts?.ready.then(()=>ScrollTrigger.refresh());
})();
