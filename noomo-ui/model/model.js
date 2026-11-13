/* Hallmark · macrostructure: glass-model-dissection · theme: amethyst-studio · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V5 · slop test: browser review pending
 */
import {ModelScene} from './scene.js?v=1';
if(document.readyState==='loading'||(!window.THREE&&document.readyState!=='complete'))await new Promise(resolve=>document.addEventListener('DOMContentLoaded',resolve,{once:true}));
const sections=[...document.querySelectorAll('[data-scene]')],progress=document.getElementById('reading-progress'),caption=document.getElementById('scene-caption');
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const descriptions={
 history:['Устойчивый профиль устройства','Основные 896 чисел описывают известное устройство по истории. Они помогают сопоставлять его поведение в разные дни.'],
 contacts:['Кого устройство встретит дальше','64 числа учатся находить соседей, чьё окружение помогает предсказывать новых собеседников. Это отдельный сигнал о будущих контактах.'],
 graph:['Место в структуре сети','64 собственных вектора графа дополняют поведение сведениями об окружении. Графовый блок имеет небольшой вес — 0,1.']
};
const captions=['Профиль устройства','Из записей — в признаки','Как работает кодировщик','История нескольких дней','896 + 64 + 64','Адаптация свежего окна','Разделение по времени','Баллы из 100','Что модель пока не доказывает'];
let scene=null,lastTime=0,active=-1;
const reveal=[...document.querySelectorAll('.chapter-copy > h1,.chapter-copy > h2,.chapter-copy > p,.chapter-copy > figure,.chapter-copy > dl,.chapter-copy > .parts,.chapter-copy > .history-strip,.chapter-copy > .training-split')].map(element=>({element,y:0}));
function revealText(){
 if(reduced)return;
 const values=reveal.map(item=>{
  const top=item.element.getBoundingClientRect().top-item.y,t=Math.max(0,Math.min(1,(innerHeight*.94-top)/(innerHeight*.18))),ease=1-Math.pow(1-t,3);
  return {item,alpha:ease,y:(1-ease)*30};
 });
 for(const {item,alpha,y} of values){item.y=y;item.element.style.opacity=alpha;item.element.style.transform=`translate3d(0,${y}px,0)`;}
}
try{scene=new ModelScene(document.getElementById('model-stage'));}catch(error){console.error('Model scene:',error);document.getElementById('scene-fallback').hidden=false;document.querySelector('.scene-tools').hidden=true;}
document.getElementById('model-stage').addEventListener('model:unavailable',()=>{scene?.dispose();scene=null;document.getElementById('scene-fallback').hidden=false;document.querySelector('.scene-tools').hidden=true;});
const resize=()=>scene?.resize();window.addEventListener('resize',resize,{passive:true});
document.getElementById('reset-view').addEventListener('click',()=>scene?.reset());
document.querySelectorAll('[data-part]').forEach((button,index)=>button.addEventListener('click',()=>{
 document.querySelectorAll('[data-part]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 const [title,copy]=descriptions[button.dataset.part];document.getElementById('part-title').textContent=title;document.getElementById('part-copy').textContent=copy;scene?.selectPart(index);
}));
function position(){
 const focus=innerHeight*.44;let index=0;
 for(let i=0;i<sections.length;i++){if(sections[i].getBoundingClientRect().top<=focus)index=i;}
 const rect=sections[index].getBoundingClientRect(),local=Math.max(0,Math.min(1,(focus-rect.top)/Math.max(1,rect.height)));
 const amount=Math.max(0,Math.min(1,scrollY/Math.max(1,document.documentElement.scrollHeight-innerHeight)));progress.style.transform=`scaleX(${amount})`;
 scene?.setChapter(index,local,sections[index].classList.contains('on-right'));
 if(index!==active){active=index;caption.textContent=captions[index];sections.forEach((s,i)=>s.classList.toggle('is-active',i===index));}
 return {index,local,amount};
}
function frame(time){requestAnimationFrame(frame);if(document.hidden)return;position();revealText();const delta=lastTime?Math.min((time-lastTime)/1000,.05):.016;lastTime=time;try{scene?.update(delta,reduced);}catch(error){console.error('Model frame:',error);scene?.dispose();scene=null;document.getElementById('scene-fallback').hidden=false;}}
position();requestAnimationFrame(frame);
window.__ptModelDebug={sections,scene,position,revealText};
