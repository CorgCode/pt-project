// Model-page integration with real Three geometry and a CPU renderer stub; not GPU visual QA.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {JSDOM}=require('jsdom'),{createCanvas,GlobalFonts}=require('@napi-rs/canvas'),THREE=require('../vendor/three.min.js');
const base=path.resolve(__dirname,'..');
async function check(){
 const dom=new JSDOM(fs.readFileSync(path.join(base,'model/index.html'),'utf8'),{runScripts:'outside-only',pretendToBeVisual:true,url:'http://127.0.0.1:8845/model/'}),w=dom.window,d=w.document;
 const style=d.createElement('style');style.textContent=fs.readFileSync(path.join(base,'model/model.css'),'utf8');d.head.append(style);
 GlobalFonts.registerFromPath(path.join(base,'fonts/onest-600.ttf'),'ModelOnest');
 w.HTMLCanvasElement.prototype.getContext=function(){if(!this.cpu||this.cpu.width!==this.width||this.cpu.height!==this.height)this.cpu=createCanvas(this.width,this.height);return this.cpu.getContext('2d');};
 let scroll=0;const frames=[];w.requestAnimationFrame=fn=>frames.push(fn);w.matchMedia=()=>({matches:false});w.innerWidth=1440;w.innerHeight=900;
 Object.defineProperty(w,'scrollY',{get:()=>scroll});
 const sections=[...d.querySelectorAll('[data-scene]')];assert.equal(sections.length,9);
 sections.forEach((el,i)=>el.getBoundingClientRect=()=>({top:i*1300-scroll,height:1300}));
 Object.defineProperty(d.documentElement,'scrollHeight',{value:sections.length*1300,configurable:true});
 class Renderer{constructor(){this.domElement=d.createElement('canvas');}setPixelRatio(){}setClearColor(){}setSize(w,h){this.width=w;this.height=h;}render(scene,camera){scene.updateMatrixWorld(true);camera.updateMatrixWorld(true);}dispose(){this.disposed=true;}}
 w.THREE={...THREE,WebGLRenderer:Renderer};
 w.eval(fs.readFileSync(path.join(base,'model/scene.js'),'utf8').replace('export class ModelScene','window.ModelScene = class ModelScene'));
 await w.eval('(async()=>{'+fs.readFileSync(path.join(base,'model/model.js'),'utf8').replace(/^import[^\n]*\n/m,'')+'\n})()');
 const debug=w.__ptModelDebug;assert(debug.scene,'standalone 3D scene initialized');assert(d.getElementById('scene-fallback').hidden);
 function frame(time){const callback=frames.shift();assert(callback);callback(time);}
 for(let index=0;index<9;index++){
  scroll=index*1300+450;frame(index*2000+16);
  assert.equal(debug.scene.chapter,index);assert.equal(d.querySelectorAll('.chapter.is-active').length,1);
  for(let i=0;i<100;i++)debug.scene.update(.016,false);
  const expected=['profile','data','encoder','history','profile','adapter','training','results','profile'][index];
  assert(debug.scene.groups.find(g=>g.name===expected).userData.alpha>.99);
  debug.scene.root.traverse(object=>{for(const value of [...object.position.toArray(),...object.scale.toArray()])assert(Number.isFinite(value));if(object.isMesh){assert(object.geometry.attributes.position.count>0);assert(Number.isFinite(object.material.opacity));}});
 }
 d.querySelector('[data-part=graph]').click();assert.equal(debug.scene.part,2);assert.match(d.getElementById('part-copy').textContent,/собственных вектора/);
 assert.equal(d.querySelectorAll('[data-part][aria-pressed=true]').length,1);
 const canvas=debug.scene.renderer.domElement;canvas.dispatchEvent(new w.MouseEvent('pointerdown',{clientX:100,clientY:100}));canvas.dispatchEvent(new w.MouseEvent('pointermove',{clientX:180,clientY:105}));canvas.dispatchEvent(new w.MouseEvent('pointerup'));assert(debug.scene.dragYaw>.1);
 d.getElementById('reset-view').click();assert.equal(debug.scene.dragYaw,0);
 for(const [width,height] of [[320,568],[375,812],[414,896],[768,1024],[1440,900]]){
  w.innerWidth=width;w.innerHeight=height;w.dispatchEvent(new w.Event('resize'));debug.scene.update(.05,false);
  assert(Math.abs(debug.scene.camera.aspect-width/height)<1e-9);assert.equal(debug.scene.renderer.width,width);
  assert(Number.isFinite(debug.scene.root.scale.x));
 }
 const ids=[...d.querySelectorAll('[id]')].map(el=>el.id);assert.equal(new Set(ids).size,ids.length);
 for(const el of d.querySelectorAll('script[src],link[href]')){const url=el.getAttribute('src')||el.getAttribute('href');if(url.startsWith('/'))assert(fs.existsSync(path.join(base,url.split('?')[0])),url);}
 assert(!d.querySelector('script[src*="pt-main-content"],script[src*="pt-page.js"],iframe'));
 debug.scene.dispose();dom.window.close();
 console.log('PASS: /model/ 9 chapters, real glass geometry and section transitions, 3 vector parts, drag/reset, viewport resizing, independent assets');
}
module.exports=check;if(require.main===module)check().catch(e=>{console.error(e);process.exitCode=1;});
