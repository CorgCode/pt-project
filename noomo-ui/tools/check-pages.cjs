// Exercise the independent pages with real Three geometry and the production CSV/search code.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {JSDOM}=require('jsdom'),{createCanvas}=require('@napi-rs/canvas');
const THREE=require('../vendor/three.min.js'),base=path.resolve(__dirname,'..');
async function check(page){
 const dom=new JSDOM(fs.readFileSync(path.join(base,page,'index.html'),'utf8'),{runScripts:'outside-only',pretendToBeVisual:true,url:`http://127.0.0.1:8845/${page}/`}),w=dom.window,d=w.document;
 const frames=[],observers=[];w.innerWidth=1440;w.innerHeight=900;
 w.requestAnimationFrame=fn=>frames.push(fn);w.matchMedia=()=>({matches:false});
 w.HTMLCanvasElement.prototype.getContext=function(){this.cpu??=createCanvas(this.width,this.height);return this.cpu.getContext('2d');};
 w.ResizeObserver=class {constructor(fn){this.fn=fn;}observe(){this.fn();}};
 w.IntersectionObserver=class {constructor(fn){this.fn=fn;observers.push(this);}observe(){}unobserve(){}};
 const stage=d.getElementById('pt-stage');stage.getBoundingClientRect=()=>({left:0,top:0,width:650,height:550});
 class Renderer {constructor(){this.domElement=d.createElement('canvas');this.domElement.getBoundingClientRect=stage.getBoundingClientRect;}setPixelRatio(){}setClearColor(){}setSize(){}render(scene){scene.updateMatrixWorld(true);}}
 w.THREE={...THREE,WebGLRenderer:Renderer};
 const charts=[];const ChartStub=class {constructor(canvas,config){assert(!canvas.closest('[aria-hidden="true"]'),'charts mount in visible page');this.data=config.data;this.config=config;charts.push(this);}update(){}};if(page==='lab'){w.Chart=ChartStub;w.Chart.defaults={font:{}};}
 const execute=name=>w.eval(fs.readFileSync(path.join(base,name),'utf8').replace(/^import[^\n]+\n/m,'').replace('await libraryReady;',''));
 execute('product-data.js');execute('pt-page-lab.js');execute('pt-page-content.js');
 assert(d.getElementById('pt-guide'));assert(!w.__ptLabStarted,'wait for visible controls before lab init');
 execute('pt-navigation.js');
 for(const name of ['Group','Mesh','MeshBasicMaterial','MeshStandardMaterial','MeshPhysicalMaterial','SphereGeometry','BoxGeometry','PlaneGeometry','BufferGeometry','Float32BufferAttribute','Texture','Vector3','Raycaster','DoubleSide'])w[name]=THREE[name];
 execute('pt-page-world.js');
 let js=fs.readFileSync(path.join(base,'pt-page.js'),'utf8').replace(/^import[^\n]+\n/m,'');
 await w.eval(`(async()=>{${js}\n})()`);
 function frame(){const jobs=frames.splice(0);for(const f of jobs)f(100);assert(!w.__ptWorldError,w.__ptWorldError);}
 frame();
 assert(w.__ptPageDebug,'independent renderer initialized');assert(w.__ptLabStarted);
 assert.equal(d.querySelectorAll('.pt-site-nav').length,1);
 assert.equal(d.querySelector('.pt-site-nav [aria-current=page]').getAttribute('href'),`/${page}/`);
 assert.equal(w.__ptWorldDebug.root.visible,true);
 assert.equal(w.__ptWorldDebug.root.children.find(x=>x.renderOrder===500).visible,false,'page has semantic text rather than duplicate 3D copy');
 const ids=[...d.querySelectorAll('[id]')].map(e=>e.id);assert.equal(new Set(ids).size,ids.length,'unique DOM ids');
 if(page==='lab'){
  assert.equal(charts.length,2);assert(!d.getElementById('csv-file').closest('[aria-hidden=true]'));
  assert(!d.getElementById('csv-file').hasAttribute('tabindex'));assert.equal(w.__ptWorldDebug.hitTargets.length,24);
  d.querySelector('[data-preset=flows]').click();frame();assert.equal(w.__ptLabState.dataset.length,11);
  const oldId=w.__ptLabState.selected.id;d.querySelector('#lab-neighbor-buttons button').click();frame();assert.notEqual(w.__ptLabState.selected.id,oldId);
  assert.equal(charts[1].data.datasets[0].data[0],w.__ptLabState.near[0].score);
  async function upload(text){Object.defineProperty(d.getElementById('csv-file'),'files',{configurable:true,value:[{name:'mine.csv',size:text.length,text:async()=>text}]});d.getElementById('csv-file').dispatchEvent(new w.Event('change'));await new Promise(setImmediate);frame();}
  await upload('host_id,emb_0,emb_1\na,1,0\nb,1,1\nc,0,1');
  assert.equal(w.__ptLabState.dataset.length,3);assert.equal(w.__ptWorldDebug.hitTargets.length,3);
  assert.equal(charts[0].data.datasets.flatMap(x=>x.data).length,3);
  await upload('host_id,emb_0\na,no\nb,1');assert.equal(w.__ptLabState.dataset.length,3);assert.match(d.getElementById('lab-status').textContent,/Не удалось/);
  const header='host_id,'+Array.from({length:1024},(_,i)=>'emb_'+i).join(',');
  await upload(header+'\na,'+Array(1024).fill(1).join(',')+'\nb,'+Array(1024).fill(2).join(','));
  assert.equal(w.__ptLabState.stats.dimensions,1024,'accept the actual model profile size');
  assert.equal(w.__ptLabState.dataset.length,2);
  const near=w.__ptLabState.near[0].score;assert(Math.abs(near-1)<1e-12);
 }else{
  assert(d.querySelectorAll('[data-scene]').length>=6);
  observers[0].fn([{isIntersecting:true,target:d.querySelector('#profile'),boundingClientRect:{top:100}}]);frame();
  assert.equal(w.__ptWorldDebug.hitTargets.length,3);
  d.querySelector('[data-page-part=graph]').click();frame();frame();
  assert.match(d.getElementById('page-part-copy').textContent,/собственных векторов/);
  assert.equal(d.querySelector('[data-page-part=graph]').getAttribute('aria-pressed'),'true');
  w.__ptWorldDebug.hitTargets[1].userData.ptAction();frame();assert.match(d.getElementById('page-part-title').textContent,/собеседниках/);
 }
 dom.window.close();console.log(`PASS: /${page}/ navigation, native geometry, visible controls, interactive state${page==='lab'?', live charts, valid/invalid and 1024-dimensional CSV':', model blocks and semantic content'}`);
}
(async()=>{await check('model');await check('lab');})().catch(e=>{console.error(e);process.exitCode=1});
