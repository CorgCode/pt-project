// Exercise the new ordinary workbench with real CSV/math and DOM interactions.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict');
const {JSDOM}=require('jsdom'),root=path.resolve(__dirname,'..');
async function check(){
 const html=fs.readFileSync(path.join(root,'lab/index.html'),'utf8');
 const dom=new JSDOM(html,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://localhost/lab/'}),w=dom.window,d=w.document;
 w.matchMedia=()=>({matches:true});w.innerWidth=1440;
 const charts=[];w.Chart=class{static defaults={font:{}};constructor(canvas,config){assert(canvas.isConnected);this.data=config.data;this.options=config.options;this.type=config.type;charts.push(this);}update(){}};
 const run=file=>w.eval(fs.readFileSync(path.join(root,file),'utf8'));
 run('product-data.js');run('lab/core.js');run('lab/lab.js');
 assert.equal(d.documentElement.lang,'ru');assert.equal(charts.length,3);assert.equal(w.__ptLabState.dataset.length,11);assert.equal(w.__ptLabState.kind,'flows');
 assert.equal(d.querySelectorAll('#neighbor-network [role=button]').length,6);
 assert(!d.querySelector('script[src*="three"],script[src*="pt-page"],iframe'),'no rejected 3D shell');
 for(const el of d.querySelectorAll('script[src],link[href],img[src],audio[src]')){const url=el.getAttribute('src')||el.getAttribute('href');assert(fs.statSync(path.join(root,url.split('?')[0])).size>0,url);}
 const ids=[...d.querySelectorAll('[id]')].map(x=>x.id);assert.equal(ids.length,new Set(ids).size);
 const first=w.__ptLabState.selected.id;d.querySelector('#lab-neighbor-buttons button').click();assert.notEqual(w.__ptLabState.selected.id,first);
 assert.equal(charts[1].data.datasets[0].data[0],w.__ptLabState.near[0].score);
 const next=w.__ptLabState.near[1].host.id;d.getElementById('compare-host').value=next;d.getElementById('compare-host').dispatchEvent(new w.Event('change'));assert.equal(w.__ptLabState.comparator.id,next);
 const map=charts[0];const plotted=map.data.datasets.findIndex(ds=>ds.data.length);const target=map.data.datasets[plotted].data[0].hostId;map.options.onClick(null,[{datasetIndex:plotted,index:0}]);assert.equal(w.__ptLabState.selected.id,target);
 const k=d.getElementById('group-count');k.value='2';k.dispatchEvent(new w.Event('change'));assert(w.__ptLabState.groups.every(x=>x<2));
 d.querySelector('[data-preset=small]').click();assert.equal(w.__ptLabState.dataset.length,24);d.querySelector('[data-preset=project]').click();assert.equal(w.__ptLabState.dataset.length,300);
 async function upload(text,name='mine.csv',size=text.length){Object.defineProperty(d.getElementById('csv-file'),'files',{configurable:true,value:[{name,size,text:async()=>text}]});d.getElementById('csv-file').dispatchEvent(new w.Event('change'));await new Promise(setImmediate);}
 await upload('host_id,emb_0,emb_1\na,1,0\nb,1,1\nc,0,1');assert.equal(w.__ptLabState.dataset.length,3);assert.equal(w.__ptLabState.selected.id,'a');assert(Math.abs(w.__ptLabState.near[0].score-Math.SQRT1_2)<1e-12);
 assert.equal(charts[0].data.datasets.flatMap(ds=>ds.data).length,3);assert.equal(d.querySelectorAll('#neighbor-network [role=button]').length,3);
 await upload('host_id,emb_0\na,bad\nb,1');assert.equal(w.__ptLabState.dataset.length,3);assert.equal(d.getElementById('lab-status').dataset.error,'true');
 await upload('anything','large.csv',4*1024*1024);assert.match(d.getElementById('lab-status').textContent,/слишком большой/);
 const header='host_id,'+Array.from({length:1024},(_,i)=>'emb_'+i).join(',');await upload(header+'\na,'+Array(1024).fill(1).join(',')+'\nb,'+Array(1024).fill(2).join(','));assert.equal(w.__ptLabState.stats.dimensions,1024);assert(Math.abs(w.__ptLabState.near[0].score-1)<1e-12);assert(w.__ptLabState.projection.every(p=>Number.isFinite(p.x)&&Number.isFinite(p.y)));
 for(const csv of ['host_id,emb_0\na,0\nb,1','host_id,emb_0\na,1\na,2','host_id,emb_1\na,1\nb,2','src,dst,bytes\na,b,-1','src,dst\n"broken,b','host_id,emb_0\na,Infinity\nb,1'])assert.throws(()=>w.PTLab.parseCSV(csv));
 assert.equal(w.PTLab.parseCSV('src,dst\n'+'a,b\n'.repeat(10000)).stats.flows,10000);assert.throws(()=>w.PTLab.parseCSV('src,dst\n'+'a,b\n'.repeat(10001)));assert.throws(()=>w.PTLab.parseCSV('src,dst,bytes\na,b,1e308\na,b,1e308'));
 const parsed=w.PTLab.parseCSV('\uFEFFhost_id;emb_0;emb_1\n"a;one";1;0\nb;0;1');assert.equal(parsed.dataset[0].id,'a;one');
 const grouping=[{id:'a',v:[1,0],norm:1},{id:'b',v:[.99,.01],norm:Math.hypot(.99,.01)},{id:'c',v:[0,1],norm:1},{id:'d',v:[.01,.99],norm:Math.hypot(.01,.99)}];const groups=w.PTLab.group(grouping,2);assert.equal(groups[0],groups[1]);assert.equal(groups[2],groups[3]);assert.notEqual(groups[0],groups[2]);assert.deepEqual(groups,w.PTLab.group(grouping,2));
 w.innerWidth=375;w.dispatchEvent(new w.Event('resize'));assert.equal(d.getElementById('neighbor-network').getAttribute('viewBox'),'0 0 360 650');
 await upload('host_id,emb_0,emb_1\n"<img src=x>",1,0\nb,0,1');assert(!d.querySelector('#selected-name img'));assert.equal(d.getElementById('selected-name').textContent,'<img src=x>');
 let release;const pending=new Promise(resolve=>release=resolve);Object.defineProperty(d.getElementById('csv-file'),'files',{configurable:true,value:[{name:'late.csv',size:80,text:()=>pending}]});d.getElementById('csv-file').dispatchEvent(new w.Event('change'));
 d.querySelector('[data-preset=flows]').click();release('host_id,emb_0\na,1\nb,2');await new Promise(setImmediate);assert.equal(w.__ptLabState.kind,'flows');assert.equal(w.__ptLabState.dataset.length,11);
 dom.window.close();console.log('PASS: /lab/ live selection, three charts, deterministic grouping, CSV imports/errors/1024D, safe rendering and responsive neighborhood');
}
module.exports=check;if(require.main===module)check().catch(e=>{console.error(e);process.exitCode=1;});
