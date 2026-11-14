const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..');
async function check(){
 const dom=new JSDOM(fs.readFileSync(path.join(root,'lab/index.html'),'utf8'),{runScripts:'outside-only',url:'http://localhost/lab/'}),w=dom.window,d=w.document;w.matchMedia=()=>({matches:true});w.innerWidth=1440;
 for(const file of ['lab/core.js','lab/changes.js'])w.eval(fs.readFileSync(path.join(root,file),'utf8'));
 assert.equal(w.__ptChangeState.current.id,'host-03');assert(w.__ptChangeState.current.added.includes('service-new'));assert(w.__ptChangeState.report.newHosts.includes('service-new'));
 assert(w.__ptChangeState.current.similarity<.999);assert.equal(w.__ptChangeState.current.after.activity.outgoing-w.__ptChangeState.current.before.activity.outgoing,70);
 d.getElementById('change-view-after').click();assert.equal(w.__ptChangeState.period,'after');assert(d.querySelector('#change-visual .change-contact.is-new'));assert.equal(d.getElementById('change-view-after').getAttribute('aria-pressed'),'true');
 d.getElementById('change-view-before').click();assert(!d.querySelector('#change-visual .change-contact.is-new'));
 const data=w.PTLab.parseCSV('src,dst,bytes,packets\na,b,20,1\nc,b,40,2');const identical=w.PTLab.compareWindows(data,data);assert(identical.changes.every(c=>Math.abs(c.similarity-1)<1e-12&&c.beforeGroup===c.afterGroup&&c.added.length===0&&c.removed.length===0));
 const vectors=w.PTLab.parseCSV('host_id,emb_0,emb_1\na,1,0\nb,0,1'),rotated=w.PTLab.parseCSV('host_id,emb_0,emb_1\na,0,1\nb,0,1');const change=w.PTLab.compareWindows(vectors,rotated).changes.find(c=>c.id==='a');assert.equal(change.similarity,0);assert.notEqual(change.beforeGroup,change.afterGroup);
 assert.throws(()=>w.PTLab.compareWindows(data,vectors));assert.throws(()=>w.PTLab.compareWindows(vectors,w.PTLab.parseCSV('host_id,emb_0\na,1\nb,2')));
 async function upload(which,text){Object.defineProperty(d.getElementById('window-'+which),'files',{configurable:true,value:[{name:which+'.csv',size:text.length,text:async()=>text}]});d.getElementById('window-'+which).dispatchEvent(new w.Event('change'));await new Promise(setImmediate);}
 await upload('before','host_id,emb_0,emb_1\na,1,0\nb,0,1');assert.equal(d.querySelector('.changes-workspace').hidden,true);assert.match(d.getElementById('change-status').textContent,/один формат/);
 await upload('after','host_id,emb_0,emb_1\na,0,1\nb,0,1');assert.equal(d.querySelector('.changes-workspace').hidden,false);assert.equal(w.__ptChangeState.current.id,'a');assert.equal(w.__ptChangeState.report.kind,'vectors');
 await upload('after','host_id,emb_0,emb_1\nx,1,0\ny,0,1');assert.equal(d.querySelector('.changes-workspace').hidden,true);assert.match(d.getElementById('change-status').textContent,/общих ID/);
 d.getElementById('change-example').click();assert.equal(d.querySelector('.changes-workspace').hidden,false);assert.equal(w.__ptChangeState.current.id,'host-03');
 w.innerWidth=375;w.dispatchEvent(new w.Event('resize'));assert.equal(d.getElementById('change-visual').getAttribute('viewBox'),'0 0 360 270');
 // More than the first twelve common IDs remain reachable, including by search.
 const many='host_id,emb_0,emb_1\n'+Array.from({length:30},(_,i)=>`device-${i},${i+1},1`).join('\n');
 await upload('before',many);await upload('after',many);
 assert.equal(d.querySelectorAll('#change-host-list button').length,12);assert.equal(d.getElementById('change-more').hidden,false);
 d.getElementById('change-more').click();assert.equal(d.querySelectorAll('#change-host-list button').length,24);
 d.getElementById('change-host-search').value='device-29';d.getElementById('change-host-search').dispatchEvent(new w.Event('input'));assert.equal(d.querySelectorAll('#change-host-list button').length,1);d.querySelector('#change-host-list button').click();assert.equal(w.__ptChangeState.current.id,'device-29');
 // A rejected file read must not erase a subsequently restored working example.
 let rejectRead;Object.defineProperty(d.getElementById('window-before'),'files',{configurable:true,value:[{name:'slow.csv',size:10,text:()=>new Promise((resolve,reject)=>{rejectRead=reject;})}]});d.getElementById('window-before').dispatchEvent(new w.Event('change'));d.getElementById('change-example').click();rejectRead(Error('stale read'));await new Promise(setImmediate);assert.equal(d.querySelector('.changes-workspace').hidden,false);assert.equal(w.__ptChangeState.current.id,'host-03');
 dom.window.close();console.log('PASS: two-window comparison, shared feature scale, stable group reference, observed contact changes, both uploads/errors and responsive diagram');
}
module.exports=check;if(require.main===module)check().catch(e=>{console.error(e);process.exitCode=1;});
