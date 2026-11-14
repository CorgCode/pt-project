// Cross-page product invariants. This complements, rather than replaces, browser review.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..');
function check(){
 for(const file of ['index.html','model/index.html','lab/index.html']){
  const dom=new JSDOM(fs.readFileSync(path.join(root,file),'utf8'),{url:'http://localhost/'+file.replace('index.html','')}),d=dom.window.document;
  assert.equal(d.documentElement.lang,'ru',file);assert(d.querySelector('link[href^="/pt-theme.css"]'),file+' shared theme');
  const body=d.body.cloneNode(true);body.querySelectorAll('script,style').forEach(e=>e.remove());assert(!/noomo|hello@|linkedin|instagram|delight/i.test(body.textContent),file+' public copy');
  for(const a of d.querySelectorAll('a[href]')){
   const href=a.getAttribute('href');assert(!/noomoagency|mailto:hello|twitter.com\/noomo|linkedin.com\/company\/noomo/i.test(href),href);
   if(href.startsWith('/')&&!href.startsWith('//')){const target=new URL(href,'http://localhost'),local=path.join(root,target.pathname);assert(fs.existsSync(local),href);if(target.hash){const html=target.pathname.endsWith('/')?path.join(local,'index.html'):local;if(fs.existsSync(html)&&html.endsWith('.html')){const doc=new JSDOM(fs.readFileSync(html,'utf8')).window.document;assert(doc.getElementById(target.hash.slice(1)),href);}}}
  }
  dom.window.close();
 }
 const adapter=fs.readFileSync(path.join(root,'pt-adapter.js'),'utf8');const fbo=adapter.slice(adapter.indexOf('window.__ptRenderTargetScene'));
 const scope={};new Function('window',fbo)(scope);
 const texture={},sample={isMesh:true,visible:true,material:{buffer:texture}},hidden={isMesh:true,visible:false,material:{buffer:texture}},safe={isMesh:true,visible:true,material:{}},array={isMesh:true,visible:true,material:[{}, {buffer:texture}]};
 const world={scene:{traverse(fn){[sample,hidden,safe,array].forEach(fn);}},camera:{},gl:{render(){assert.equal(sample.visible,false);assert.equal(array.visible,false);assert.equal(safe.visible,true);throw Error('render interrupted');}}};
 assert.throws(()=>scope.__ptRenderTargetScene(world,{texture}),/interrupted/);assert.equal(sample.visible,true);assert.equal(array.visible,true);assert.equal(hidden.visible,false);
 const publicModules=['C4-CPqZe.js','403X8TDT.js','3GHhKpoT.js','CA3Rur4S.js'];for(const file of publicModules)assert(!/noomoagency\.com|hello@noomoagency/.test(fs.readFileSync(path.join(root,'_nuxt',file),'utf8')),file);
 console.log('PASS: three Russian product pages, shared theme, local routes/fragments, no borrowed public branding and safe offscreen-material restoration');
}
module.exports=check;if(require.main===module)try{check();}catch(error){console.error(error);process.exitCode=1;}
