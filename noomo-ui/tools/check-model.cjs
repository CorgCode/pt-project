// Verify the transferred case template and local media; no synthetic 3D renderer is used.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {JSDOM}=require('jsdom'),root=path.resolve(__dirname,'..');
async function check(){
 const html=fs.readFileSync(path.join(root,'model/index.html'),'utf8'),dom=new JSDOM(html,{runScripts:'outside-only',url:'http://127.0.0.1:8845/model/'}),d=dom.window.document;
 assert.equal(d.documentElement.lang,'ru');assert(d.querySelector('.case-hero .wrapper .info-block .right'),'original Noomo case structure');
 assert.equal(d.querySelectorAll('.model-section').length,8);assert.equal(d.querySelectorAll('canvas,iframe,#model-stage,.chapter').length,0);
 assert(!d.querySelector('script[src*="three"],script[src*="scene.js"],script[src*="pt-page"]'),'rejected 3D implementation is disconnected');
 const video=d.querySelector('video');assert(video.hasAttribute('muted')&&video.hasAttribute('playsinline'));assert(d.querySelector('.source-figure img'));
 assert(!/FWA|Awards|Wallis|Salesforce/.test(d.body.textContent),'no borrowed client claims or awards');
 for(const element of d.querySelectorAll('script[src],link[href],img[src],video[poster],source[src]')){
  const url=element.getAttribute('src')||element.getAttribute('href')||element.getAttribute('poster');if(url.startsWith('/'))assert(fs.statSync(path.join(root,url.split('?')[0])).size>0,url);
 }
 const manifest=JSON.parse(fs.readFileSync(path.join(root,'model/reference/source-manifest.json'),'utf8'));
 for(const item of manifest.files){if(item.sha256)assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'model/reference',item.file))).digest('hex'),item.sha256,item.file);}
 const ids=[...d.querySelectorAll('[id]')].map(e=>e.id);assert.equal(new Set(ids).size,ids.length);
 const js=fs.readFileSync(path.join(root,'model/model.js'),'utf8');assert(js.includes("scale:.82"));assert(js.includes("start:'center bottom'"));assert(js.includes("end:'bottom top'"));
 assert(d.querySelector('a[href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html"]'),'source of the diagram is credited');
 dom.window.close();console.log('PASS: original Noomo case structure and stylesheet hashes, local video/fonts/official diagram, no custom 3D, valid asset references and model content');
}
module.exports=check;if(require.main===module)check().catch(e=>{console.error(e);process.exitCode=1;});
