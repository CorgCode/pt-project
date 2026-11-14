// Verify the transferred case template and local media; no synthetic 3D renderer is used.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),crypto=require('node:crypto');
const {JSDOM}=require('jsdom'),root=path.resolve(__dirname,'..');
async function check(){
 const html=fs.readFileSync(path.join(root,'model/index.html'),'utf8'),dom=new JSDOM(html,{runScripts:'outside-only',url:'http://127.0.0.1:8845/model/'}),d=dom.window.document;
 assert.equal(d.documentElement.lang,'ru');assert(d.querySelector('.case-hero .wrapper .info-block .right'),'original Noomo case structure');
 assert.equal(d.querySelectorAll('.model-section').length,11);assert.equal(d.querySelectorAll('canvas,iframe,#model-stage,.chapter').length,0);
 assert(!d.querySelector('script[src*="three"],script[src*="scene.js"],script[src*="pt-page"]'),'rejected 3D implementation is disconnected');
 assert.equal(d.querySelectorAll('video').length,0);assert(!/jellyfish-case|jellyfish-poster/.test(html));assert(d.querySelector('.source-figure img'));assert.equal(d.querySelectorAll('[data-figure]').length,16);assert.equal(d.querySelector('audio').getAttribute('src'),'/audio/BG5.mp3');
 assert(!/FWA|Awards|Wallis|Salesforce/.test(d.body.textContent),'no borrowed client claims or awards');
 for(const element of d.querySelectorAll('script[src],link[href],img[src],audio[src],source[srcset]')){
  const url=element.getAttribute('src')||element.getAttribute('href')||element.getAttribute('srcset');if(url.startsWith('/'))assert(fs.statSync(path.join(root,url.split('?')[0])).size>0,url);
 }
 const manifest=JSON.parse(fs.readFileSync(path.join(root,'model/reference/source-manifest.json'),'utf8'));
 for(const item of manifest.files){if(item.sha256)assert.equal(crypto.createHash('sha256').update(fs.readFileSync(path.join(root,'model/reference',item.file))).digest('hex'),item.sha256,item.file);}
 const ids=[...d.querySelectorAll('[id]')].map(e=>e.id);assert.equal(new Set(ids).size,ids.length);
 const js=fs.readFileSync(path.join(root,'model/model.js'),'utf8');assert(js.includes("scale:.82"));assert(js.includes("start:'center bottom'"));assert(js.includes("end:'bottom top'"));
 assert(d.querySelector('a[href="https://scikit-learn.org/stable/modules/neural_networks_supervised.html"]'),'source of the diagram is credited');
 const evidence=JSON.parse(fs.readFileSync(path.join(root,'model/assets/evidence.json'),'utf8'));
 const scoreboard=fs.readFileSync(path.join(root,'../scoreboard/README.md'));
 assert.equal(evidence.source_sha256,crypto.createHash('sha256').update(scoreboard).digest('hex'));
 assert.deepEqual(evidence.models.map(m=>m.score),[56.5,64.9,70]);assert.deepEqual(evidence.vector_parts,[896,64,64]);assert.equal(evidence.ranking.length,17);assert.equal(evidence.ranking[0].id,'multiday_codebook_plus');assert.equal(evidence.ranking[0].score,70);
 // Browser autoplay rejection and a subsequent user toggle must both be handled.
 const w=dom.window,audio=d.querySelector('audio'),button=d.querySelector('#music-toggle');let paused=true,blocked=true;
 Object.defineProperty(audio,'paused',{get:()=>paused});audio.play=async()=>{if(blocked)throw Error('autoplay blocked');paused=false;audio.dispatchEvent(new w.Event('play'));};audio.pause=()=>{paused=true;audio.dispatchEvent(new w.Event('pause'));};
 w.matchMedia=()=>({matches:false});w.sessionStorage.setItem('pt-model-music','off');w.eval(js);await new Promise(resolve=>setImmediate(resolve));
 assert.equal(audio.volume,.12);assert.equal(button.getAttribute('aria-pressed'),'false');assert.equal(button.dataset.blocked,'true');
 blocked=false;button.click();await new Promise(resolve=>setImmediate(resolve));assert.equal(button.getAttribute('aria-pressed'),'true');
 button.click();assert.equal(paused,true);assert.equal(button.getAttribute('aria-pressed'),'false');
 const motion=new JSDOM(html,{runScripts:'outside-only',url:'http://localhost/model/'}),mw=motion.window,calls=[];
 mw.matchMedia=()=>({matches:false});mw.document.querySelector('audio').play=async()=>{};
 mw.gsap={registerPlugin(){},fromTo(target,from,to){calls.push({target,from,to});},to(){}};mw.ScrollTrigger={refresh(){}};
 mw.eval(js);assert(mw.document.querySelectorAll('h2 .word-inner').length>20);assert(calls.some(c=>c.to.stagger===.055));assert(calls.some(c=>c.to.duration===1.5));
 for(const heading of mw.document.querySelectorAll('h1,h2,h3'))assert.equal(heading.getAttribute('aria-label'),heading.textContent.trim());
 motion.window.close();
 dom.window.close();console.log('PASS: original Noomo case structure and stylesheet hashes, local music/fonts/diagrams, real research data and autoplay fallback, no custom 3D, valid asset references and model content');
}
module.exports=check;if(require.main===module)check().catch(e=>{console.error(e);process.exitCode=1;});
