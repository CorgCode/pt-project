// Verify that the landing uses the original component and a fresh compiled module graph.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const dom=new JSDOM(html,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://127.0.0.1:8845/'}),w=dom.window,d=w.document;
assert(!d.querySelector('script[src^="/pt-world.js"],script[src^="/pt-story.js"]'),'replacement cards and tour are not booted on the landing');
const imports=JSON.parse(d.querySelector('script[type=importmap]').textContent).imports;
for(const file of fs.readdirSync(path.join(root,'_nuxt')).filter(f=>f.endsWith('.js'))){assert(imports['/_nuxt/'+file]?.includes('?pt-build=native-'),'compiled modules all get fresh URLs');}
assert(d.querySelector('script[type=module][src*="pt-build=native-"]'),'the entry URL uses the same build identity');
w.matchMedia=()=>({matches:false});
global.window=w;global.document=d;global.getComputedStyle=w.getComputedStyle.bind(w);
const realGsap=require("gsap").gsap;
w.eval(fs.readFileSync(path.join(root,'pt-main-content.js'),'utf8'));
let options;
const gsap={set:realGsap.set,core:{globals:()=>({})},timeline:config=>{options=config;return realGsap.timeline({paused:true,onUpdate:config.onUpdate});}};
const layer=d.querySelector('.scene-texts'),left=layer.querySelector('.left-text'),right=layer.querySelector('.right-text');
w.__ptConfigureOriginalText({gsap,ScrollTrigger:{refresh(){}},layer,left,right,bottom:[...layer.querySelectorAll('.bottom-text')]});
const {blocks,sequence}=w.__ptNativeCopy;
assert.equal(blocks.length,6);assert.equal(layer.querySelectorAll('.pt-product-copy').length,6);
assert.equal(d.querySelectorAll('.pt-main-copy,.pt-world-ui,.pt-story-panel').length,0,'no added text layer or cards');
assert.equal(options.scrollTrigger.scrub,1.2);
assert.equal(options.scrollTrigger.start,'top+=850 top');
assert.equal(options.scrollTrigger.end,'top+=12850 top');
assert.equal(sequence.duration(),6,'one shared timeline gives every chapter its own interval');
function visibleAt(time){sequence.time(time,false);w.__ptNativeCopy.syncVisibility();const visible=blocks.filter(b=>!b.hidden);assert(visible.length<=1,`overlap at ${time}`);return visible;}
for(let i=0;i<=600;i++){const time=i/100,visible=visibleAt(time);if(time>0)assert.equal(visible[0],blocks[Math.min(5,Math.floor(time))]);}
for(let i=600;i>=0;i--)visibleAt(i/100);
for(const time of [5.8,.2,3.99,4.01,1.9,.01,0])visibleAt(time);
visibleAt(.05);const firstLine=blocks[0].querySelector('.pt-copy-line'),earlyY=realGsap.getProperty(firstLine,'yPercent');
visibleAt(.35);assert(realGsap.getProperty(firstLine,'yPercent')<earlyY,'actual GSAP transform moves text from bottom to top');
assert(parseFloat(d.querySelector('.home-page').style.height)>13000);
sequence.kill();realGsap.ticker.sleep();
const sceneCode=fs.readFileSync(path.join(root,'_nuxt/DkR7Kq9H.js'),'utf8');assert(sceneCode.includes(';lo=40,co='));assert(sceneCode.includes('window.__ptConfigureOriginalText({gsap:V,ScrollTrigger:De'));
for(const file of ['models/Sphere_footer_baked.glb','textures/cross.jpg'])assert(fs.statSync(path.join(root,file)).size>0);
dom.window.close();console.log('PASS: actual GSAP playback forward/backward/jumps, at most one visible chapter across 1200 positions, upward transforms, original scene and fresh module URLs');
