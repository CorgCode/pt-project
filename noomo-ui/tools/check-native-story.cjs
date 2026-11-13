// Verify that the landing uses the original component and a fresh compiled module graph.
const fs=require('node:fs'),path=require('node:path'),assert=require('node:assert/strict'),{JSDOM}=require('jsdom');
const root=path.resolve(__dirname,'..'),html=fs.readFileSync(path.join(root,'index.html'),'utf8');
const dom=new JSDOM(html,{runScripts:'outside-only',pretendToBeVisual:true,url:'http://127.0.0.1:8845/'}),w=dom.window,d=w.document;
assert(!d.querySelector('script[src^="/pt-world.js"],script[src^="/pt-story.js"]'),'replacement cards and tour are not booted on the landing');
const imports=JSON.parse(d.querySelector('script[type=importmap]').textContent).imports;
for(const file of fs.readdirSync(path.join(root,'_nuxt')).filter(f=>f.endsWith('.js'))){assert(imports['/_nuxt/'+file]?.includes('?pt-build=native-'),'compiled modules all get fresh URLs');}
assert(d.querySelector('script[type=module][src*="pt-build=native-"]'),'the entry URL uses the same build identity');
w.matchMedia=()=>({matches:false});w.requestAnimationFrame=fn=>fn();
w.eval(fs.readFileSync(path.join(root,'pt-main-content.js'),'utf8'));
const sequences=[],sets=[];let refreshes=0;
const gsap={set:(target,props)=>sets.push({target,props}),core:{globals:()=>({})},timeline:options=>{const seq={options,actions:[],fromTo(target,from,to,at){this.actions.push({target,from,to,at});return this;},to(target,to,at){this.actions.push({target,to,at});return this;}};sequences.push(seq);return seq;}};
const layer=d.querySelector('.scene-texts'),left=layer.querySelector('.left-text'),right=layer.querySelector('.right-text');
w.__ptConfigureOriginalText({gsap,ScrollTrigger:{refresh:()=>refreshes++},layer,left,right,bottom:[...layer.querySelectorAll('.bottom-text')]});
assert.equal(sequences.length,6);assert.equal(layer.querySelectorAll('.pt-product-copy').length,6);
assert.equal(d.querySelectorAll('.pt-main-copy,.pt-world-ui,.pt-story-panel').length,0,'no added text layer or cards');
for(const sequence of sequences){assert.equal(sequence.actions[0].from.yPercent,110);assert.equal(sequence.actions[0].to.yPercent,0);assert.equal(sequence.options.scrollTrigger.scrub,1.2);}
assert.equal(sequences[0].options.scrollTrigger.start,'top+=850 top');
assert.equal(sequences[5].options.scrollTrigger.end,'top+=12850 top');
assert(parseFloat(d.querySelector('.home-page').style.height)>13000);assert.equal(refreshes,1);
const sceneCode=fs.readFileSync(path.join(root,'_nuxt/DkR7Kq9H.js'),'utf8');assert(sceneCode.includes(';lo=40,co='));assert(sceneCode.includes('window.__ptConfigureOriginalText({gsap:V,ScrollTrigger:De'));
for(const file of ['models/Sphere_footer_baked.glb','textures/cross.jpg'])assert(fs.statSync(path.join(root,file)).size>0);
dom.window.close();console.log('PASS: original scene text component, 6 native upward scroll sequences, no added cards, fresh compiled module URLs, finale sphere setup and assets');
