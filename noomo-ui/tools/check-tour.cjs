const fs=require("node:fs"),path=require("node:path"),assert=require("node:assert/strict");
const {JSDOM}=require("jsdom");
const {createCanvas,GlobalFonts}=require("@napi-rs/canvas");
const THREE=require("../../ui-prototype/vendor/three.min.js");
const base=path.resolve(__dirname,"..");
GlobalFonts.registerFromPath(path.join(base,"fonts/onest-400.ttf"),"PTOnest");
GlobalFonts.registerFromPath(path.join(base,"fonts/pt-display.ttf"),"PTDisplay");
(async()=>{
 const dom=new JSDOM(fs.readFileSync(path.join(base,"index.html"),"utf8"),{runScripts:"outside-only",pretendToBeVisual:true,url:"http://127.0.0.1:8845/"});
 const w=dom.window,d=w.document,callbacks=[];
 let scroll=0,refreshes=0;
 w.innerWidth=1440;w.innerHeight=900;
 w.requestAnimationFrame=fn=>callbacks.push(fn);w.matchMedia=()=>({matches:false});
 w.gsap={core:{globals:()=>({ScrollSmoother:{get:()=>({scrollTop:()=>scroll,scrollTo:y=>scroll=y})},ScrollTrigger:{refresh:()=>refreshes++}})}};
 w.HTMLCanvasElement.prototype.getContext=function(){
   if(!this.__cpu||this.__cpu.width!==this.width||this.__cpu.height!==this.height)this.__cpu=createCanvas(this.width,this.height);
   return this.__cpu.getContext("2d");
 };
 const execute=name=>w.eval(fs.readFileSync(path.join(base,name),"utf8").replace(/^import[^\n]+\n/m,"").replace("await libraryReady;", ""));
 execute("pt-adapter.js");execute("product-data.js");execute("pt-lab.js");execute("pt-content.js");
 assert.equal(d.getElementById("pt-guide"),null,"do not mutate Nuxt before hydration");
 d.getElementById("__nuxt").__vue_app__={};
 w.__ptMountContent();
 const home=d.createElement("div");home.id="home-scene";home.style.opacity=1;d.body.append(home);
 execute("pt-story.js");
 assert(w.__ptPages.length>15);assert.equal(new Set(w.__ptPages.map(p=>p.id)).size,6);
 assert(!d.querySelector("iframe,dialog"));
 assert(d.getElementById("pt-guide").classList.contains("pt-scene-source"));
 assert(d.getElementById("pt-guide").getAttribute("aria-hidden")==="true");
 const scene=new THREE.Scene(),camera=new THREE.PerspectiveCamera(45,1440/900,.1,1000);camera.setFocalLength(50);camera.position.set(0,0,10);scene.add(camera);
 const canvas=d.createElement("canvas");canvas.getBoundingClientRect=()=>({left:0,top:0,width:w.innerWidth,height:w.innerHeight});home.append(canvas);
 w.__testThree=THREE;
 for(const name of ["Group","Mesh","MeshBasicMaterial","MeshStandardMaterial","MeshPhysicalMaterial","SphereGeometry","BoxGeometry","PlaneGeometry","BufferGeometry","Float32BufferAttribute","Texture","Vector3","Raycaster","DoubleSide"])w[name]=THREE[name];
 execute("pt-world.js");
 const frame=()=>{callbacks.shift()();w.__ptWorldFrame({scene,camera,renderer:{domElement:canvas},delta:.016});scene.updateMatrixWorld(true);assert(!w.__ptWorldError,w.__ptWorldError);};
 const visit=index=>{scroll=850+index*850+400;frame();assert.equal(w.__ptWorldState.index,index);assert(w.__ptWorldDebug.root.visible);};
 const ids=new Set();
 for(let index=0;index<w.__ptPages.length;index++){
   visit(index);ids.add(w.__ptWorldState.page.kind);
   assert(scene.userData.ptTextLayout.height<=scene.userData.ptTextLayout.maxHeight,`desktop clipping at page ${index}: ${w.__ptWorldState.page.title}`);
 }
 assert(ids.has("encoder")&&ids.has("training")&&ids.has("demo")&&ids.has("results"));
 const modelAt=w.__ptPages.findIndex(p=>p.kind==="architecture");visit(modelAt);
 assert.equal(w.__ptWorldDebug.hitTargets.length,3);
 w.__ptWorldDebug.hitTargets[2].userData.ptAction();frame();
 assert.match(d.getElementById("pt-world-readable").textContent,/собственных векторов/);
 const demoAt=w.__ptPages.findIndex(p=>p.id==="demo");visit(demoAt);
 assert.equal(w.__ptWorldDebug.hitTargets.length,24);
 assert.equal(d.getElementById("pt-world-host").options.length,300);
 d.querySelector('[data-world-preset="flows"]').click();frame();
 assert.equal(w.__ptLabState.dataset.length,11);
 assert.equal(w.__ptWorldDebug.hitTargets.length,11);
 const node=w.__ptWorldDebug.hitTargets[3],position=new THREE.Vector3();node.getWorldPosition(position);position.project(camera);
 const ray=new THREE.Raycaster();ray.setFromCamera({x:position.x,y:position.y},camera);
 const expected=ray.intersectObjects(w.__ptWorldDebug.hitTargets,false)[0].object.userData.hostId;
 const coords={clientX:(position.x+1)*w.innerWidth/2,clientY:(1-position.y)*w.innerHeight/2,bubbles:true};
 canvas.dispatchEvent(new w.MouseEvent("pointerdown",coords));canvas.dispatchEvent(new w.MouseEvent("pointerup",coords));frame();
 assert.equal(w.__ptLabState.selected.id,expected,"pointer raycast selects the actual 3D host");
 async function upload(text,size=text.length){
   Object.defineProperty(d.getElementById("csv-file"),"files",{configurable:true,value:[{name:"test.csv",size,text:async()=>text}]});
   d.getElementById("csv-file").dispatchEvent(new w.Event("change"));await new Promise(setImmediate);frame();
 }
 await upload("host_id,emb_0,emb_1\na,1,0\nb,1,1\nc,0,1");assert.equal(w.__ptLabState.dataset.length,3);assert.equal(w.__ptWorldDebug.hitTargets.length,3);
 await upload("host_id,emb_0\na,nope\nb,1");assert.equal(w.__ptLabState.dataset.length,3);assert.match(d.getElementById("pt-world-status").textContent,/Не удалось/);
 await upload("",4*1024*1024);assert.match(d.getElementById("pt-world-status").textContent,/слишком большой/);
 d.querySelector('[data-world-preset="project"]').click();frame();
 // Inspect the actual rasterized text surfaces at all required viewport widths.
 for(const [width,height] of [[320,568],[375,812],[414,896],[768,1024],[1440,900]]){
   w.innerWidth=width;w.innerHeight=height;camera.aspect=width/height;camera.updateProjectionMatrix();
   w.dispatchEvent(new w.Event("resize"));await new Promise(resolve=>setTimeout(resolve,220));
   const count=w.__ptPages.length;
   for(let i=0;i<count;i++){
     visit(i);
     const layout=scene.userData.ptTextLayout;
     assert(layout.height<=layout.maxHeight,`${width}px text clipped at ${i} / ${w.__ptWorldState.page.title}: ${JSON.stringify(layout)} ${JSON.stringify(w.__ptWorldState.page.paragraphs)}`);
     assert(Number.isFinite(w.__ptWorldDebug.model.scale.x));
   }
   visit(w.__ptPages.findIndex(p=>p.id==="model"));
   const text=w.__ptWorldDebug.root.children.find(m=>m.renderOrder===500);
   fs.writeFileSync(`/private/tmp/pt-ui-qa/tour-text-${width}.png`,text.material.map.image.__cpu.toBuffer("image/png"));
 }
 scroll=100;frame();assert(!w.__ptWorldDebug.root.visible);assert(d.getElementById("pt-world-ui").hidden);
 assert(fs.readFileSync(path.join(base,"_nuxt/DkR7Kq9H.js"),"utf8").includes("window.__ptWorldFrame?.({scene:m,camera:f,renderer:y,delta:N})"));
 console.log("PASS: native-scene hook, full narrative pagination, 3D model parts, host picking, CSV validation, real text rendering at 320/375/414/768/1440px, no dialogs or iframes");
 w.__ptStatusObserver.disconnect();
 dom.window.close();
})().catch(error=>{console.error(error);process.exitCode=1;});
