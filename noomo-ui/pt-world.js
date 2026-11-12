/* Hallmark · macrostructure: spatial-product-tour · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: geometry checks required; browser review pending
 */
import {k as Group, i as Mesh, p as MeshBasicMaterial, M as MeshStandardMaterial, ptPhysical as MeshPhysicalMaterial, t as SphereGeometry, B as BoxGeometry, m as PlaneGeometry, a6 as BufferGeometry, a7 as Float32BufferAttribute, ak as Texture, q as Vector3, aa as Raycaster, J as DoubleSide, __tla as libraryReady} from "/_nuxt/iNjJ8EkR.js";
await libraryReady;
const colors = {ink:0x20182f,accent:0x8c65b3,bright:0xf7f0ff,glass:0xd7b8ec,dark:0x563575,line:0xb999d5};
const drawColors = {ink:"#20182f",accent:"#755298",muted:"#665b74",line:"#c8b5db",white:"#faf5ff",backgroundStart:"#f9f3ff",backgroundMiddle:"#eae0f4",backgroundEnd:"#d5bce8"};
let world, camera, renderer, root, model, textMesh, textTexture, backdrop, plaque, pageKey = "", phase=0, yaw=0, pitch=0, targetYaw=0, targetPitch=0, drag=null, transition=1;
let view={w:1,h:1,textW:1,textH:1}, lastLab=null, labDirty=false, selectedPart=-1,inspection=null;
const hitTargets=[], animated=[], particlePaths=[];
const ray = new Raycaster(), pointer={x:0,y:0};
const sphereGeometry=new SphereGeometry(1,24,16), boxGeometry=new BoxGeometry(1,1,1);
const assets = new Set();
const clamp=(n,a,b)=>Math.min(b,Math.max(a,n));
function material(kind="solid",color=colors.accent) {
  const m=kind==="glass"?new MeshPhysicalMaterial({color,roughness:.13,metalness:.04,transmission:.78,thickness:.28,ior:1.42,clearcoat:1,clearcoatRoughness:.12,transparent:true,opacity:.93,depthTest:false,depthWrite:false}):new MeshStandardMaterial({color,roughness:.24,metalness:.2,emissive:color,emissiveIntensity:.15,depthTest:false,depthWrite:false});
  assets.add(m); return m;
}
function basic(color,opacity=1) {const m=new MeshBasicMaterial({color,opacity,transparent:opacity<1,depthTest:false,depthWrite:false,toneMapped:false,side:DoubleSide});assets.add(m);return m;}
function box(parent,x,y,z,w,h,d,mat) {const m=new Mesh(boxGeometry,mat);m.position.set(x,y,z);m.scale.set(w,h,d);m.renderOrder=410;parent.add(m);return m;}
function sphere(parent,x,y,z,r,mat,action) {const m=new Mesh(sphereGeometry,mat);m.position.set(x,y,z);m.scale.setScalar(r);m.renderOrder=420;parent.add(m);if(action){m.userData.ptAction=action;hitTargets.push(m);}return m;}
function line(parent,a,b,r,mat) {
  // A narrow prism keeps every connection in actual 3D without another renderer.
  const direction=new Vector3().subVectors(b,a),m=box(parent,0,0,0,r,direction.length(),r,mat);
  m.position.copy(a).add(b).multiplyScalar(.5);m.quaternion.setFromUnitVectors(new Vector3(0,1,0),direction.normalize());return m;
}
function ring(parent,radius,tube,mat) {
  const geometry=new BufferGeometry(),position=[],normal=[],index=[],rs=72,ts=8;
  for(let i=0;i<=rs;i++)for(let j=0;j<=ts;j++){
    const u=i/rs*Math.PI*2,v=j/ts*Math.PI*2,cv=Math.cos(v),sv=Math.sin(v),cu=Math.cos(u),su=Math.sin(u);
    position.push((radius+tube*cv)*cu,(radius+tube*cv)*su,tube*sv);normal.push(cv*cu,cv*su,sv);
  }
  for(let i=0;i<rs;i++)for(let j=0;j<ts;j++){const a=i*(ts+1)+j,b=a+ts+1;index.push(a,b,a+1,b,b+1,a+1);}
  geometry.setAttribute("position",new Float32BufferAttribute(position,3));geometry.setAttribute("normal",new Float32BufferAttribute(normal,3));geometry.setIndex(index);geometry.computeBoundingSphere();assets.add(geometry);
  const m=new Mesh(geometry,mat);m.renderOrder=415;parent.add(m);return m;
}
function textureFrom(canvas) {const t=new Texture(canvas);t.colorSpace="srgb";t.needsUpdate=true;assets.add(t);return t;}
function label(parent,text,x,y,z,width=.8) {
  const canvas=document.createElement("canvas");canvas.width=768;canvas.height=160;
  const ctx=canvas.getContext("2d");if(!ctx)return;
  ctx.fillStyle=drawColors.ink;ctx.font="600 50px PTOnest, sans-serif";ctx.textAlign="center";ctx.textBaseline="middle";ctx.fillText(text,384,80,740);
  const mat=new MeshBasicMaterial({map:textureFrom(canvas),transparent:true,depthTest:false,depthWrite:false,toneMapped:false,side:DoubleSide});assets.add(mat);
  const plane=new Mesh(new PlaneGeometry(width,width*160/768),mat);assets.add(plane.geometry);plane.position.set(x,y,z);plane.renderOrder=480;parent.add(plane);return plane;
}
function wrap(ctx,text,width) {
  const lines=[];let line="";
  for(const word of text.split(/\s+/)){const next=line?line+" "+word:word;if(ctx.measureText(next).width>width&&line){lines.push(line);line=word;}else line=next;}
  if(line)lines.push(line);return lines;
}
function paintPage(page,index,total) {
  const canvas=document.createElement("canvas");canvas.width=1200;canvas.height=Math.round(1200*view.textH/view.textW);
  const ctx=canvas.getContext("2d");if(!ctx)return;
  const mobile=innerWidth<760,screenW=mobile?innerWidth-36:innerWidth*.43,scale=1200/screenW;
  const pad=25*scale,width=1200-pad*2;
  let heading=(mobile?26:39)*scale,body=(mobile?15:17)*scale;
  const topSpace=(mobile?42:55)*scale,gap=(mobile?14:25)*scale,footerSpace=(mobile?12:28)*scale;
  let titleLines,paragraphLines,totalHeight;
  for(let pass=0;pass<12;pass++){
    ctx.font=`600 ${heading}px PTDisplay, sans-serif`;titleLines=wrap(ctx,page.title,width);
    ctx.font=`400 ${body}px PTOnest, sans-serif`;paragraphLines=page.paragraphs.map(p=>wrap(ctx,p,width));
    totalHeight=topSpace+titleLines.length*heading*1.13+gap+paragraphLines.reduce((s,ls)=>s+ls.length*body*1.65+14*scale,0)+footerSpace;
    if(totalHeight<canvas.height-pad||body<=13*scale)break;
    body*=.96;heading*=.96;
  }
  ctx.fillStyle=drawColors.accent;ctx.font=`400 ${10*scale}px PTOnest, sans-serif`;
  const names={problem:"ЗАДАЧА",product:"ПРОДУКТ",model:"УСТРОЙСТВО МОДЕЛИ",training:"ОБУЧЕНИЕ",results:"РЕЗУЛЬТАТЫ",demo:"ДЕМОНСТРАЦИЯ"};
  ctx.fillText(`${names[page.id]} · ${index+1} / ${total}`,pad,(mobile?22:32)*scale);
  let y=topSpace;ctx.fillStyle=drawColors.ink;ctx.font=`600 ${heading}px PTDisplay, sans-serif`;ctx.textBaseline="top";
  titleLines.forEach(line=>{ctx.fillText(line,pad,y);y+=heading*1.13;});y+=gap;
  ctx.font=`400 ${body}px PTOnest, sans-serif`;ctx.fillStyle=drawColors.muted;
  paragraphLines.forEach(lines=>{lines.forEach(line=>{ctx.fillText(line,pad,y);y+=body*1.65;});y+=14*scale;});
  if(textTexture){textTexture.dispose();assets.delete(textTexture);}
  textTexture=textureFrom(canvas);textMesh.material.map=textTexture;textMesh.material.needsUpdate=true;
  world.userData.ptTextLayout={height:y,maxHeight:canvas.height,font:body/scale,lines:paragraphLines.flat().length};
}
function disposeModel() {
  for(const a of assets){if(a===textTexture||a===textMesh.material||a===plaque.material||a===backdrop.material||a===backdrop.material.map)continue;a.dispose?.();assets.delete(a);}
  model.clear();hitTargets.length=0;animated.length=0;particlePaths.length=0;
}
function buildNetwork(kind) {
  const glass=material("glass"),solid=material("solid"),dark=material("solid",colors.dark),wire=basic(colors.line,.65);
  const center=sphere(model,0,0,0,.28,glass);animated.push({mesh:center,type:"float",base:0});
  const count=kind==="product"?6:8;
  for(let i=0;i<count;i++){
    const angle=i/count*Math.PI*2,pos=new Vector3(Math.cos(angle)*.78,Math.sin(angle)*.65,Math.sin(angle*2)*.25);
    sphere(model,pos.x,pos.y,pos.z,.075,i%3?solid:dark);
    line(model,new Vector3(),pos,.009,wire);
    const dot=sphere(model,0,0,0,.018,solid);particlePaths.push({mesh:dot,from:pos,to:new Vector3(),offset:i/count});
  }
  const r=ring(model,.44,.009,basic(colors.accent,.55));r.rotation.x=.7;
  label(model,"Профиль хоста",0,-1.02,.12,1.8);
}
function buildEncoder(kind) {
  const glass=material("glass"),solid=material("solid"),dark=material("solid",colors.dark),wire=basic(colors.line,.6);
  for(let layer=0;layer<3;layer++){
    const x=(layer-1)*.6,z=(layer-1)*-.15;
    const plate=box(model,x,0,z,.1,1.1,.58,glass);plate.rotation.y=-.2;
    for(let i=0;i<5;i++){
      const y=(i-2)*.18;
      sphere(model,x,y,z+.33,.035,layer===2?dark:solid);
      if(layer<2)line(model,new Vector3(x,y,z+.33),new Vector3(x+.6,(2-i)*.18,z+.18),.006,wire);
    }
  }
  for(let i=0;i<8;i++){
    const dot=sphere(model,0,0,.42,.018,solid);particlePaths.push({mesh:dot,from:new Vector3(-.85,(i%5-2)*.17,.4),to:new Vector3(.85,(2-i%5)*.17,.1),offset:i/8});
  }
  label(model,kind==="input"?"Трафик → признаки":"Признаки → профиль",0,-.91,.4,2);
  label(model,"Схема, не точное число слоёв",0,-1.1,.4,2.2);
}
function buildArchitecture(kind) {
  const glass=material("glass"),solid=material("solid"),light=material("solid",colors.glass);
  const parts=[{x:-.64,h:1.35,w:.65,t:"896 · история"},{x:.13,h:.65,w:.4,t:"64 · контакты"},{x:.7,h:.65,w:.4,t:"64 · граф"}];
  parts.forEach((p,i)=>{
    const mesh=box(model,p.x,0,0,p.w,p.h,.3,i===selectedPart?solid:i===0?glass:light);mesh.rotation.y=-.14;mesh.userData.ptAction=()=>{
      selectedPart=i;document.querySelector(`[data-model-part="${["history","contacts","graph"][i]}"]`)?.click();
      inspection={...window.__ptWorldState.page,title:document.getElementById("pt-model-part-title").textContent,paragraphs:[document.getElementById("pt-model-part-copy").textContent]};
      window.dispatchEvent(new CustomEvent("pt:inspect",{detail:inspection}));
      labDirty=true;
    };hitTargets.push(mesh);animated.push({mesh,type:"float",base:0,offset:i});
    label(model,p.t,p.x,-.87,.3,i===0?.95:.9);
  });
  const band=ring(model,1.07,.012,solid);band.scale.y=.35;band.rotation.x=.4;
  label(model,"1 024 числа · выберите часть",0,1.04,.2,2.4);
  if(selectedPart>=0){
    const text=document.getElementById("pt-model-part-title")?.textContent;
    if(text)label(model,text,0,-1.12,.4,2.8);
  }
}
function buildTraining() {
  const glass=material("glass"),solid=material("solid"),lineMat=basic(colors.line,.6);
  const xs=[-.75,0,.75];
  xs.forEach((x,i)=>{
    const plate=box(model,x,0,-i*.08,.55,.8,.16,glass);plate.rotation.y=-.16;
    label(model,["02–12","13","14–16"][i],x,.11,.2,.65);
    label(model,["Обучение","Подбор","Проверка"][i],x,-.64,.2,.82);
    if(i<2)line(model,new Vector3(x+.28,0,.1),new Vector3(x+.47,0,.1),.012,lineMat);
  });
  for(let i=0;i<5;i++){const dot=sphere(model,0,0,.24,.024,solid);particlePaths.push({mesh:dot,from:new Vector3(-.9,-.85,.24),to:new Vector3(.9,-.85,.24),offset:i/5});}
  label(model,"Сначала прошлое. Потом будущее.",0,1,.2,2.5);
}
function buildResults() {
  const solid=material("solid"),glass=material("glass"),lineMat=basic(colors.line,.7);
  const values=[56.5,64.9,70],names=["Кодировщик","История","Три части"];
  values.forEach((value,i)=>{
    const h=value/70*1.2,x=(i-1)*.65,mesh=box(model,x,-.6+h/2,0,.36,h,.28,i===2?solid:glass);
    label(model,String(value).replace(".",","),x,-.6+h+.18,.2,.7);
    label(model,names[i],x,-.86,.2,.9);
    animated.push({mesh,type:"bar",height:h,base:-.6});
  });
  line(model,new Vector3(-1,-.62,.1),new Vector3(1,-.62,.1),.008,lineMat);
  label(model,"Сравнение моделей · баллы из 100",0,1.05,.2,2.6);
}
function buildLab() {
  const state=window.__ptLabState;
  if(!state)return;
  const solid=material("solid"),glass=material("glass"),dark=material("solid",colors.dark),wire=basic(colors.line,.65);
  const nearIds=new Set(state.near.map(n=>n.host.id));
  const pool=[state.selected,...state.near.map(n=>n.host),...state.dataset.filter(h=>h!==state.selected&&!nearIds.has(h.id))].slice(0,24);
  const points=pool.map(h=>state.projection.find(p=>p.hostId===h.id));
  const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
  const posFor=(p,i)=>new Vector3(((p.x-minX)/(maxX-minX||1)-.5)*1.85,((p.y-minY)/(maxY-minY||1)-.5)*1.35,Math.sin(i*2.4)*.25);
  const origin=posFor(points[0],0);
  pool.forEach((host,i)=>{
    const p=posFor(points[i],i),isNear=nearIds.has(host.id),r=i===0?.11:isNear?.075:.035;
    const node=sphere(model,p.x,p.y,p.z,r,i===0?dark:isNear?glass:solid,()=>window.__ptSelectHost?.(host.id));
    node.userData.hostId=host.id;
    if(isNear)line(model,origin,p,.006,wire);
  });
  label(model,`${state.selected.id} · ${state.dataset.length} хостов`,0,1.05,.4,2.8);
  label(model,"Выберите точку · показано до 24 профилей",0,-1.03,.4,2.7);
  // Five nearest scores are real, computed in the full vector space.
  state.near.forEach((n,i)=>{
    const x=-.9+i*.43,h=(n.score+1)/2*.3;
    box(model,x,-1.4+h/2,.3,.24,h,.12,i===0?dark:solid);
    label(model,n.score.toFixed(3),x,-1.51,.5,.43);
  });
  label(model,"Расположение условное; сходство по всем признакам",0,-1.76,.4,2.9);
  lastLab=state;
}
function layout() {
  if(!camera||!root)return;
  const distance=6,h=2*Math.tan((camera.getEffectiveFOV?.()??camera.fov)*Math.PI/360)*distance,w=h*camera.aspect,mobile=innerWidth<760;
  view={w,h,textW:mobile?w*.91:w*.43,textH:mobile?h*.50:h*.69};
  backdrop.scale.set(w*1.12,h*1.12,1);backdrop.position.set(0,0,-6.4);
  const x=mobile?0:-w*.255,y=mobile?-h*.145:0;
  plaque.position.set(x,y,-6.05);plaque.scale.set(view.textW,view.textH,.018);
  textMesh.position.set(x,y,-6);textMesh.scale.set(view.textW,view.textH,1);
  const modelScale=mobile?h*.085:Math.min(w*.20,h*.27);
  model.position.set(mobile?0:w*.24,mobile?h*.22:0,-5.9);model.scale.setScalar(modelScale);
  pageKey="";
}
function setup(scene,cam,rend) {
  camera=cam;renderer=rend;world=scene;
  root=new Group();root.name="PT_Product_Spatial_Tour";scene.add(root);root.visible=false;
  const backgroundCanvas=document.createElement("canvas");backgroundCanvas.width=512;backgroundCanvas.height=512;
  const backgroundContext=backgroundCanvas.getContext("2d");
  if(backgroundContext){const gradient=backgroundContext.createLinearGradient(0,0,512,512);gradient.addColorStop(0,drawColors.backgroundStart);gradient.addColorStop(.55,drawColors.backgroundMiddle);gradient.addColorStop(1,drawColors.backgroundEnd);backgroundContext.fillStyle=gradient;backgroundContext.fillRect(0,0,512,512);}
  const backgroundMaterial=new MeshBasicMaterial({map:textureFrom(backgroundCanvas),color:colors.bright,depthTest:true,depthWrite:true,toneMapped:false});assets.add(backgroundMaterial);
  backdrop=new Mesh(new PlaneGeometry(1,1),backgroundMaterial);backdrop.renderOrder=390;root.add(backdrop);
  plaque=box(root,0,0,-6.05,1,1,.018,new MeshPhysicalMaterial({color:colors.bright,transmission:.2,roughness:.2,transparent:true,opacity:.8,depthTest:false,depthWrite:false,clearcoat:1}));assets.add(plaque.material);plaque.renderOrder=400;
  textMesh=new Mesh(new PlaneGeometry(1,1),new MeshBasicMaterial({transparent:true,depthTest:false,depthWrite:false,toneMapped:false,side:DoubleSide}));textMesh.renderOrder=500;assets.add(textMesh.material);root.add(textMesh);
  model=new Group();model.name="PT_Interactive_Diagram";root.add(model);
  layout();
  document.fonts?.ready.then(()=>{pageKey="";});
  window.__ptWorldResize=layout;
  window.__ptWorldDebug={root,model,hitTargets,view:()=>view};
}
function drawWorld({scene,camera:cam,renderer:rend,delta}) {
  if(!root)setup(scene,cam,rend);
  const state=window.__ptWorldState;
  root.visible=!!state?.visible;
  if(!root.visible)return;
  camera.updateMatrixWorld();root.position.copy(camera.getWorldPosition(new Vector3()));root.quaternion.copy(camera.getWorldQuaternion(root.quaternion));
  const key=state.index+":"+innerWidth+":"+innerHeight;
  if(pageKey!==key||labDirty||(state.page.id==="demo"&&lastLab!==window.__ptLabState)){
    const changed=pageKey!==key;if(changed){inspection=null;selectedPart=-1;transition=0;}disposeModel();
    if(changed||inspection)paintPage(inspection||state.page,state.index,state.total);
    const kind=state.page.kind;
    if(kind==="results")buildResults();else if(kind==="training")buildTraining();else if(kind==="demo")buildLab();else if(kind==="architecture"||kind==="history")buildArchitecture(kind);else if(kind==="encoder"||kind==="input")buildEncoder(kind);else buildNetwork(kind);
    pageKey=key;labDirty=false;
  }
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  phase+=Math.min(delta||.016,.05)*(reduced?0:1);
  const smoothing=1-Math.exp(-Math.min(delta||.016,.05)*9);
  yaw+=(targetYaw-yaw)*smoothing;pitch+=(targetPitch-pitch)*smoothing;
  transition=reduced?1:Math.min(1,transition+Math.min(delta||.016,.05)*2.4);
  const ease=1-Math.pow(1-transition,3);
  model.rotation.set(pitch,yaw+(1-ease)*.12,0);
  textMesh.material.opacity=ease;
  animated.forEach((a,i)=>{
    if(a.type==="float")a.mesh.position.y=a.base+(reduced?0:Math.sin(phase*.7+(a.offset||i))*.025);
    if(a.type==="bar"){
      const t=reduced?1:clamp(state.progress*5,.08,1);
      a.mesh.scale.y=a.height*t;a.mesh.position.y=a.base+a.height*t/2;
    }
  });
  particlePaths.forEach(p=>p.mesh.position.lerpVectors(p.from,p.to,reduced?.5:(phase*.18+p.offset)%1));
  const enter=reduced?1:clamp(state.progress/.12,0,1);
  textMesh.position.z=-6-(1-enter)*.12;
  world.userData.ptActivePage=state.index;
};
let failed=false;
window.__ptWorldFrame = function (context) {
  if(failed)return;
  try {drawWorld(context);}
  catch(error) {
    failed=true;if(root)root.visible=false;
    window.__ptWorldError=error.message;
    const readable=document.getElementById("pt-world-readable");if(readable)readable.hidden=false;
    console.error("PT spatial tour:",error);
  }
};
window.addEventListener("pt:lab-update",()=>{labDirty=true;});
function hits(event) {
  if(!root?.visible||!renderer)return [];
  const bounds=renderer.domElement.getBoundingClientRect();
  pointer.x=(event.clientX-bounds.left)/bounds.width*2-1;pointer.y=-(event.clientY-bounds.top)/bounds.height*2+1;
  ray.setFromCamera(pointer,camera);return ray.intersectObjects(hitTargets,false);
}
function isUI(event){return event.target.closest?.(".pt-world-ui,.pt-scene-source,header,.pt-product-link");}
window.addEventListener("pointerdown",event=>{
  if(!root?.visible||isUI(event))return;
  const mobile=innerWidth<760,inModel=mobile?event.clientY<innerHeight*.39:event.clientX>innerWidth*.50;
  if(!inModel)return;
  drag={x:event.clientX,y:event.clientY,yaw:targetYaw,pitch:targetPitch,moved:false,hit:hits(event)[0]?.object};
  event.stopImmediatePropagation();
},true);
window.addEventListener("pointermove",event=>{
  if(!drag)return;
  const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
  drag.moved||=Math.hypot(dx,dy)>6;targetYaw=drag.yaw+dx*.006;targetPitch=clamp(drag.pitch+dy*.004,-.55,.55);
  event.stopImmediatePropagation();
},true);
window.addEventListener("pointerup",event=>{
  if(!drag)return;
  if(!drag.moved)drag.hit?.userData.ptAction?.();
  drag=null;event.stopImmediatePropagation();
},true);
window.addEventListener("pointercancel",()=>{drag=null;});

window.__ptResetView = () => {targetYaw=0;targetPitch=0;};
["mousedown","mouseup","click"].forEach(type=>window.addEventListener(type,event=>{
  if(root?.visible&&!isUI(event))event.stopImmediatePropagation();
},true));
