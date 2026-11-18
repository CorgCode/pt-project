/* Hallmark · macrostructure: spatial-product-tour · theme: violet-glass · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V4 · slop test: geometry checks required; browser review pending
 */
import {k as Group, i as Mesh, p as MeshBasicMaterial, M as MeshStandardMaterial, ptPhysical as MeshPhysicalMaterial, t as SphereGeometry, B as BoxGeometry, m as PlaneGeometry, a6 as BufferGeometry, a7 as Float32BufferAttribute, ak as Texture, q as Vector3, aa as Raycaster, J as DoubleSide, __tla as libraryReady} from "/_nuxt/iNjJ8EkR.js";
await libraryReady;
const colors = {ink:0x20182f,accent:0x8c65b3,bright:0xf7f0ff,glass:0xd7b8ec,dark:0x563575,line:0xb999d5,metal:0xded5e9,signal:0x9571c8};
const drawColors = {ink:"#20182f",accent:"#755298",muted:"#665b74",line:"#c8b5db",white:"#faf5ff",backgroundStart:"#f9f3ff",backgroundMiddle:"#eae0f4",backgroundEnd:"#d5bce8"};
let world, camera, renderer, root, model, textMesh, textTexture, backdrop, plaque, pageKey = "", phase=0, yaw=0, pitch=0, targetYaw=0, targetPitch=0, drag=null, transition=1;
let view={w:1,h:1,textW:1,textH:1}, lastLab=null, labDirty=false, selectedPart=-1,inspection=null;
const hitTargets=[], animated=[], particlePaths=[], graphEdges=[];
const graphNodes=new Map(), nodePositions=new Map();
let zoom=1, composition=null, networkAge=0;
const ray = new Raycaster(), pointer={x:0,y:0};
const sphereGeometry=new SphereGeometry(1,24,16), boxGeometry=new BoxGeometry(1,1,1);
const assets = new Set();
const clamp=(n,a,b)=>Math.min(b,Math.max(a,n));
function material(kind="solid",color=colors.accent) {
  const m=kind==="glass"?new MeshPhysicalMaterial({color,roughness:.13,metalness:.04,transmission:.78,thickness:.28,ior:1.42,clearcoat:1,clearcoatRoughness:.12,transparent:true,opacity:.93,depthTest:true,depthWrite:false}):new MeshStandardMaterial({color,roughness:.24,metalness:.2,emissive:color,emissiveIntensity:.07,depthTest:true,depthWrite:true});
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
  const guided=!!page.guided;
  const canvas=document.createElement("canvas");canvas.width=1200;canvas.height=Math.round(1200*view.textH/view.textW);
  const ctx=canvas.getContext("2d");if(!ctx)return;
  const mobile=innerWidth<760,screenW=mobile?innerWidth-36:innerWidth*(guided?.36:.43),scale=1200/screenW;
  const pad=25*scale,width=1200-pad*2;
  let heading=(mobile?26:guided?42:39)*scale,body=(mobile?15:guided?19:17)*scale;
  const topSpace=(mobile?42:55)*scale,gap=(mobile?14:25)*scale,footerSpace=(mobile?12:28)*scale;
  let titleLines,paragraphLines,totalHeight,noteLines=[];
  for(let pass=0;pass<12;pass++){
    ctx.font=`600 ${heading}px PTDisplay, sans-serif`;titleLines=wrap(ctx,page.title,width);
    ctx.font=`400 ${body}px PTOnest, sans-serif`;paragraphLines=page.paragraphs.map(p=>wrap(ctx,p,width));
    ctx.font=`400 ${11*scale}px PTOnest, sans-serif`;noteLines=page.note?wrap(ctx,page.note,width):[];
    totalHeight=noteLines.length*16*scale+topSpace+titleLines.length*heading*1.13+gap+paragraphLines.reduce((s,ls)=>s+ls.length*body*1.65+14*scale,0)+footerSpace;
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
  if(noteLines.length){ctx.font=`400 ${11*scale}px PTOnest, sans-serif`;ctx.fillStyle=drawColors.accent;noteLines.forEach(line=>{ctx.fillText(line,pad,y);y+=16*scale;});}
  if(textTexture){textTexture.dispose();assets.delete(textTexture);}
  textTexture=textureFrom(canvas);textMesh.material.map=textTexture;textMesh.material.needsUpdate=true;
  world.userData.ptTextLayout={height:y,maxHeight:canvas.height,font:body/scale,lines:paragraphLines.flat().length};
}
function disposeModel() {
  for(const a of assets){if(a===textTexture||a===textMesh.material||a===plaque.material||a===backdrop.material||a===backdrop.material.map)continue;a.dispose?.();assets.delete(a);}
  model.clear();hitTargets.length=0;animated.length=0;particlePaths.length=0;graphEdges.length=0;graphNodes.clear();
}
// Rounded geometry is generated in-place; these are actual meshes, not device images.
function roundedPart(parent,x,y,z,w,h,d,r,mat) {
  r=Math.min(r,w/2,h/2,d/2)*.99;
  const geometry=new BufferGeometry(),vertices=[],normals=[],indices=[],steps=5;
  const faces=[[0,1,2,1],[0,1,2,-1],[1,2,0,1],[1,2,0,-1],[2,0,1,1],[2,0,1,-1]],half=[w/2,h/2,d/2];
  for(const [axis,u,v,sign] of faces){const start=vertices.length/3;
    for(let iy=0;iy<=steps;iy++)for(let ix=0;ix<=steps;ix++){
      const p=[0,0,0];p[axis]=half[axis]*sign;p[u]=(ix/steps*2-1)*half[u];p[v]=(iy/steps*2-1)*half[v];
      const inner=p.map((n,i)=>clamp(n,-half[i]+r,half[i]-r)),n=p.map((a,i)=>a-inner[i]),length=Math.hypot(...n)||1;
      for(let i=0;i<3;i++){n[i]/=length;vertices.push(inner[i]+n[i]*r);normals.push(n[i]);}
    }
    for(let iy=0;iy<steps;iy++)for(let ix=0;ix<steps;ix++){const a=start+iy*(steps+1)+ix,b=a+steps+1;const tri=sign>0?[a,a+1,b,b,a+1,b+1]:[a,b,a+1,b,b+1,a+1];indices.push(...tri);}
  }
  geometry.setAttribute("position",new Float32BufferAttribute(vertices,3));geometry.setAttribute("normal",new Float32BufferAttribute(normals,3));geometry.setIndex(indices);geometry.computeBoundingSphere();assets.add(geometry);
  const mesh=new Mesh(geometry,mat);mesh.position.set(x,y,z);mesh.renderOrder=410;parent.add(mesh);return mesh;
}
function laptop(parent,x,y,z,scale=1) {
  const group=new Group();group.name="PT_Device_Laptop";group.position.set(x,y,z);group.scale.setScalar(scale);parent.add(group);
  const metal=material("solid",colors.metal),dark=material("solid",colors.dark),light=material("solid",colors.glass);
  roundedPart(group,0,-.34,.15,1.05,.065,.68,.025,metal);
  roundedPart(group,0,-.297,.22,.26,.008,.18,.003,dark);
  // Keyboard is a recessed surface with rows, not a fake application window.
  for(let row=0;row<4;row++)for(let col=0;col<10;col++)roundedPart(group,(col-4.5)*.082,-.297,-.02+row*.06,.064,.012,.043,.005,dark);
  const lid=new Group();lid.position.set(0,-.3,-.17);lid.rotation.x=-.12;group.add(lid);
  roundedPart(lid,0,.35,0,1.05,.7,.045,.025,metal);
  roundedPart(lid,0,.35,.027,.94,.58,.012,.006,dark);
  const screen=roundedPart(lid,0,.35,.038,.84,.46,.008,.003,light);
  const pulse=basic(colors.accent,.75);
  for(let i=0;i<7;i++)box(lid,(i-3)*.09,.30,.046,.025,.05+(i%3)*.06,.005,pulse);
  sphere(lid,0,.66,.027,.008,metal);
  for(let i=0;i<2;i++)roundedPart(group,-.527,-.337,.03+i*.14,.009,.022,.09,.004,dark);
  group.rotation.y=-.27;return group;
}
function serverDevice(parent,x,y,z,scale=1) {
  const group=new Group();group.name="PT_Device_Server";group.position.set(x,y,z);group.scale.setScalar(scale);parent.add(group);
  const metal=material("solid",colors.metal),dark=material("solid",colors.dark),glass=material("glass");
  roundedPart(group,0,0,0,.55,1.02,.48,.045,metal);
  for(let i=0;i<3;i++){
    roundedPart(group,0,.29-i*.29,.254,.47,.24,.025,.012,dark);
    roundedPart(group,-.055,.29-i*.29,.272,.29,.11,.018,.005,glass);
    sphere(group,.165,.29-i*.29,.283,.015,basic(colors.signal));
  }
  for(let i=0;i<6;i++)box(group,-.03,.47-i*.025,.25,.29,.008,.008,dark);
  group.rotation.y=-.25;return group;
}
function routerDevice(parent,x,y,z,scale=1) {
  const group=new Group();group.name="PT_Device_Network_Node";group.position.set(x,y,z);group.scale.setScalar(scale);parent.add(group);
  const metal=material("solid",colors.metal),dark=material("solid",colors.dark);
  roundedPart(group,0,0,0,.7,.16,.4,.045,metal);
  for(let i=0;i<5;i++){roundedPart(group,(i-2)*.105,0,.205,.075,.055,.018,.006,dark);sphere(group,(i-2)*.105,.056,.217,.009,basic(colors.signal));}
  for(const x of [-.27,.27]){const antenna=roundedPart(group,x,.22,-.1,.025,.43,.025,.009,dark);antenna.rotation.z=x>0?-.15:.15;}
  group.rotation.y=-.18;return group;
}
function buildDeviceStory(kind) {
  const host=laptop(model,-.05,-.1,.18,kind==="device"?1.9:1.35);
  host.userData.ptAction=()=>{
    inspection={...window.__ptWorldState.page,title:"Хост — устройство в сети",paragraphs:["Здесь выбран один компьютер. Рядом — условные сервисы, с которыми он обменивается данными. Назначение реального хоста мы не определяем по этой иллюстрации."]};
    window.dispatchEvent(new CustomEvent("pt:inspect",{detail:inspection}));labDirty=true;
  };hitTargets.push(host.children[0]);host.children[0].userData.ptAction=host.userData.ptAction;
  label(model,"Выбранный хост",-.05,-1,.3,1.65);
  if(kind==="device")return;
  serverDevice(model,1.02,.39,-.35,.75);routerDevice(model,-1.05,.54,-.22,.85);
  label(model,"Сервис",1.03,.98,.2,.8);label(model,"Сеть",-1.05,1,.2,.8);
  const a=new Vector3(0,.08,.3),ends=[new Vector3(.99,.35,-.12),new Vector3(-1.02,.47,-.04)],wire=basic(colors.line,.55),signal=basic(colors.signal);
  ends.forEach((end,j)=>{
    line(model,a,end,.012,wire);
    for(let i=0;i<4;i++){const packet=box(model,0,0,0,.035,.035,.07,signal);particlePaths.push({mesh:packet,from:j?a:end,to:j?end:a,offset:i/4});}
  });
  if(kind==="features"){
    [["С кем",-.95,-.69],["Сервисы",-.32,-.9],["Объём",.38,-.91],["Время",1,-.7]].forEach(([text,x,y])=>{
      const tile=roundedPart(model,x,y,.4,.5,.19,.06,.035,material("glass"));label(model,text,x,y,.46,.55);
      animated.push({mesh:tile,type:"float",base:y,offset:x});
    });
  }
}
function buildSearch() {
  const state=window.__ptLabState;if(!state)return;
  const found=window.__ptSearchFound;
  laptop(model,-.72,-.2,.1,1.3);label(model,"Хост A",-.72,-.9,.3,1);
  const right=new Group();right.position.set(.8,.08,-.08);model.add(right);
  if(found){
    laptop(right,0,0,0,1.1);label(model,"Ближайший профиль",.8,-.72,.3,1.65);
    label(model,state.near[0].host.id,.8,-.91,.3,1.2);
    const edge=line(model,new Vector3(-.15,.05,.3),new Vector3(.35,.05,.3),.025,basic(colors.signal));
    label(model,"Сходство "+state.near[0].score.toFixed(3),0,.64,.4,1.8);
    const halo=ring(right,.65,.015,material("solid",colors.signal));halo.rotation.x=.5;
    const click=sphere(right,0,.1,.2,.48,basic(colors.accent,0));click.userData.ptAction=()=>window.__ptSelectHost?.(state.near[0].host.id);hitTargets.push(click);
  }else{
    state.near.forEach((n,i)=>{
      const x=(i%2)*.55-.3,y=.55-Math.floor(i/2)*.4;
      const m=roundedPart(right,x,y,0,.28,.25,.14,.04,material("glass"));m.userData.ptAction=()=>window.__ptFindSimilar?.();hitTargets.push(m);
    });
    label(model,"Кандидаты",.8,-.9,.3,1.2);
  }
  label(model,"Одинаковая иконка не означает одинаковую роль",0,-1.2,.4,2.7);
  lastLab=state;
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
  if(window.__ptWorldState?.page.kind==="graph-final" && !window.__ptGraphRevealed && !document.body.dataset.page){
    laptop(model,0,0,0,1.7);label(model,"Один хост. Целая сеть вокруг.",0,-1,.3,2.8);return;
  }
  const state=window.__ptLabState;
  if(!state)return;
  networkAge=0;
  const solid=material("solid"),glass=material("glass"),dark=material("solid",colors.dark),wire=basic(colors.accent,.65),softWire=basic(colors.line,.25);
  const nearIds=new Set(state.near.map(n=>n.host.id));
  const pool=[state.selected,...state.near.map(n=>n.host),...state.dataset.filter(h=>h!==state.selected&&!nearIds.has(h.id))].slice(0,24);
  const points=pool.map(h=>state.projection.find(p=>p.hostId===h.id));
  const minX=Math.min(...points.map(p=>p.x)),maxX=Math.max(...points.map(p=>p.x)),minY=Math.min(...points.map(p=>p.y)),maxY=Math.max(...points.map(p=>p.y));
  const posFor=(p,i)=>new Vector3(((p.x-minX)/(maxX-minX||1)-.5)*1.85,((p.y-minY)/(maxY-minY||1)-.5)*1.35,Math.sin(i*2.4)*.25);
  const origin=posFor(points[0],0);
  pool.forEach((host,i)=>{
    const p=nodePositions.get(host.id)?.clone()||posFor(points[i],i),isNear=nearIds.has(host.id),r=i===0?.11:isNear?.075:.035;
    const node=sphere(model,p.x,p.y,p.z,r,i===0?dark:isNear?glass:solid,()=>window.__ptSelectHost?.(host.id));
    node.userData.hostId=host.id;node.userData.radius=r;node.userData.revealOrder=i;graphNodes.set(host.id,node);
    if(isNear){const edge=line(model,origin,p,.008,wire);graphEdges.push({mesh:edge,from:state.selected.id,to:host.id});}
  });
  // Context edges describe the two closest visible profiles, computed in full vector space.
  const edgeIds=new Set(graphEdges.map(e=>[e.from,e.to].sort().join("\u0000")));
  for(const host of pool){
    const candidates=pool.filter(other=>other!==host).map(other=>({host:other,score:host.v.reduce((sum,value,j)=>sum+value*other.v[j],0)/(host.norm*other.norm)})).sort((a,b)=>b.score-a.score).slice(0,2);
    for(const candidate of candidates){
      const id=[host.id,candidate.host.id].sort().join("\u0000");if(edgeIds.has(id))continue;edgeIds.add(id);
      const a=graphNodes.get(host.id).position,b=graphNodes.get(candidate.host.id).position;
      const edge=line(model,a,b,.004,softWire);graphEdges.push({mesh:edge,from:host.id,to:candidate.host.id});
    }
  }
  const halo=ring(model,.17,.008,basic(colors.accent,.65));halo.position.copy(graphNodes.get(state.selected.id).position);
  label(model,`${state.selected.id} · ${state.dataset.length} хостов`,0,1.05,.4,2.8);
  label(model,"Тяните узлы · выбирайте профили",0,-1.03,.4,2.7);
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
  const guided=!!window.__ptWorldState?.page?.guided;
  const distance=6,h=2*Math.tan((camera.getEffectiveFOV?.()??camera.fov)*Math.PI/360)*distance,w=h*camera.aspect,mobile=innerWidth<760;
  view={w,h,textW:mobile?w*.91:w*(guided?.36:.43),textH:mobile?h*(guided?.47:.50):h*.69};
  backdrop.scale.set(w*2.5,h*2.5,1);backdrop.position.set(0,0,-14);
  const x=mobile?0:-w*(guided?.31:.255),y=mobile?-h*(guided?.18:.145):0;
  plaque.visible=!guided;
  plaque.position.set(x,y,-6.05);plaque.scale.set(view.textW,view.textH,.018);
  textMesh.position.set(x,y,-6);textMesh.scale.set(view.textW,view.textH,1);
  const modelScale=mobile?(guided?Math.min(w*.32,h*.14):h*.085):Math.min(w*(guided?.24:.20),h*(guided?.32:.27));
  model.position.set(mobile?0:w*(guided?.15:.24),mobile?h*(guided?.21:.22):0,-5.9);model.scale.setScalar(modelScale);model.userData.baseScale=modelScale;
  if(document.body.dataset.page){
    plaque.visible=false;textMesh.visible=false;backdrop.visible=false;
    model.position.set(0,0,-5.9);
    model.userData.baseScale=Math.min(w*.29,h*.29);
  }
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
  if(composition!==!!state.page.guided){composition=!!state.page.guided;layout();}
  const key=(state.page.guided?"guided:":"details:")+state.index+":"+innerWidth+":"+innerHeight;
  if(pageKey!==key||labDirty||(state.page.id==="demo"&&lastLab!==window.__ptLabState)){
    const changed=pageKey!==key;if(changed){inspection=null;selectedPart=-1;transition=0;}disposeModel();
    if(changed||inspection)paintPage(inspection||state.page,state.index,state.total);
    const kind=state.page.kind;
    if(["device","traffic","features"].includes(kind))buildDeviceStory(kind);else if(kind==="search")buildSearch();else if(kind==="graph-final")buildLab();else if(kind==="results")buildResults();else if(kind==="training")buildTraining();else if(kind==="demo")buildLab();else if(kind==="architecture"||kind==="history")buildArchitecture(kind);else if(kind==="encoder"||kind==="input")buildEncoder(kind);else buildNetwork(kind);
    pageKey=key;labDirty=false;
  }
  const reduced=matchMedia("(prefers-reduced-motion: reduce)").matches;
  phase+=Math.min(delta||.016,.05)*(reduced?0:1);
  const smoothing=1-Math.exp(-Math.min(delta||.016,.05)*9);
  yaw+=(targetYaw-yaw)*smoothing;pitch+=(targetPitch-pitch)*smoothing;
  transition=reduced?1:Math.min(1,transition+Math.min(delta||.016,.05)*2.4);
  const ease=1-Math.pow(1-transition,3);
  model.rotation.set(pitch,yaw+(1-ease)*.12,0);
  model.scale.setScalar(model.userData.baseScale*zoom);
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
  networkAge+=Math.min(delta||.016,.05);
  for(const node of graphNodes.values()){
    const t=reduced?1:clamp((networkAge-node.userData.revealOrder*.018)/.55,0,1);
    node.scale.setScalar(node.userData.radius*(.3+.7*(1-Math.pow(1-t,3))));
  }
  for(const edge of graphEdges){
    const a=graphNodes.get(edge.from)?.position,b=graphNodes.get(edge.to)?.position;if(!a||!b)continue;
    const direction=new Vector3().subVectors(b,a);edge.mesh.position.copy(a).add(b).multiplyScalar(.5);
    edge.mesh.scale.y=direction.length()*(reduced?1:clamp(networkAge/.65,.05,1));edge.mesh.quaternion.setFromUnitVectors(new Vector3(0,1,0),direction.normalize());
  }
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
function isUI(event){return event.target.closest?.(".pt-world-ui,.pt-scene-source,header,.pt-product-link,a,button,input,select,.pt-site-nav,.page-reading");}
window.addEventListener("pointerdown",event=>{
  if(!root?.visible||isUI(event))return;
  const mobile=innerWidth<760,inModel=document.body.dataset.page ? event.target===renderer.domElement : mobile?event.clientY<innerHeight*.39:event.clientX>innerWidth*.50;
  if(!inModel&&!hits(event)[0]?.object?.userData.hostId)return;
  const hit=hits(event)[0]?.object;
  drag={x:event.clientX,y:event.clientY,yaw:targetYaw,pitch:targetPitch,moved:false,hit,node:window.__ptWorldState?.page.kind==="graph-final"&&hit?.userData.hostId?hit:null};
  if(drag.node){drag.planePoint=drag.node.getWorldPosition(new Vector3());drag.planeNormal=camera.getWorldDirection(new Vector3());}
  event.stopImmediatePropagation();
},true);
window.addEventListener("pointermove",event=>{
  if(!drag)return;
  const dx=event.clientX-drag.x,dy=event.clientY-drag.y;
  drag.moved||=Math.hypot(dx,dy)>6;
  if(drag.node&&drag.moved){
    const bounds=renderer.domElement.getBoundingClientRect();ray.setFromCamera({x:(event.clientX-bounds.left)/bounds.width*2-1,y:-(event.clientY-bounds.top)/bounds.height*2+1},camera);
    const distance=new Vector3().subVectors(drag.planePoint,ray.ray.origin).dot(drag.planeNormal)/ray.ray.direction.dot(drag.planeNormal);
    const point=model.worldToLocal(ray.ray.at(distance,new Vector3()));point.x=clamp(point.x,-1.4,1.4);point.y=clamp(point.y,-1,1);
    drag.node.position.copy(point);nodePositions.set(drag.node.userData.hostId,point.clone());
  }else if(!drag.node){targetYaw=drag.yaw+dx*.006;targetPitch=clamp(drag.pitch+dy*.004,-.55,.55);}
  event.stopImmediatePropagation();
},true);
window.addEventListener("pointerup",event=>{
  if(!drag)return;
  if(!drag.moved)drag.hit?.userData.ptAction?.();
  drag=null;event.stopImmediatePropagation();
},true);
window.addEventListener("pointercancel",()=>{drag=null;});

window.__ptFindSimilar = () => {
  if(window.__ptSearchFound){window.__ptGoGraph?.();return;}
  window.__ptSelectHost?.(window.__ptLabState.selected.id);
  window.__ptSearchFound=true;labDirty=true;
  const button=document.getElementById("pt-find-similar");if(button)button.textContent=window.__ptSearchFound?"Исследовать граф →":"Найти похожего";
};
window.__ptResetView = () => {targetYaw=0;targetPitch=0;zoom=1;nodePositions.clear();labDirty=true;};
window.__ptZoomView = delta => {zoom=clamp(zoom*delta,.7,1.5);};
["mousedown","mouseup","click"].forEach(type=>window.addEventListener(type,event=>{
  if(root?.visible&&!isUI(event))event.stopImmediatePropagation();
},true));
