/* Hallmark · macrostructure: glass-model-dissection · theme: amethyst-studio · genre: atmospheric
 * pre-emit critique: P4 H4 E4 S5 R4 V5 · slop test: GPU review pending
 */
export class ModelScene {
 constructor(container){
  this.T=window.THREE;if(!this.T)throw Error('3D-движок не загружен');
  const T=this.T,style=getComputedStyle(document.documentElement),token=name=>style.getPropertyValue(name).trim();
  this.palette={glass:token('--scene-glass'),violet:token('--scene-violet'),deep:token('--scene-deep'),metal:token('--scene-metal'),white:token('--scene-white'),ground:token('--scene-ground'),ink:token('--color-ink')};
  this.font=token('--font-body');this.container=container;this.groups=[];this.resources=new Set();this.time=0;this.chapter=0;this.progress=0;this.dragYaw=0;this.dragPitch=0;this.part=0;this.mobile=innerWidth<761;
  this.renderer=new T.WebGLRenderer({antialias:true,alpha:true,powerPreference:'high-performance'});
  this.renderer.setPixelRatio(Math.min(devicePixelRatio||1,1.7));this.renderer.setClearColor(this.palette.white,0);
  this.renderer.outputColorSpace=T.SRGBColorSpace;this.renderer.toneMapping=T.ACESFilmicToneMapping;this.renderer.toneMappingExposure=1.16;
  this.renderer.domElement.setAttribute('aria-label','Вращаемая объёмная схема профиля устройства');container.append(this.renderer.domElement);
  this.renderer.domElement.addEventListener('webglcontextlost',event=>{event.preventDefault();container.dispatchEvent(new Event('model:unavailable'));});
  this.scene=new T.Scene();this.camera=new T.PerspectiveCamera(34,1,.1,80);this.camera.position.set(0,0,8.4);
  this.scene.add(new T.HemisphereLight(this.palette.white,this.palette.ground,2.2));
  for(const [x,y,z,power] of [[4,5,5,3.2],[-4,1,3,2.1],[1,2,-4,3]]){const light=new T.DirectionalLight(this.palette.white,power);light.position.set(x,y,z);this.scene.add(light);}
  this.root=new T.Group();this.scene.add(this.root);
  this.baseSphere=new T.SphereGeometry(1,24,16);this.resources.add(this.baseSphere);
  this.buildProfile();this.buildData();this.buildEncoder();this.buildHistory();this.buildAdapter();this.buildTraining();this.buildResults();
  this.resize();this.bind();
  if(T.RGBELoader)new T.RGBELoader().load('/hdri/photo_studio_01_1k.hdr',texture=>{
   const pmrem=new T.PMREMGenerator(this.renderer),environment=pmrem.fromEquirectangular(texture);this.scene.environment=environment.texture;
   texture.dispose();pmrem.dispose();this.resources.add(environment);
  },undefined,()=>{});
  document.fonts?.ready.then(()=>this.labels?.forEach(refresh=>refresh()));
 }
 group(name){const g=new this.T.Group();g.name=name;g.userData.materials=[];g.userData.paths=[];g.userData.alpha=0;this.groups.push(g);this.root.add(g);return g;}
 material(group,kind='glass',color=this.palette.glass){
  const T=this.T,m=kind==='glass'?new T.MeshPhysicalMaterial({color,roughness:.16,metalness:.03,transmission:.87,thickness:.32,ior:1.46,clearcoat:1,clearcoatRoughness:.08,iridescence:.45,iridescenceIOR:1.3,iridescenceThicknessRange:[140,380],transparent:true,opacity:1,depthWrite:false,side:T.DoubleSide}):new T.MeshStandardMaterial({color,roughness:.24,metalness:.35,transparent:true,opacity:1});
  group.userData.materials.push(m);this.resources.add(m);return m;
 }
 shape(w,h,r){const s=new this.T.Shape(),x=-w/2,y=-h/2;s.moveTo(x+r,y);s.lineTo(x+w-r,y);s.quadraticCurveTo(x+w,y,x+w,y+r);s.lineTo(x+w,y+h-r);s.quadraticCurveTo(x+w,y+h,x+w-r,y+h);s.lineTo(x+r,y+h);s.quadraticCurveTo(x,y+h,x,y+h-r);s.lineTo(x,y+r);s.quadraticCurveTo(x,y,x+r,y);return s;}
 plate(group,w,h,d,mat,x=0,y=0,z=0){
  const geo=new this.T.ExtrudeGeometry(this.shape(w,h,Math.min(w,h)*.16),{depth:d,bevelEnabled:true,bevelSegments:4,steps:1,bevelSize:.025,bevelThickness:.025,curveSegments:16});geo.translate(0,0,-d/2);this.resources.add(geo);
  const mesh=new this.T.Mesh(geo,mat);mesh.position.set(x,y,z);group.add(mesh);return mesh;
 }
 bead(group,x,y,z,r,mat){const mesh=new this.T.Mesh(this.baseSphere,mat);mesh.position.set(x,y,z);mesh.scale.setScalar(r);group.add(mesh);return mesh;}
 ring(group,r,tube,mat){const geo=new this.T.TorusGeometry(r,tube,12,100);this.resources.add(geo);const mesh=new this.T.Mesh(geo,mat);group.add(mesh);return mesh;}
 wire(group,a,b,mat,r=.008){
  const T=this.T,from=new T.Vector3(...a),to=new T.Vector3(...b),mid=from.clone().lerp(to,.5);mid.z+=.16;
  const curve=new T.CatmullRomCurve3([from,mid,to]),geo=new T.TubeGeometry(curve,24,r,6,false);this.resources.add(geo);const mesh=new T.Mesh(geo,mat);group.add(mesh);return curve;
 }
 flow(group,a,b,mat,count=4){const curve=this.wire(group,a,b,mat,.005);for(let i=0;i<count;i++){const bead=this.bead(group,0,0,0,.025,mat);group.userData.paths.push({curve,mesh:bead,offset:i/count});}}
 label(group,text,x,y,z,width=.9){
  const T=this.T,canvas=document.createElement('canvas');canvas.width=768;canvas.height=160;const texture=new T.CanvasTexture(canvas);texture.colorSpace=T.SRGBColorSpace;
  const refresh=()=>{const ctx=canvas.getContext('2d');if(!ctx)return;ctx.clearRect(0,0,768,160);ctx.fillStyle=this.palette.ink;ctx.font=`600 48px ${this.font}`;ctx.textAlign='center';ctx.textBaseline='middle';ctx.fillText(text,384,80,740);texture.needsUpdate=true;};
  (this.labels??=[]).push(refresh);refresh();this.resources.add(texture);
  const mat=new T.MeshBasicMaterial({map:texture,transparent:true,depthTest:false,depthWrite:false,toneMapped:false,side:T.DoubleSide});group.userData.materials.push(mat);this.resources.add(mat);
  const geo=new T.PlaneGeometry(width,width*160/768);this.resources.add(geo);const mesh=new T.Mesh(geo,mat);mesh.position.set(x,y,z);mesh.renderOrder=10;group.add(mesh);return mesh;
 }
 buildProfile(){
  const g=this.group('profile'),glass=this.material(g),violet=this.material(g,'solid',this.palette.violet),metal=this.material(g,'solid',this.palette.metal);
  const frameShape=this.shape(1.85,2.05,.48),hole=new this.T.Path();const inner=this.shape(1.56,1.76,.36);hole.curves=inner.curves;frameShape.holes.push(hole);
  const frameGeo=new this.T.ExtrudeGeometry(frameShape,{depth:.2,bevelEnabled:true,bevelSegments:5,bevelThickness:.035,bevelSize:.035,curveSegments:24});frameGeo.translate(0,0,-.1);this.resources.add(frameGeo);
  this.housing=new this.T.Mesh(frameGeo,glass);this.housing.rotation.y=.1;g.add(this.housing);
  this.parts=[this.plate(g,.65,1.4,.17,glass,-.43,0,.18),this.plate(g,.4,1.05,.17,this.material(g,'glass',this.palette.violet),.1,0,.1),this.plate(g,.4,1.05,.17,glass,.57,0,0)];
  for(let row=0;row<8;row++)for(let col=0;col<3;col++)this.bead(g,-.58+col*.16,(row-3.5)*.14,.3,.026,col===1?violet:metal);
  this.partLabels=[this.label(g,'896',-.43,-.88,.28,.6),this.label(g,'64',.1,-.88,.28,.43),this.label(g,'64',.57,-.88,.28,.43)];
  const band=this.ring(g,1.34,.025,this.material(g,'glass'));band.rotation.set(.75,.2,.1);this.profileBand=band;
  this.label(g,'1 024 числа',0,1.37,.2,1.8);g.userData.alpha=1;
 }
 buildData(){
  const g=this.group('data'),glass=this.material(g),solid=this.material(g,'solid',this.palette.violet),metal=this.material(g,'solid',this.palette.metal);
  const screen=this.plate(g,1.05,.7,.08,glass,-.5,.15,0);screen.rotation.y=.16;this.plate(g,1.13,.075,.58,metal,-.5,-.28,.12);
  for(let i=0;i<3;i++){const x=1,y=.65-i*.65;this.plate(g,.42,.35,.17,glass,x,y,-.1);this.flow(g,[-.02,.1,.1],[x-.25,y,-.1],solid);}
  this.label(g,'Устройство',-.5,-.63,.3,1.2);this.label(g,'Взаимодействия',.9,1.1,.1,1.5);
  ['С кем','Как','Сколько','Когда'].forEach((text,i)=>{this.plate(g,.52,.25,.07,glass,(i-1.5)*.65,-1.04,.15);this.label(g,text,(i-1.5)*.65,-1.04,.24,.55);});
 }
 buildEncoder(){
  const g=this.group('encoder'),glass=this.material(g),solid=this.material(g,'solid',this.palette.violet),metal=this.material(g,'solid',this.palette.metal);
  for(let col=0;col<3;col++){const x=(col-1)*.77;this.plate(g,.16,1.35,.63,glass,x,0,-col*.1);for(let i=0;i<6;i++)this.bead(g,x,(i-2.5)*.19,.38-col*.1,.042,col===2?solid:metal);if(col<2)for(let i=0;i<4;i++)this.flow(g,[x,(i-1.5)*.2,.3],[x+.77,(1.5-i)*.16,.2],solid,1);}
  this.label(g,'Признаки',-.85,-1,.3,.95);this.label(g,'Кодировщик',0,1.1,.3,1.65);this.label(g,'896 чисел',.85,-1,.3,1.1);
 }
 buildHistory(){
  const g=this.group('history'),glass=this.material(g),solid=this.material(g,'solid',this.palette.violet);
  this.historySheets=[];for(let i=0;i<5;i++){const plate=this.plate(g,1.05,1.4,.08,glass,(i-2)*.24,0,(i-2)*-.23);plate.rotation.y=-.12;this.historySheets.push(plate);}
  ['03','05','08','10','12'].forEach((text,i)=>this.label(g,text,(i-2)*.28,-1.1,.6,.4));
  const halo=this.ring(g,1.28,.02,solid);halo.rotation.set(1.05,0,-.1);this.label(g,'История устройства',0,1.27,.4,2);
 }
 buildAdapter(){
  const g=this.group('adapter'),glass=this.material(g),solid=this.material(g,'solid',this.palette.violet);
  this.plate(g,.55,.68,.12,glass,-1.05,0,.12);
  this.plate(g,.22,1.2,.55,glass,0,0,0);this.plate(g,.6,.9,.2,glass,1.05,0,-.2);
  this.flow(g,[-.74,0,.15],[-.15,0,.15],solid);this.flow(g,[.18,0,0],[.7,0,-.1],solid);
  for(let i=0;i<20;i++){const x=-.35+(i%5)*.14,y=-.9+(Math.floor(i/5)-1.5)*.1;this.bead(g,x,y,.15,.025,solid);}
  this.flow(g,[.35,-.9,.15],[1,-.35,-.1],solid,3);
  this.label(g,'Свежее окно',-1.05,-.57,.3,1.1);this.label(g,'Ridge',0,.95,.3,.8);this.label(g,'Профиль',1.05,-.85,.3,1.1);this.label(g,'20 соседей',-.07,-1.4,.3,1.1);
 }
 buildTraining(){
  const g=this.group('training'),glass=this.material(g),solid=this.material(g,'solid',this.palette.violet);
  const days=['02–12','13','14–16'],names=['Обучение','Настройки','Проверка'];for(let i=0;i<3;i++){const x=(i-1)*.9;this.plate(g,.65,.95,.2,glass,x,0,-i*.12);this.label(g,days[i],x,.1,.32,.7);this.label(g,names[i],x,-.8,.3,.95);if(i<2)this.flow(g,[x+.35,0,.1],[x+.55,0,0],solid,2);}
 }
 buildResults(){
  const g=this.group('results'),glass=this.material(g),solid=this.material(g,'solid',this.palette.violet);this.bars=[];
  [56.5,64.9,70].forEach((value,i)=>{const height=value/70*1.6,x=(i-1)*.75;const bar=this.plate(g,.44,height,.28,i===2?solid:glass,x,-.8+height/2,0);this.bars.push({mesh:bar,height});this.label(g,String(value).replace('.',','),x,height-.6,.4,.7);this.label(g,['Кодировщик','История','Plus'][i],x,-1.15,.4,1);});
 }
 setChapter(index,progress,right){this.chapter=index;this.progress=Math.max(0,Math.min(1,progress));this.right=right;}
 selectPart(index){this.part=index;}
 reset(){this.dragYaw=0;this.dragPitch=0;}
 resize(){this.mobile=innerWidth<761;this.renderer.setSize(innerWidth,innerHeight);this.camera.aspect=innerWidth/innerHeight;this.camera.updateProjectionMatrix();}
 bind(){
  const canvas=this.renderer.domElement;let drag=null;
  canvas.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,yaw:this.dragYaw,pitch:this.dragPitch,horizontal:false};});
  canvas.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(!drag.horizontal&&Math.abs(dx)>8&&Math.abs(dx)>Math.abs(dy)){drag.horizontal=true;canvas.setPointerCapture?.(e.pointerId);}if(!drag.horizontal)return;this.dragYaw=drag.yaw+dx*.006;this.dragPitch=Math.max(-.5,Math.min(.5,drag.pitch+dy*.004));});
  const release=()=>{drag=null;};canvas.addEventListener('pointerup',release);canvas.addEventListener('pointercancel',release);
 }
 update(dt,reduced=false){
  const speed=1-Math.exp(-Math.min(dt,.05)*5),T=this.T;this.time+=reduced?0:dt;
  const names=['profile','data','encoder','history','profile','adapter','training','results','profile'],active=names[this.chapter]||'profile';
  const worldWidth=2*Math.tan(this.camera.fov*Math.PI/360)*8.4*this.camera.aspect;
  const desktopScale=Math.min(1.1,innerWidth/1400),side=Math.max(0,Math.min(1.12,worldWidth*.24,worldWidth/2-desktopScale*1.5-.08));
  const targetX=this.mobile?0:this.right?-side:side,targetY=this.mobile?1.28:0;
  this.root.position.x+=(targetX-this.root.position.x)*speed;this.root.position.y+=(targetY-this.root.position.y)*speed;
  const scale=this.mobile?Math.min(.67,innerWidth/520):Math.min(1.1,innerWidth/1400);
  this.root.scale.setScalar(scale);
  const rotation=-.35+this.chapter*.1+(reduced?0:this.progress*.16)+this.dragYaw;
  this.root.rotation.y+=(rotation-this.root.rotation.y)*speed;this.root.rotation.x+=(this.dragPitch+.08-this.root.rotation.x)*speed;
  for(const g of this.groups){const target=g.name===active?1:0;g.userData.alpha+=(target-g.userData.alpha)*speed;g.visible=g.userData.alpha>.008;g.scale.setScalar(.92+.08*g.userData.alpha);for(const mat of g.userData.materials)mat.opacity=g.userData.alpha;for(const path of g.userData.paths)path.mesh.position.copy(path.curve.getPoint((this.time*.16+path.offset)%1));}
  const exploded=this.chapter===4?1:0;
  this.parts.forEach((part,i)=>{const x=[-.43,.1,.57][i]+(i-1)*exploded*.38;part.position.x+=(x-part.position.x)*speed;const targetZ=exploded?(i===this.part?.48:0):[.18,.1,0][i];part.position.z+=(targetZ-part.position.z)*speed;this.partLabels[i].position.x=part.position.x;});
  this.renderer.render(this.scene,this.camera);
 }
 dispose(){for(const resource of this.resources)resource.dispose?.();this.renderer.dispose();this.renderer.domElement.remove();}
}
