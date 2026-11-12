// CPU geometry preview for asset QA; this is not a browser/WebGL screenshot.
const fs=require('node:fs');
const {createCanvas}=require('@napi-rs/canvas');
module.exports=function(model,camera,THREE,file){
 const canvas=createCanvas(1000,800),ctx=canvas.getContext('2d'),faces=[];
 ctx.fillStyle='#e9e0f1';ctx.fillRect(0,0,1000,800);
 model.updateWorldMatrix(true,true);camera.updateMatrixWorld();
 const light=new THREE.Vector3(-.5,.8,1).normalize();
 model.traverse(mesh=>{
  if(!mesh.isMesh||!mesh.material||mesh.material.map||mesh.material.opacity===0)return;
  const geo=mesh.geometry,pos=geo.attributes.position,index=geo.index;
  for(let t=0;t<(index?index.count:pos.count);t+=3){
   const a=new THREE.Vector3().fromBufferAttribute(pos,index?index.getX(t):t).applyMatrix4(mesh.matrixWorld),b=new THREE.Vector3().fromBufferAttribute(pos,index?index.getX(t+1):t+1).applyMatrix4(mesh.matrixWorld),c=new THREE.Vector3().fromBufferAttribute(pos,index?index.getX(t+2):t+2).applyMatrix4(mesh.matrixWorld);
   const normal=new THREE.Vector3().subVectors(b,a).cross(new THREE.Vector3().subVectors(c,a)).normalize();
   if(normal.dot(new THREE.Vector3().subVectors(camera.position,a))<0)continue;
   const points=[a,b,c].map(p=>p.clone().project(camera));
   if(points.some(p=>!Number.isFinite(p.x)||Math.abs(p.x)>3||Math.abs(p.y)>3))continue;
   const shade=.68+.32*Math.max(0,normal.dot(light)),color=mesh.material.color.clone().multiplyScalar(shade);
   faces.push({points,z:points.reduce((s,p)=>s+p.z,0)/3,color:'#'+color.getHexString(),alpha:mesh.material.transmission?.35:mesh.material.opacity});
  }
 });
 faces.sort((a,b)=>b.z-a.z);
 for(const face of faces){ctx.globalAlpha=face.alpha;ctx.fillStyle=face.color;ctx.beginPath();face.points.forEach((p,i)=>{const x=(p.x-.02)*1200+360,y=(1-p.y)*400;(i?ctx.lineTo:ctx.moveTo).call(ctx,x,y);});ctx.closePath();ctx.fill();}
 fs.writeFileSync(file,canvas.toBuffer('image/png'));
};
