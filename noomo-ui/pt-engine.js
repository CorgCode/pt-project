// Native Three.js exports for independent product pages. The landing keeps its original engine.
if(!window.THREE && document.readyState==="loading") await new Promise(resolve=>document.addEventListener("DOMContentLoaded",resolve,{once:true}));
const T=window.THREE;
if(!T) throw new Error("Локальный 3D-движок не загрузился");
export const k=T.Group,i=T.Mesh,p=T.MeshBasicMaterial,M=T.MeshStandardMaterial,ptPhysical=T.MeshPhysicalMaterial,
t=T.SphereGeometry,B=T.BoxGeometry,m=T.PlaneGeometry,a6=T.BufferGeometry,a7=T.Float32BufferAttribute,
ak=T.Texture,q=T.Vector3,aa=T.Raycaster,J=T.DoubleSide,__tla=Promise.resolve();
