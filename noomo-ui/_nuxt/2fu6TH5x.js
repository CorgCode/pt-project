import{M as Je,P as Ke,V as y,b as Ce,c as j,d as Te,L as C,W as De,D as Qe,e as Ye,U as Ze,f as B,u as Ue,g as et,R as tt,G as at,T as R,h as F,i as k,S as rt,j as it,N as ot,k as $,O as st,E as J,H as nt,l as Be,m as lt,n as K,B as dt,p as Q,q as mt,C as ut,r as ct,s as Re,t as Fe,v as Ee,w as We,_ as ht,x as pt,y as ft,z as vt,__tla as xt}from"./iNjJ8EkR.js";import{K as wt,E as gt,a as Pt,B as L,b as _t,R as bt,T as St,H as Mt,c as yt,d as V,S as Ct,e as Tt,__tla as Dt}from"./evonP4q8.js";import{G as Ut}from"./DNkUmkFf.js";import{a as Bt,E as Rt,j as Ft,H as T,F as D,L as Et,J as Wt,M as o,W as kt,N as ke,Q as Le,P as q,u as G}from"./CkGjpDBV.js";let Ie,Oe,Lt=Promise.all([(()=>{try{return xt}catch{}})(),(()=>{try{return Dt}catch{}})()]).then(async()=>{class He extends Je{constructor(e,s,d,a,{mixBlur:r=0,mixStrength:g=1,resolution:U=256,blur:E=[0,0],minDepthThreshold:c=.9,maxDepthThreshold:f=1,depthScale:W=0,depthToBlurRatioBias:_=.25,mirror:m=0,distortion:I=1,mixContrast:O=1,distortionMap:v,reflectorOffset:b=0,bufferSamples:S=8,planeNormal:w=new y(0,0,1)}={}){super(),this.gl=e,this.camera=s,this.scene=d,this.parent=a,this.hasBlur=E[0]+E[1]>0,this.reflectorPlane=new Ke,this.normal=new y,this.reflectorWorldPosition=new y,this.cameraWorldPosition=new y,this.rotationMatrix=new Ce,this.lookAtPosition=new y(0,-1,0),this.clipPlane=new j,this.view=new y,this.target=new y,this.q=new j,this.textureMatrix=new Ce,this.virtualCamera=new Te,this.reflectorOffset=b,this.planeNormal=w,this.setupBuffers(U,E,S),this.reflectorProps={mirror:m,textureMatrix:this.textureMatrix,mixBlur:r,tDiffuse:this.fbo1.texture,tDepth:this.fbo1.depthTexture,tDiffuseBlur:this.fbo2.texture,hasBlur:this.hasBlur,mixStrength:g,minDepthThreshold:c,maxDepthThreshold:f,depthScale:W,depthToBlurRatioBias:_,distortion:I,distortionMap:v,mixContrast:O,"defines-USE_BLUR":this.hasBlur?"":void 0,"defines-USE_DEPTH":W>0?"":void 0,"defines-USE_DISTORTION":v?"":void 0}}setupBuffers(e,s,d){const a={minFilter:C,magFilter:C,colorSpace:this.gl.outputColorSpace},r=new De(e,e,a);r.depthBuffer=!0,r.depthTexture=new Qe(e,e),r.format=Ye,r.type=Ze;const g=new De(e,e,a);this.gl.capabilities.isWebGL2&&(r.samples=d),this.fbo1=r,this.fbo2=g,this.kawaseBlurPass=new wt,this.kawaseBlurPass.setSize(s[0],s[1])}beforeRender(){if(!this.parent||(this.reflectorWorldPosition.setFromMatrixPosition(this.parent.matrixWorld),this.cameraWorldPosition.setFromMatrixPosition(this.camera.matrixWorld),this.rotationMatrix.extractRotation(this.parent.matrixWorld),this.normal.copy(this.planeNormal),this.normal.applyMatrix4(this.rotationMatrix),this.reflectorWorldPosition.addScaledVector(this.normal,this.reflectorOffset),this.view.subVectors(this.reflectorWorldPosition,this.cameraWorldPosition),this.view.dot(this.normal)>0))return;this.view.reflect(this.normal).negate(),this.view.add(this.reflectorWorldPosition),this.rotationMatrix.extractRotation(this.camera.matrixWorld),this.lookAtPosition.set(0,0,-1),this.lookAtPosition.applyMatrix4(this.rotationMatrix),this.lookAtPosition.add(this.cameraWorldPosition),this.target.subVectors(this.reflectorWorldPosition,this.lookAtPosition),this.target.reflect(this.normal).negate(),this.target.add(this.reflectorWorldPosition),this.virtualCamera.position.copy(this.view),this.virtualCamera.up.set(0,1,0),this.virtualCamera.up.applyMatrix4(this.rotationMatrix),this.virtualCamera.up.reflect(this.normal),this.virtualCamera.lookAt(this.target),this.virtualCamera.far=this.camera.far,this.virtualCamera.updateMatrixWorld(),this.virtualCamera.projectionMatrix.copy(this.camera.projectionMatrix),this.textureMatrix.set(.5,0,0,.5,0,.5,0,.5,0,0,.5,.5,0,0,0,1),this.textureMatrix.multiply(this.virtualCamera.projectionMatrix),this.textureMatrix.multiply(this.virtualCamera.matrixWorldInverse),this.textureMatrix.multiply(this.parent.matrixWorld),this.reflectorPlane.setFromNormalAndCoplanarPoint(this.normal,this.reflectorWorldPosition),this.reflectorPlane.applyMatrix4(this.virtualCamera.matrixWorldInverse),this.clipPlane.set(this.reflectorPlane.normal.x,this.reflectorPlane.normal.y,this.reflectorPlane.normal.z,this.reflectorPlane.constant);const e=this.virtualCamera.projectionMatrix;this.q.x=(Math.sign(this.clipPlane.x)+e.elements[8])/e.elements[0],this.q.y=(Math.sign(this.clipPlane.y)+e.elements[9])/e.elements[5],this.q.z=-1,this.q.w=(1+e.elements[10])/e.elements[14],this.clipPlane.multiplyScalar(2/this.clipPlane.dot(this.q)),e.elements[2]=this.clipPlane.x,e.elements[6]=this.clipPlane.y,e.elements[10]=this.clipPlane.z+1,e.elements[14]=this.clipPlane.w}update(){if(this.parent.material!==this)return;this.parent.visible=!1;const e=this.gl.xr.enabled,s=this.gl.shadowMap.autoUpdate;this.beforeRender(),this.gl.xr.enabled=!1,this.gl.shadowMap.autoUpdate=!1,this.gl.setRenderTarget(this.fbo1),this.gl.state.buffers.depth.setMask(!0),this.gl.autoClear||this.gl.clear(),this.gl.render(this.scene,this.virtualCamera),this.hasBlur&&this.kawaseBlurPass.render(this.gl,this.fbo1,this.fbo2),this.gl.xr.enabled=e,this.gl.shadowMap.autoUpdate=s,this.parent.visible=!0,this.gl.setRenderTarget(null)}onBeforeCompile(e,...s){super.onBeforeCompile(e,...s),this.defines===void 0&&(this.defines={}),this.defines.USE_UV||(this.defines.USE_UV=""),this.reflectorProps["defines-USE_BLUR"]!==void 0&&(this.defines.USE_BLUR=""),this.reflectorProps["defines-USE_DEPTH"]!==void 0&&(this.defines.USE_DEPTH=""),this.reflectorProps["defines-USE_DISTORTION"]!==void 0&&(this.defines.USE_DISTORTION="");let d=this.reflectorProps;for(let a in d)e.uniforms[a]={get value(){return d[a]}};e.vertexShader=`
            uniform mat4 textureMatrix;
            varying vec4 my_vUv;     
          ${e.vertexShader}`,e.vertexShader=e.vertexShader.replace("#include <project_vertex>",`
          #include <project_vertex>
          my_vUv = textureMatrix * vec4( position, 1.0 );
          gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
          `),e.fragmentShader=`
            uniform sampler2D tDiffuse;
            uniform sampler2D tDiffuseBlur;
            uniform sampler2D tDepth;
            uniform sampler2D distortionMap;
            uniform float distortion;
            uniform float cameraNear;
            uniform float cameraFar;
            uniform bool hasBlur;
            uniform float mixBlur;
            uniform float mirror;
            uniform float mixStrength;
            uniform float minDepthThreshold;
            uniform float maxDepthThreshold;
            uniform float mixContrast;
            uniform float depthScale;
            uniform float depthToBlurRatioBias;
            varying vec4 my_vUv;        
            ${e.fragmentShader}`,e.fragmentShader=e.fragmentShader.replace("#include <emissivemap_fragment>",`
          #include <emissivemap_fragment>
        
          float distortionFactor = 0.0;
          #ifdef USE_DISTORTION
            distortionFactor = texture2D(distortionMap, vUv).r * distortion;
          #endif
    
          vec4 new_vUv = my_vUv;
          new_vUv.x += distortionFactor;
          new_vUv.y += distortionFactor;
    
          vec4 base = texture2DProj(tDiffuse, new_vUv);
          vec4 blur = texture2DProj(tDiffuseBlur, new_vUv);
          
          vec4 merge = base;
          
          #ifdef USE_NORMALMAP
            vec2 normal_uv = vec2(0.0);
            vec4 normalColor = texture2D(normalMap, vUv);
            vec3 my_normal = normalize( vec3( normalColor.r * 2.0 - 1.0, normalColor.b,  normalColor.g * 2.0 - 1.0 ) );
            vec3 coord = new_vUv.xyz / new_vUv.w;
            normal_uv = coord.xy + coord.z * my_normal.xz * 0.05 * normalScale;
            vec4 base_normal = texture2D(tDiffuse, normal_uv);
            vec4 blur_normal = texture2D(tDiffuseBlur, normal_uv);
            merge = base_normal;
            blur = blur_normal;
          #endif
    
          float depthFactor = 0.0001;
          float blurFactor = 0.0;
    
          #ifdef USE_DEPTH
            vec4 depth = texture2DProj(tDepth, new_vUv);
            depthFactor = smoothstep(minDepthThreshold, maxDepthThreshold, 1.0-(depth.r * depth.a));
            depthFactor *= depthScale;
            depthFactor = max(0.0001, min(1.0, depthFactor));
    
            #ifdef USE_BLUR
              blur = blur * min(1.0, depthFactor + depthToBlurRatioBias);
              merge = merge * min(1.0, depthFactor + 0.5);
            #else
              merge = merge * depthFactor;
            #endif
      
          #endif
    
          float reflectorRoughnessFactor = roughness;
          #ifdef USE_ROUGHNESSMAP
            vec4 reflectorTexelRoughness = texture2D( roughnessMap, vUv );
            
            reflectorRoughnessFactor *= reflectorTexelRoughness.g;
          #endif
          
          #ifdef USE_BLUR
            blurFactor = min(1.0, mixBlur * reflectorRoughnessFactor);
            merge = mix(merge, blur, blurFactor);
          #endif
    
          vec4 newMerge = vec4(0.0, 0.0, 0.0, 1.0);
          newMerge.r = (merge.r - 0.5) * mixContrast + 0.5;
          newMerge.g = (merge.g - 0.5) * mixContrast + 0.5;
          newMerge.b = (merge.b - 0.5) * mixContrast + 0.5;
          
          diffuseColor.rgb = diffuseColor.rgb * ((1.0 - min(1.0, mirror)) + newMerge.rgb * mixStrength);
          `)}}const Ae={fragmentShader:`
    // original shader by Evan Wallace

    #define MAX_ITERATIONS 100

    uniform float blur;
    uniform float taper;
    uniform vec2 start;
    uniform vec2 end;
    uniform vec2 direction;
    uniform int sampleCount;

    float random(vec3 scale, float seed) {
        /* use the fragment position for a different seed per-pixel */
        return fract(sin(dot(gl_FragCoord.xyz + seed, scale)) * 43758.5453 + seed);
    }

    void mainImage(const in vec4 inputColor, const in vec2 uv, out vec4 outputColor) {

        vec4 color = vec4(0.0);
        float total = 0.0;
        vec2 startPixel = vec2(start.x * resolution.x, start.y * resolution.y);
        vec2 endPixel = vec2(end.x * resolution.x, end.y * resolution.y);

        // use screen diagonal to normalize blur radii
        float maxScreenDistance = distance(vec2(0.0), resolution); // diagonal distance

        float gradientRadius = taper * (maxScreenDistance);
        float blurRadius = blur * (maxScreenDistance / 16.0);

        /* randomize the lookup values to hide the fixed number of samples */
        float offset = random(vec3(12.9898, 78.233, 151.7182), 0.0);

        vec2 normal = normalize(vec2(startPixel.y - endPixel.y, endPixel.x - startPixel.x));
        float radius = smoothstep(0.0, 1.0, abs(dot(uv * resolution - startPixel, normal)) / gradientRadius) * blurRadius;

        float f_samples = float(sampleCount);
        float half_samples = f_samples / 2.0;

        #pragma unroll_loop_start
        for (int i = 0; i <= MAX_ITERATIONS; i++) {
            
            if (i >= sampleCount) { break; } // return early if over sample count

            float f_i = float(i);
            float s_i = -half_samples + f_i;

            float percent = (s_i + offset - 0.5) / half_samples;
            float weight = 1.0 - abs(percent);

            vec4 sample_i = texture2D(inputBuffer, uv + normalize(direction) / resolution * percent * radius);

            /* switch to pre-multiplied alpha to correctly blur transparent images */
            sample_i.rgb *= sample_i.a;

            color += sample_i * weight;
            total += weight;
        }
        #pragma unroll_loop_end

        outputColor = color / total;

        /* switch back from pre-multiplied alpha */
        outputColor.rgb /= outputColor.a + 0.00001;
    }
    `};class Ne extends gt{constructor({blendFunction:e=L.Normal,blur:s=1,taper:d=.5,start:a=[.01,.01],end:r=[1,1],sampleCount:g=40,direction:U=[1,1]}={}){super("TiltShiftEffect",Ae.fragmentShader,{blendFunction:e,attributes:Pt.CONVOLUTION,uniforms:new Map([["blur",new B(s)],["taper",new B(d)],["start",new B(a)],["end",new B(r)],["sampleCount",new B(g)],["direction",new B(U)]])})}}let Y,Z,ee,te,ae,re,ie,oe,se,ne,le,de,me,ue,ce,he,pe,fe,ve,xe,we,ge,Pe,_e,be;Y={id:"cases-scene"},Z=["src"],ee={__name:"caseScene",props:{video:{type:Object,default:{video:"",poster:""}}},setup(h){const e=h,s=Bt(null);let d=`
varying vec2 vUv;
void main() {
  vUv = uv;
  gl_Position = projectionMatrix * modelViewMatrix * vec4( position, 1.0 );
}
`,a=`
varying vec2 vUv;

uniform float dispFactor;
uniform float dpr;
uniform sampler2D disp;

uniform sampler2D texture1;
uniform sampler2D texture2;
uniform float angle1;
uniform float angle2;
uniform float intensity1;
uniform float intensity2;
uniform vec4 res;
uniform vec2 parent;

mat2 getRotM(float angle) {
  float s = sin(angle);
  float c = cos(angle);
  return mat2(c, -s, s, c);
}

void main() {

  vec2 uv = vUv;
  vec2 myUV = (uv - vec2(0.5))*res.zw + vec2(0.5);


  vec2 distortedPosition1 = myUV;
  vec4 _texture1 = texture2D(texture1, distortedPosition1);
  gl_FragColor = _texture1;
}
`;const r=Ue(),{isMobile:g,isDesktop:U}=et(),E=new tt;new at;let c,f,W,_,m,I,O,v,b,S,w;new R;let M,H,P,A,x,ze=F.context(()=>{});Rt(()=>{je()}),Ft(()=>{window.removeEventListener("mousemove",Se),window.removeEventListener("resize",Me),ze.revert(),M.destroy(),c.traverse(t=>{t instanceof k&&(t.geometry.dispose(),t.material.dispose())}),F.ticker.remove(ye)});const je=()=>{M=new Ut,W=document.getElementById("cases-scene"),H=M.addFolder("Tilt Shift"),P=M.addFolder("Tone Mapping"),A=M.addFolder("Hue Saturation"),x=M.addFolder("Reflector"),M.hide(),c=new rt,f=new it({powerPreference:"high-performance",antialias:!1,stencil:!1,depth:!0,alpha:!0}),f.setPixelRatio(window.devicePixelRatio>1?1.5:1),f.toneMapping=ot,f.setSize(window.innerWidth,window.innerHeight),W.appendChild(f.domElement),m=new Te(U?45:65,window.innerWidth/window.innerHeight,.1,1e3),m.position.set(0,0,8);const t=new $;t.add(m),I=F.quickTo(t.rotation,"x",{duration:.4,ease:"power1.out"}),O=F.quickTo(t.rotation,"y",{duration:.4,ease:"power1.out"}),_=new st(m,f.domElement),_.enableDamping=!0,_.enabled=!0,_.target.set(0,0,0),Ve(),window.addEventListener("mousemove",Se,!1),U&&window.addEventListener("resize",Me,!1)},Se=t=>{O((t.clientX/window.innerWidth*2-1)/30),I((t.clientY/window.innerHeight*2-1)/30)},Ve=async()=>{const t=await E.loadAsync("/hdri/photo_studio_01_1k.hdr");t.mapping=J,qe()},qe=()=>{w=new _t(f,{frameBufferType:nt,multisampling:4}),w.addPass(new bt(c,m));const t=new Ne({blur:g?.1:.8,taper:g?1:.85,start:[.5,0],end:[.5,1],sampleCount:75}),i=new St({blendFunction:L.NORMAL,mode:3,exposure:2.1,whitePoint:2,middleGrey:1.6,minLuminance:.001,maxLuminance:4}),l=new Mt({hue:0,saturation:.5,brightness:.59,opacity:1,blendFunction:L.ALPHA});H.add(t.uniforms.get("blur"),"value",0,1,.01).name("Blur"),H.add(t.uniforms.get("taper"),"value",0,1,.01).name("Taper"),H.add(t.uniforms.get("sampleCount"),"value",0,100,1).name("Sample Count"),P.add(i,"mode",yt).name("mode"),P.add(i,"whitePoint",0,10,.01).name("White Point"),P.add(i,"middleGrey",0,10,.01).name("Middle Grey"),P.add(i,"averageLuminance",0,10,.01).name("averageLuminance"),P.add(i.adaptiveLuminanceMaterial,"minLuminance",0,10,.01).name("min Luminance"),P.add(i.uniforms.get("maxLuminance"),"value",0,10,.01).name("Max Luminance"),P.add(i.blendMode,"blendFunction",L).name("blend mode"),A.add(l.blendMode,"blendFunction",L).name("Hue bland mode"),A.add(l,"hue",0,1,.01).name("Hue"),A.add(l,"saturation",0,1,.01).name("Saturation"),w.addPass(new V(m,t)),w.addPass(new V(m,i)),w.addPass(new V(m,l)),w.addPass(new V(m,new Ct({preset:Tt.HIGH}))),Ge()},Ge=async()=>{const t=await new R().loadAsync(e.video.poster);let i=new Be({uniforms:{texture1:{type:"t",value:t},res:{type:"vec4",value:new j(1600/2,900/2,1,1)}},vertexShader:d,fragmentShader:a});const l=new lt(16/8*3,9/8*3,1);v=new k(l,i),b=v.clone(),S=v.clone();let u=new K(s.value);u.magFilter=C;let p=new Be({uniforms:{texture1:{type:"t",value:u},res:{type:"vec4",value:new j(1600/2,900/2,1,1)}},vertexShader:d,fragmentShader:a});s.value.addEventListener("loadeddata",()=>{s.value.play(),v.material=p,b.material=p,S.material=p,u=new K(s.value),u.magFilter=C,u.minFilter=C,p.uniforms.texture1.value=u},!1),s.value.play(),v.material=p,b.material=p,S.material=p,u=new K(s.value),u.magFilter=C,u.minFilter=C,p.uniforms.texture1.value=u;const X=window.innerWidth>767?.65:1.4,N=new $,z=new $;N.position.set(16/8*3/2,0,-1.5),N.rotation.set(0,2*Math.PI*(-25/360),0),z.position.set(-(16/8*3)/2,0,-1.5),z.rotation.set(0,2*Math.PI*(25/360),0),v.position.set(0,X,-1.5),b.position.set(16/8*3/2,X,0),S.position.set(-(16/8*3)/2,X,0),c.add(v),N.add(b),z.add(S),c.add(N),c.add(z),await Xe()};let n;const Xe=async()=>{const t=new dt(60,60,.1,40);n=new k(t,new Q),n.material=new He(f,m,c,n,{resolution:1024,blur:[256,128],mixBlur:8,mixContrast:.62,mirror:.61,mixStrength:3.49,minDepthThreshold:.56,maxDepthThreshold:.7,depthScale:.66,depthToBlurRatioBias:5,reflectorOffset:0}),n.material.setValues({roughnessMap:new R().load("/textures/reflect/roughness.jpg"),normalMap:new R().load("/textures/reflect/normal.png"),normalScale:new mt(.3,.3)}),n.material.color=new ut(.7176470588235294,.8156862745098039,.996078431372549),x.add(n.material.reflectorProps,"mixBlur",0,30,1).name("Mix Blur"),x.add(n.material.reflectorProps,"mixContrast",0,2,.01).name("mixContrast"),x.add(n.material.reflectorProps,"mirror",0,1,.01).name("mirror"),x.add(n.material.reflectorProps,"mixStrength",0,10,.01).name("mixStrength"),x.add(n.material.reflectorProps,"minDepthThreshold",0,1,.01).name("minDepthThreshold"),x.add(n.material.reflectorProps,"maxDepthThreshold",0,1,.01).name("maxDepthThreshold"),x.add(n.material.reflectorProps,"depthScale",0,10,.01).name("depthScale"),x.add(n.material.reflectorProps,"depthToBlurRatioBias",0,10,.01).name("depthToBlurRatioBias"),x.addColor(n.material,"color").name("color");const i=window.innerWidth>767?-1.09:-.34;n.rotateX(-Math.PI/2),n.position.set(0,i,0),c.add(n);const l=new ct(16777215,1);l.position.set(2,4,1),c.add(l),F.ticker.fps(60),F.ticker.add(ye),$e()},$e=()=>{new R().load("/textures/sphere2.png",t=>{const i=t;i.mapping=J,i.colorSpace=Re;const l=new Fe(60,64,64),u=new Q({map:i,side:Ee}),p=new k(l,u);c.add(p)}),new R().load("/textures/patern2.png",t=>{const i=t;i.mapping=J,i.colorSpace=Re,i.wrapS=We,i.wrapT=We,i.repeat.set(60,60);const l=new Fe(60,64,64),u=new Q({map:t,side:Ee,transparent:!0}),p=new k(l,u);c.add(p),r.finishTransition(),r.setPreloaderDone(!0),r.setSceneStartingPosition()})},Me=()=>{m.aspect=window.innerWidth/window.innerHeight,m.updateProjectionMatrix(),f.setSize(window.innerWidth,window.innerHeight)},ye=()=>{n.material.update(),_.update(),w.render()};return(t,i)=>{const l=ht;return T(),D("div",Y,[Et(l,null,{default:Wt(()=>[o("video",{ref_key:"sceneVideo",ref:s,autoplay:"",muted:"",loop:"",playsinline:"",id:"video"},[o("source",{src:e.video.video,type:"video/mp4"},null,8,Z)],512)]),_:1})])}}},Ie=ee,te=pt("/images/icons/awards.svg"),ae={class:"case-info"},re={class:"awards"},ie={key:0,alt:"awards",src:te},oe={class:"wrapper"},se={class:"left"},ne={class:"center"},le=["href"],de=o("span",{class:"elem-1 elem"},"[",-1),me=o("span",{class:"elem-2 elem"},"]",-1),ue=o("span",{class:"elem-3 elem"},"[",-1),ce=o("span",{class:"elem-4 elem"},"]",-1),he={class:"circle"},pe={class:"inner-circle"},fe=["innerHTML"],ve={class:"right"},xe=o("span",null,"Next Work",-1),we=o("img",{alt:"arrow",src:vt},null,-1),ge=[xe,we],Pe={class:"tags"},_e={class:"tag"},be={__name:"caseInfo",props:["page"],setup(h){const e=Ue(),s=d=>{ft(d)};return(d,a)=>(T(),D("div",ae,[o("div",re,[h.page.awards?(T(),D("img",ie)):kt("",!0),(T(!0),D(ke,null,Le(h.page.awards,r=>(T(),D("p",null,q(r),1))),256))]),o("div",oe,[o("div",se,[o("h1",null,q(h.page.name),1),o("h2",null,q(h.page.description),1)]),o("div",ne,[o("a",{target:"_blank",href:h.page.link,class:"send"},[o("div",{onMouseleave:a[0]||(a[0]=r=>G(e).hideCursor("","none")),onMouseenter:a[1]||(a[1]=r=>G(e).blendCursor("difference")),class:"child"},[de,me,ue,ce,o("div",he,[o("div",pe,[o("p",{innerHTML:h.page.buttonText},null,8,fe)])])],32)],8,le)]),o("div",ve,[o("div",{onMouseleave:a[2]||(a[2]=r=>G(e).hideCursor("","none")),onMouseenter:a[3]||(a[3]=r=>G(e).blendCursor("difference")),onClick:a[4]||(a[4]=r=>s(h.page.nextCase)),class:"next"},ge,32),o("div",Pe,[(T(!0),D(ke,null,Le(h.page.tags,r=>(T(),D("p",_e,q(r),1))),256))])])])]))}},Oe=be});export{Ie as C,Lt as __tla,Oe as a};
