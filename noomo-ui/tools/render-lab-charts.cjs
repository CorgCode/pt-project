// Render the actual workbench Chart.js configs to CPU canvases, without browser automation.
const fs=require('node:fs'),path=require('node:path'),{JSDOM}=require('jsdom'),{createCanvas,GlobalFonts,loadImage}=require('@napi-rs/canvas');
const Chart=require('../vendor/chart.min.js'),root=path.resolve(__dirname,'..');
const dom=new JSDOM(fs.readFileSync(path.join(root,'lab/index.html'),'utf8'),{runScripts:'outside-only',url:'http://localhost/lab/'}),w=dom.window,configs=[];
w.matchMedia=()=>({matches:true});w.innerWidth=1440;w.Chart=class{static defaults={font:{}};constructor(canvas,c){this.data=c.data;this.options=c.options;configs.push(c);}update(){}};
const style=w.document.createElement('style');style.textContent=fs.readFileSync(path.join(root,'model/model.css'),'utf8')+fs.readFileSync(path.join(root,'lab/lab.css'),'utf8');w.document.head.append(style);
for(const file of ['product-data.js','lab/core.js','lab/lab.js'])w.eval(fs.readFileSync(path.join(root,file),'utf8'));
GlobalFonts.registerFromPath(path.join(root,'fonts/onest-400.ttf'),'OnestCase');Chart.defaults.font.family='OnestCase';
const output='/private/tmp/pt-ui-qa';fs.mkdirSync(output,{recursive:true});
for(const [i,c]of configs.entries()){
 const canvas=createCanvas(i===0?1040:800,i===0?600:500);
 c.plugins??=[];c.plugins.unshift({id:'preview-background',beforeDraw(c){c.ctx.save();c.ctx.fillStyle='#d2d9e9';c.ctx.fillRect(0,0,c.width,c.height);c.ctx.restore();}});
 c.options.responsive=false;c.options.animation=false;delete c.options.onClick;c.options.plugins.tooltip.enabled=false;
 const chart=new Chart(canvas,c);fs.writeFileSync(path.join(output,['lab-neighbors.png','lab-features.png'][i]),canvas.toBuffer('image/png'));chart.destroy();
}
(async()=>{
 const css=style.textContent,variables=Object.fromEntries([...css.matchAll(/(--[\w-]+)\s*:\s*([^;{}]+);/g)].map(m=>[m[1],m[2]]));
 let resolved=css;for(let i=0;i<8;i++)resolved=resolved.replace(/var\((--[\w-]+)\)/g,(match,key)=>variables[key]||match);
 for(const mobile of [false,true]){
  if(mobile){w.innerWidth=375;w.dispatchEvent(new w.Event('resize'));}
  const svg=w.document.getElementById('neighbor-network').cloneNode(true),s=w.document.createElementNS('http://www.w3.org/2000/svg','style');s.textContent=resolved;svg.prepend(s);svg.setAttribute('xmlns','http://www.w3.org/2000/svg');svg.setAttribute('width',mobile?'360':'1000');svg.setAttribute('height',mobile?'790':'600');
  const image=await loadImage(Buffer.from(svg.outerHTML));const canvas=createCanvas(mobile?360:1000,mobile?790:600),ctx=canvas.getContext('2d');ctx.fillStyle='#d2d9e9';ctx.fillRect(0,0,canvas.width,canvas.height);ctx.drawImage(image,0,0);fs.writeFileSync(path.join(output,mobile?'lab-neighborhood-mobile.png':'lab-neighborhood.png'),canvas.toBuffer('image/png'));
 }
 const art=w.document.getElementById('soc-overview').cloneNode(true),artStyle=w.document.createElementNS('http://www.w3.org/2000/svg','style');artStyle.textContent=resolved;art.prepend(artStyle);art.setAttribute('xmlns','http://www.w3.org/2000/svg');art.setAttribute('width','520');art.setAttribute('height','440');
 const artImage=await loadImage(Buffer.from(art.outerHTML)),artCanvas=createCanvas(520,440),artContext=artCanvas.getContext('2d');artContext.fillStyle='#d2d9e9';artContext.fillRect(0,0,520,440);artContext.drawImage(artImage,0,0);fs.writeFileSync(path.join(output,'lab-soc-overview.png'),artCanvas.toBuffer('image/png'));
 dom.window.close();console.log('Rendered 2 Chart.js charts, desktop/mobile neighborhood and SOC overview to',output);
})().catch(error=>{console.error(error);process.exitCode=1;});
