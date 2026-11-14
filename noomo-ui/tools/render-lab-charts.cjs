// Render the actual workbench Chart.js configs to CPU canvases, without browser automation.
const fs=require('node:fs'),path=require('node:path'),{JSDOM}=require('jsdom'),{createCanvas,GlobalFonts}=require('@napi-rs/canvas');
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
 const chart=new Chart(canvas,c);fs.writeFileSync(path.join(output,['lab-map.png','lab-neighbors.png','lab-features.png'][i]),canvas.toBuffer('image/png'));chart.destroy();
}
dom.window.close();console.log('Rendered 3 live Chart.js configurations to',output);
