// Build the product's own social preview. No borrowed logos or screenshots.
const fs=require('node:fs'),path=require('node:path'),{createCanvas,GlobalFonts}=require('@napi-rs/canvas');
const root=path.resolve(__dirname,'..');GlobalFonts.registerFromPath(path.join(root,'fonts/onest-400.ttf'),'PTBody');GlobalFonts.registerFromPath(path.join(root,'fonts/onest-600.ttf'),'PTBold');GlobalFonts.registerFromPath(path.join(root,'model/reference/NeueMachina-Regular.otf'),'PTDisplay');
const canvas=createCanvas(1200,630),ctx=canvas.getContext('2d');
const c={paper:'#d2d9e9',ink:'#181520',muted:'#626671',accent:'#8231fe',edge:'#ffffffb8',glass:'#ffffff65',line:'#8231fe45'};
ctx.fillStyle=c.paper;ctx.fillRect(0,0,1200,630);const gradient=ctx.createRadialGradient(900,220,10,900,220,480);gradient.addColorStop(0,'#ffffffb0');gradient.addColorStop(1,'#ffffff00');ctx.fillStyle=gradient;ctx.fillRect(0,0,1200,630);
ctx.fillStyle=c.ink;ctx.font='26px PTBold';ctx.fillText('PT NETWORK',56,72);ctx.font='13px PTBody';ctx.fillText('МФТИ × Positive Technologies',56,99);
ctx.font='64px PTDisplay';ctx.fillText('ПОНЯТЬ',56,260);ctx.fillText('СВОЮ СЕТЬ.',56,336);
ctx.font='21px PTBody';ctx.fillText('Сетевое поведение становится контекстом.',56,402);ctx.fillText('Модель, исследование и работа с данными.',56,438);
ctx.font='14px PTBody';ctx.fillStyle=c.muted;ctx.fillText('ПРОДУКТ  /  МОДЕЛЬ  /  ЛАБОРАТОРИЯ',56,566);
function box(x,y,w,h,title,selected=false){ctx.beginPath();ctx.roundRect(x,y,w,h,20);ctx.fillStyle=selected?c.ink:c.glass;ctx.fill();ctx.strokeStyle=c.edge;ctx.lineWidth=1;ctx.stroke();ctx.fillStyle=selected?c.paper:c.ink;ctx.font='18px PTBody';ctx.textAlign='center';ctx.fillText(title,x+w/2,y+h/2+7);ctx.textAlign='left';}
ctx.strokeStyle=c.line;ctx.lineWidth=2;for(const [x,y]of[[835,135],[1040,370],[800,485]]){ctx.beginPath();ctx.moveTo(915,290);ctx.lineTo(x,y);ctx.stroke();}
box(755,90,180,90,'Похожие');box(825,240,190,100,'Устройство',true);box(960,335,180,90,'Контакты');box(715,455,180,90,'Группы');
fs.writeFileSync(path.join(root,'OpenGraph.jpg'),canvas.toBuffer('image/jpeg'));
console.log('Generated own PT Network social preview');
