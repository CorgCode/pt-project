/* Browser-local workbench. No inference service, telemetry or remote file upload. */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),math=window.PTLab;
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const token=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
 const colors={ink:token('--model-ink'),muted:token('--model-muted'),accent:token('--model-accent'),line:token('--lab-grid'),edge:token('--lab-edge'),base:token('--bar-muted'),groups:[token('--model-muted'),token('--lab-purple-soft'),token('--lab-purple-pale'),token('--model-accent'),token('--lab-soft'),token('--bar-muted')]};
 const sample='src,dst,bytes,packets,dst_port,proto\n'+Array.from({length:120},(_,i)=>{
  const device=i%8,service=i%7===0?(device+1)%3:Math.floor(device/3)%3;
  return `device-${String(device+1).padStart(2,'0')},service-${service+1},${(device+1)*280+(i%13)*170},${(device+1)*2+i%9},${[443,53,22][service]},${service===1?'UDP':'TCP'}`;
 }).join('\n');
 const featureNames=['Исходящие записи','Входящие записи','Байты отправлено','Байты получено','Пакеты отправлено','Пакеты получено','Разных контактов','Всего записей',...Array.from({length:6},(_,i)=>`Сервисы · исходящие ${i+1}`),...Array.from({length:6},(_,i)=>`Сервисы · входящие ${i+1}`)];
 let state=null,comparator=null,fileRequest=0,view=null,fullView=null,dragged=false;
 const charts={};
 function element(tag,text,className){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(className)e.className=className;return e;}
 function status(text,error=false){$('lab-status').textContent=text;$('lab-status').dataset.error=String(error);}
 function preset(value){document.querySelectorAll('[data-preset]').forEach(b=>b.setAttribute('aria-pressed',String(b.dataset.preset===value)));}
 function options(select,hosts,selected){select.replaceChildren(...hosts.map(h=>{const o=element('option',h.id);o.value=h.id;return o;}));select.value=selected;}
 function refreshLayout(){window.ScrollTrigger?.refresh();}
 function install(data,label,description){
  const k=Math.min(Number($('group-count').value),data.dataset.length);
  $('group-count').value=String(k);
  for(const option of $('group-count').options)option.disabled=Number(option.value)>data.dataset.length;
  state={...data,groups:math.group(data.dataset,k),label};
  $('dataset-description').textContent=description;
  $('lab-mode').textContent=data.kind==='flows'?'20 простых признаков':'Готовые числовые профили';
  options($('lab-host'),state.dataset,state.dataset[0].id);
  $('lab-summary').replaceChildren(...[[state.dataset.length,'устройств'],[data.stats.dimensions,'признаков'],...(data.stats.flows?[[data.stats.flows,'сетевых записей']]:[])].map(([value,label])=>{const d=element('div');d.append(element('strong',String(value)),element('small',label));return d;}));
  fullView=window.innerWidth<600?{minX:0,maxX:360,minY:0,maxY:790}:{minX:0,maxX:1000,minY:0,maxY:600};view={...fullView};
  comparator=null;select(state.dataset[0].id);
  status(`${label} · ${state.dataset.length} устройств. Выберите устройство на карте или в списке.`);
  refreshLayout();
 }
 function select(id){
  if(!state)return;
  const selected=state.dataset.find(h=>h.id===id);if(!selected)return;
  if(state.selected?.id!==id)view={...fullView};
  state.selected=selected;
  state.near=state.dataset.filter(h=>h!==selected).map(host=>({host,score:math.cosine(selected,host)})).sort((a,b)=>b.score-a.score).slice(0,5);
  $('lab-host').value=id;
  $('selected-name').textContent=id;
  const index=state.dataset.indexOf(selected),group=state.groups[index];
  $('selected-group').textContent=`Группа ${group+1} · ${selected.v.length} числовых признаков`;
  $('lab-neighbor-buttons').replaceChildren(...state.near.map(n=>{const button=element('button');button.type='button';button.append(element('span',n.host.id+' ↗'),element('span',n.score.toFixed(3)));button.setAttribute('aria-label',`Выбрать ${n.host.id}, сходство ${n.score.toFixed(3)}`);button.addEventListener('click',()=>select(n.host.id));return button;}));
  $('lab-insight-title').textContent=`Ближе всего — ${state.near[0].host.id}`;
  $('lab-insight-copy').textContent=`Косинусное сходство ${state.near[0].score.toFixed(3)}. Поиск выполнен по всем ${state.dataset.length} устройствам и всем ${selected.v.length} признакам. Близость профилей помогает выбрать соседей для сравнения; одинаковую роль она не доказывает.`;
  if(!comparator||comparator===id||!state.dataset.some(h=>h.id===comparator))comparator=state.near[0].host.id;
  options($('compare-host'),state.dataset.filter(h=>h!==selected),comparator);
  drawMap();drawNeighbors();drawGroups();drawComparison();
  if(window.gsap&&!reduced)gsap.fromTo('.inspector #selected-name,.inspector #selected-group',{opacity:.25,y:8},{opacity:1,y:0,duration:.5,overwrite:true});
  window.__ptLabState=state;
 }
 function chart(name,id,type,data,options){
  if(!window.Chart)return;
  if(charts[name]){charts[name].data=data;charts[name].options=options;charts[name].update(reduced?'none':undefined);return;}
  charts[name]=new Chart($(id),{type,data,options,plugins:[]});
 }
 function common(){return {responsive:true,maintainAspectRatio:false,animation:reduced?false:{duration:850,easing:'easeOutQuart'},plugins:{legend:{display:false},tooltip:{displayColors:false}},layout:{padding:8}};}
 function drawMap(){
  drawNetwork();
  const ratio=(fullView.maxX-fullView.minX)/(view.maxX-view.minX);$('map-zoom').textContent=ratio.toFixed(1)+'×';
 }
 function drawNeighbors(){
  const opt=common();opt.indexAxis='y';opt.scales={x:{min:-1,max:1,grid:{color:colors.line},border:{display:false},ticks:{font:{size:10}}},y:{grid:{display:false},border:{display:false},ticks:{font:{size:11},callback:function(value){const label=this.getLabelForValue(value);return label.length>20?label.slice(0,18)+'…':label;}}}};
  opt.plugins.tooltip.callbacks={label:c=>c.raw.toFixed(3)};
  chart('neighbors','lab-neighbors-chart','bar',{labels:state.near.map(n=>n.host.id),datasets:[{data:state.near.map(n=>n.score),backgroundColor:state.near.map((_,i)=>i===0?colors.accent:colors.base),barThickness:15,borderRadius:2}]},opt);
 }
 function drawGroups(){
  const current=state.groups[state.dataset.indexOf(state.selected)];
  $('group-summary').replaceChildren(...[...new Set(state.groups)].sort((a,b)=>a-b).map(group=>{
   const ids=state.dataset.filter((_,i)=>state.groups[i]===group);
   const b=element('button',`Группа ${group+1} · ${ids.length}`);b.type='button';b.setAttribute('aria-pressed',String(group===current));b.setAttribute('aria-label',`Группа ${group+1}: ${ids.length} устройств. Выбрать первое устройство`);b.addEventListener('click',()=>select(ids[0].id));return b;
  }));
 }
 function drawComparison(){
  const a=state.selected,b=state.dataset.find(h=>h.id===comparator);if(!b)return;
  const av=a.v.map(x=>x/a.norm),bv=b.v.map(x=>x/b.norm);
  const flow=state.kind==='flows';
  const values=a.v.map((_,i)=>({i,a:flow?a.v[i]:av[i],b:flow?b.v[i]:bv[i]}));
  const visible=values.slice().sort((x,y)=>(Math.abs(y.a)+Math.abs(y.b))-(Math.abs(x.a)+Math.abs(x.b))).slice(0,flow?8:10);
  const name=i=>flow?featureNames[i]:`Координата ${i}`;
  $('comparison-description').textContent=flow?'Сравните направления, объёмы и активность двух устройств на одном графике.':'Сравните значения нормированных координат. Для готового embedding смысл отдельных координат заранее неизвестен.';
  $('feature-caption').textContent=flow?'Восемь наиболее выраженных признаков пары. Применены log(1+x) и масштабирование по набору; высота столбика не показывает сырые байты или пакеты.':'Десять координат с наибольшей суммой модулей у пары после L2-нормировки. Это фрагмент профиля; сходство рассчитано по всем координатам.';
  const opt=common();opt.indexAxis='y';opt.plugins.legend={display:true,position:'bottom',labels:{usePointStyle:true,boxWidth:6,font:{size:10}}};opt.plugins.tooltip.callbacks={label:c=>`${c.dataset.label}: ${Number(c.raw).toFixed(3)}`};
  opt.scales={x:{grid:{color:colors.line},border:{display:false},ticks:{font:{size:9}}},y:{grid:{display:false},border:{display:false},ticks:{font:{size:10}}}};
  chart('features','lab-features-chart','bar',{labels:visible.map(v=>name(v.i)),datasets:[{label:a.id.length>20?a.id.slice(0,18)+'…':a.id,data:visible.map(v=>v.a),backgroundColor:colors.accent,barThickness:7},{label:b.id.length>20?b.id.slice(0,18)+'…':b.id,data:visible.map(v=>v.b),backgroundColor:colors.base,barThickness:7}]},opt);
  const score=math.cosine(a,b);$('pair-score-value').textContent=score.toFixed(3);$('pair-score-marker').style.left=((score+1)*50)+'%';
  drawFingerprint(a,b,av,bv);
  $('pair-title').textContent=`Сходство пары — ${score.toFixed(3)}`;
  $('pair-copy').textContent=flow?'Справа — близкие по значениям простые признаки этой пары. Нулевые у обоих признаки исключены. Сходство может объясняться несколькими сторонами поведения сразу; это сравнение признаков, а не объяснение нейросети.':'Справа — близкие по значениям ненулевые координаты. Без словаря признаков нельзя назвать их сервисами или ролями. Поиск работает с готовыми векторами и не восстанавливает причину их происхождения.';
  const close=values.filter(v=>Math.max(Math.abs(v.a),Math.abs(v.b))>1e-8).sort((x,y)=>Math.abs(x.a-x.b)-Math.abs(y.a-y.b)).slice(0,4);
  $('shared-features').replaceChildren(...close.map(v=>{const d=element('div');d.append(element('span',name(v.i)),element('span',`${v.a.toFixed(2)} / ${v.b.toFixed(2)}`));return d;}));
  state.comparator=b;state.pairScore=score;
 }
 const ns='http://www.w3.org/2000/svg';
 function svg(tag,attributes,text){const e=document.createElementNS(ns,tag);for(const [key,value]of Object.entries(attributes))e.setAttribute(key,String(value));if(text!==undefined)e.textContent=text;return e;}
 function drawNetwork(){
  const network=$('neighbor-network'),mobile=window.innerWidth<600;
  network.setAttribute('viewBox',`${view.minX} ${view.minY} ${view.maxX-view.minX} ${view.maxY-view.minY}`);
  const center=mobile?[180,380]:[500,290];
  const spots=mobile?[[85,105],[275,105],[275,560],[85,560],[180,730]]:[[180,110],[820,110],[830,455],[170,455],[500,525]];
  const fragment=document.createDocumentFragment();
  state.near.forEach((n,i)=>{
   const p=spots[i],path=svg('path',{d:`M ${center[0]} ${center[1]} L ${p[0]} ${p[1]}`,class:'clear-edge'});
   const x=center[0]+(p[0]-center[0])*.52,y=center[1]+(p[1]-center[1])*.52;
   fragment.append(path,svg('rect',{x:x-(mobile?22:26),y:y-11,width:mobile?44:52,height:23,rx:12,class:'edge-label-bg'}),svg('text',{x,y:y+5,'text-anchor':'middle',class:'edge-score'},n.score.toFixed(3)));
  });
  function node(host,point,selected,score){
   const group=state.groups[state.dataset.indexOf(host)],width=mobile?142:190,height=selected?108:96;
   const g=svg('g',{class:`network-node clear-node${selected?' is-selected':''}${$('show-groups').checked?' tinted':''}`,transform:`translate(${point[0]} ${point[1]})`,role:'button',tabindex:'0','data-group':group,'aria-label':selected?`Выбрано ${host.id}`:`Выбрать ${host.id}, сходство ${score.toFixed(3)}`});
   g.append(svg('title',{},host.id),svg('rect',{x:-width/2,y:-height/2,width,height,rx:14,class:'device-tile'}));
   g.append(svg('rect',{x:-12,y:-height/2+14,width:24,height:16,rx:2,class:'device-icon'}),svg('path',{d:`M 0 ${-height/2+30} V ${-height/2+35} M -8 ${-height/2+35} H 8`,class:'device-icon'}));
   const label=host.id.length>(mobile?18:25)?host.id.slice(0,mobile?16:23)+'…':host.id;
   g.append(svg('text',{y:10,'text-anchor':'middle',class:'device-name'},label),svg('text',{y:32,'text-anchor':'middle',class:'device-detail'},selected?'Выбрано':`Группа ${group+1} · выбрать ↗`));
   g.addEventListener('click',()=>{if(!dragged)select(host.id);});g.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select(host.id);}});return g;
  }
  state.near.forEach((n,i)=>fragment.append(node(n.host,spots[i],false,n.score)));fragment.append(node(state.selected,center,true,1));network.replaceChildren(fragment);
 }
 function zoom(factor){
  const next=(view.maxX-view.minX)*factor,base=fullView.maxX-fullView.minX;if(next<base/20||next>base*4)return;
  const cx=(view.minX+view.maxX)/2,cy=(view.minY+view.maxY)/2,h=(view.maxY-view.minY)*factor/2;
  view={minX:cx-next/2,maxX:cx+next/2,minY:cy-h,maxY:cy+h};drawMap(false);
 }
 function drawFingerprint(a,b,av,bv){
  const root=$('profile-fingerprint'),mobile=window.innerWidth<600;
  const indices=av.map((_,i)=>i).sort((i,j)=>Math.abs(av[j])+Math.abs(bv[j])-Math.abs(av[i])-Math.abs(bv[i])).slice(0,32);
  const n=indices.length,cols=Math.min(n,mobile?8:16),rows=Math.ceil(n/cols),cell=mobile?34:45,gap=mobile?5:8,pad=4,block=rows*(cell+gap)+40,width=cols*(cell+gap)+pad*2;
  root.setAttribute('viewBox',`0 0 ${width} ${block*2+40}`);
  const max=Math.max(.00001,...indices.flatMap(i=>[Math.abs(av[i]),Math.abs(bv[i])])),fragment=document.createDocumentFragment();
  const label=(value,selected)=>{const text=svg('text',{x:pad,y:selected?18:block+18,class:'fingerprint-label'},value.length>42?value.slice(0,40)+'…':value);fragment.append(text);};
  label(a.id,true);label(b.id,false);
  [av,bv].forEach((v,row)=>indices.forEach((index,j)=>{
   const x=pad+j%cols*(cell+gap),y=32+row*block+Math.floor(j/cols)*(cell+gap);
   const rect=svg('rect',{x,y,width:cell,height:cell,rx:4,class:v[index]<0?'fingerprint-cell negative':'fingerprint-cell','fill-opacity':.12+.88*Math.abs(v[index])/max,tabindex:0,role:'button','aria-label':`${row?b.id:a.id}, координата ${index}: ${v[index].toFixed(4)}`});
   const show=()=>{$('fingerprint-readout').textContent=`Координата ${index}: ${a.id} — ${av[index].toFixed(4)}; ${b.id} — ${bv[index].toFixed(4)}.`;};
   rect.append(svg('title',{},`Координата ${index}: ${v[index].toFixed(4)}`));rect.addEventListener('pointerenter',show);rect.addEventListener('focus',show);rect.addEventListener('click',show);rect.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();show();}});fragment.append(rect);
  }));
  root.replaceChildren(fragment);$('fingerprint-copy').textContent=`${n} наиболее выраженных координат после L2-нормировки. Порядок одинаков для обоих устройств. Насыщенность — модуль значения; серый цвет — отрицательное значение. Это фрагмент числового профиля, не изображение трафика.`;
  $('fingerprint-readout').textContent='Наведите на ячейку или выберите её клавиатурой, чтобы увидеть точные значения.';
 }
 async function upload(file){
  if(!file)return;const request=++fileRequest;
  if(file.size>3*1024*1024){status('Файл слишком большой. Поддерживается CSV до 3 МБ.',true);return;}
  status('Читаем CSV и проверяем значения…');
  try{const text=await file.text();if(request!==fileRequest)return;const data=math.parseCSV(text);install(data,file.name,data.kind==='flows'?'Ваш трафик · простые признаки построены в браузере.':'Ваши готовые профили · поиск по всем координатам.');preset('uploaded');}
  catch(error){if(request===fileRequest)status('Не удалось прочитать файл: '+error.message,true);}
 }
 function download(text,name){const url=URL.createObjectURL(new Blob([text],{type:'text/csv;charset=utf-8'})),a=element('a');a.href=url;a.download=name;a.click();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 $('zoom-in').addEventListener('click',()=>zoom(.75));$('zoom-out').addEventListener('click',()=>zoom(1/.75));
 $('reset-map').addEventListener('click',()=>{view={...fullView};drawMap();});

 const mapCanvas=$('neighbor-network');let drag=null;
 mapCanvas.addEventListener('pointerdown',e=>{if(e.button!==0)return;dragged=false;drag={x:e.clientX,y:e.clientY,start:{...view}};});
 mapCanvas.addEventListener('pointermove',e=>{
  if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)<6&&!dragged)return;if(!dragged)mapCanvas.setPointerCapture?.(e.pointerId);dragged=true;
  const rect=mapCanvas.getBoundingClientRect(),width=rect.width||500,height=rect.height||400;
  const vx=dx/width*(drag.start.maxX-drag.start.minX),vy=dy/height*(drag.start.maxY-drag.start.minY);
  view={minX:drag.start.minX-vx,maxX:drag.start.maxX-vx,minY:drag.start.minY+vy,maxY:drag.start.maxY+vy};mapCanvas.classList.add('is-dragging');drawMap(false);
 });
 const stopDrag=()=>{drag=null;mapCanvas.classList.remove('is-dragging');};mapCanvas.addEventListener('pointerup',stopDrag);mapCanvas.addEventListener('pointercancel',stopDrag);mapCanvas.addEventListener('lostpointercapture',stopDrag);
 $('download-sample').addEventListener('click',()=>download(sample,'example-network.csv'));
 $('csv-file').addEventListener('change',e=>{const file=e.target.files[0];e.target.value='';upload(file);});
 const drop=$('upload-zone');drop.addEventListener('dragover',e=>{e.preventDefault();drop.classList.add('drag-over');});drop.addEventListener('dragleave',()=>drop.classList.remove('drag-over'));drop.addEventListener('drop',e=>{e.preventDefault();drop.classList.remove('drag-over');upload(e.dataTransfer.files[0]);});
 $('lab-host').addEventListener('change',()=>select($('lab-host').value));
 $('compare-host').addEventListener('change',()=>{comparator=$('compare-host').value;drawComparison();});
 $('show-groups').addEventListener('change',drawMap);
 $('group-count').addEventListener('change',()=>{state.groups=math.group(state.dataset,Number($('group-count').value));select(state.selected.id);});
 document.querySelectorAll('[data-preset]').forEach(b=>b.addEventListener('click',()=>{fileRequest++;preset(b.dataset.preset);if(b.dataset.preset==='flows'){install(math.parseCSV(sample),'Учебный трафик','Небольшой сгенерированный пример соединений — для знакомства с поиском.');return;}const dataset=window.HOSTS.slice(0,b.dataset.preset==='small'?24:300).map(h=>({...h,v:[...h.v],norm:Math.hypot(...h.v)}));install({dataset,stats:{hosts:dataset.length,dimensions:dataset[0].v.length},kind:'project'},'Учебные профили','Синтетические тестовые профили. Они не являются результатом multiday_codebook_plus.');}));
 if(window.Chart){Chart.defaults.font.family=token('--model-font');Chart.defaults.color=colors.muted;}
 install(math.parseCSV(sample),'Учебный трафик','Небольшой сгенерированный пример соединений — для знакомства с поиском.');
 if(!window.Chart)status('Графики не загрузились. Поиск, группы и сравнение значений доступны ниже.',true);
 let mobile=window.innerWidth<600;
 window.addEventListener('resize',()=>{const next=window.innerWidth<600;if(mobile!==next){mobile=next;fullView=mobile?{minX:0,maxX:360,minY:0,maxY:790}:{minX:0,maxX:1000,minY:0,maxY:600};view={...fullView};drawMap();drawComparison();}},{passive:true});
 if(window.gsap&&window.ScrollTrigger&&!reduced){document.querySelectorAll('.lab-hero-visual,.data-layout,.workspace-grid,.comparison-grid,.explanation,.neighborhood-figure,.format-grid,.boundary-grid').forEach(e=>gsap.fromTo(e,{opacity:0,y:32},{opacity:1,y:0,duration:1.25,ease:'power3.out',scrollTrigger:{trigger:e,start:'top 94%',once:true}}));}
 window.__ptSelectHost=select;
})();
