/* Compare two observed windows; this is a local exploration tool, not an alert model. */
(()=>{
 const $=id=>document.getElementById(id),math=window.PTLab,ns='http://www.w3.org/2000/svg';
 const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
 const exampleBefore='src,dst,bytes,packets,dst_port,proto\n'+Array.from({length:160},(_,i)=>{const h=i%8,s=Math.floor(h/2)%4;return `host-${String(h+1).padStart(2,'0')},service-${s+1},${300+h*100+i%13*40},${2+i%7},${[443,53,22,445][s]},${s===1?'UDP':'TCP'}`;}).join('\n');
 const exampleAfter=exampleBefore+'\n'+Array.from({length:70},(_,i)=>`host-03,service-new,${12000+i*80},${40+i%7},8443,TCP`).join('\n');
 let inputs={},report=null,current=null,period='before',requests={before:0,after:0},listLimit=12;
 function element(tag,text,cls){const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;}
 function svg(tag,attrs,text){const e=document.createElementNS(ns,tag);for(const [k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;}
 function resultVisibility(visible){document.querySelector('.changes-workspace').hidden=!visible;document.querySelector('.change-timeline').hidden=!visible;$('change-visual').hidden=!visible;}
 function calculate(){
  try{report=math.compareWindows(inputs.before,inputs.after,3);current=report.changes[0];period='before';listLimit=12;$('change-host-search').value='';resultVisibility(true);$('change-status').textContent=`Общих устройств: ${report.changes.length}. Появилось: ${report.newHosts.length}; отсутствует во втором окне: ${report.missingHosts.length}. Выберите устройство для разбора.`;render();}
  catch(error){resultVisibility(false);$('change-status').textContent='Сравнение не построено: '+error.message;}
 }
 function renderList(){
  const query=$('change-host-search').value.trim().toLocaleLowerCase('ru');
  const matches=report.changes.filter(change=>change.id.toLocaleLowerCase('ru').includes(query));
  $('change-list-status').textContent=`Найдено ${matches.length} из ${report.changes.length}. Показано ${Math.min(listLimit,matches.length)}.`;
  $('change-more').hidden=matches.length<=listLimit;
  $('change-host-list').replaceChildren(...matches.slice(0,listLimit).map(change=>{const b=element('button');b.type='button';b.setAttribute('aria-pressed',String(change===current));b.append(element('span',change.id),element('span',change.similarity.toFixed(3)));b.setAttribute('aria-label',`${change.id}, сходство с прежним профилем ${change.similarity.toFixed(3)}`);b.addEventListener('click',()=>{current=change;render();});return b;}));
 }
 function render(){
  renderList();
  $('change-host-name').textContent=current.id;
  $('change-summary').textContent=current.similarity<.999999?`Профиль отличается от первого окна. Сходство с прежним состоянием — ${current.similarity.toFixed(3)}. Это повод посмотреть на изменения в данных, а не оценка вероятности атаки.`:'Профиль в этих окнах практически не изменился. Число около 1 означает близкие направления числовых представлений.';
  const items=[[current.similarity.toFixed(3),'сходство раньше и сейчас'],[`${current.beforeGroup+1} → ${current.afterGroup+1}`,'группа относительно первого окна']];
  if(report.kind==='flows')items.push([`${current.before.activity.outgoing} → ${current.after.activity.outgoing}`,'исходящих записей'],[`${current.added.length} / ${current.removed.length}`,'новых / исчезнувших контактов']);
  $('change-measures').replaceChildren(...items.map(([value,label])=>{const d=element('div');d.append(element('strong',value),element('span',label));return d;}));
  $('change-contacts').replaceChildren();
  for(const [kind,ids]of [['Новые',current.added],['Исчезнувшие',current.removed]]){const row=element('div');row.append(element('strong',kind+': '),element('span',ids.length?ids.join(', '):'нет'));$('change-contacts').append(row);}
  $('change-contact-copy').textContent=report.kind==='flows'?'Контакты взяты из двух CSV. Пропажа из второго окна означает отсутствие в этих записях, а не отключение устройства от сети.':'Загружены готовые векторы. Без записей обмена нельзя восстановить новые и исчезнувшие контакты.';
  $('change-view-before').setAttribute('aria-pressed',String(period==='before'));$('change-view-after').setAttribute('aria-pressed',String(period==='after'));
  draw();window.__ptChangeState={report,current,period};
 }
 function draw(){
  const root=$('change-visual'),mobile=innerWidth<600,contacts=period==='before'?current.beforeContacts:current.afterContacts;
  const count=Math.min(5,contacts.length),height=count<=2?270:count<=4?380:520;
  root.setAttribute('viewBox',mobile?`0 0 360 ${height}`:'0 0 900 350');const center=mobile?[180,70]:[170,175],fragment=document.createDocumentFragment();
  const slots=mobile?[[90,210],[270,210],[90,320],[270,320],[180,430]]:Array.from({length:count},(_,i)=>[650,175+(i-(count-1)/2)*70]);
  const title=period==='before'?'Первое окно':'Второе окно';
  fragment.append(svg('rect',{x:center[0]-(mobile?78:110),y:center[1]-35,width:mobile?156:220,height:70,rx:18,class:'change-selected'}),svg('text',{x:center[0],y:center[1]-1,'text-anchor':'middle',class:'change-selected-label'},current.id.length>22?current.id.slice(0,20)+'…':current.id),svg('text',{x:center[0],y:center[1]+21,'text-anchor':'middle',class:'change-period-label'},title));
  const highlighted=period==='after'?current.added:current.removed;
  const visibleContacts=[...contacts].sort((a,b)=>Number(highlighted.includes(b))-Number(highlighted.includes(a))).slice(0,5);
  $('change-map-caption').textContent=contacts.length?`Показано ${visibleContacts.length} из ${contacts.length} контактов. Изменившиеся контакты показаны первыми; новые выделены цветом. Полный список изменений — выше.`:'Без записей контактов схема не строится.';
  visibleContacts.forEach((id,i)=>{const p=slots[i],isNew=period==='after'&&current.added.includes(id);fragment.prepend(svg('path',{d:`M ${center[0]} ${center[1]} L ${p[0]} ${p[1]}`,class:isNew?'change-link is-new':'change-link'}));const node=svg('g',{class:'change-contact'+(isNew?' is-new':''),role:report.changes.some(c=>c.id===id)?'button':'img',tabindex:report.changes.some(c=>c.id===id)?0:-1,'aria-label':id+(isNew?', новый контакт':'')});node.append(svg('rect',{x:p[0]-(mobile?74:120),y:p[1]-25,width:mobile?148:240,height:50,rx:13}),svg('text',{x:p[0],y:p[1]+4,'text-anchor':'middle'},id.length>20?id.slice(0,18)+'…':id));
   const choose=()=>{const change=report.changes.find(c=>c.id===id);if(change){current=change;render();}};node.addEventListener('click',choose);node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});fragment.append(node);
  });
  if(!contacts.length)fragment.append(svg('text',{x:mobile?180:620,y:mobile?220:180,'text-anchor':'middle',class:'change-empty'},report.kind==='flows'?'Контактов в этом окне не наблюдалось':'Векторы не содержат записей контактов'));
  root.replaceChildren(fragment);
  if(!reduced&&window.gsap)window.gsap.fromTo(root.querySelectorAll('.change-contact'),{opacity:.35,y:10},{opacity:1,y:0,duration:.65,stagger:.06,ease:'power2.out'});
 }
 async function load(which,file){
  if(!file)return;const request=++requests[which];
  try{if(file.size>3*1024*1024)throw Error('Поддерживается CSV до 3 МБ.');const text=await file.text();if(request!==requests[which])return;inputs[which]=math.parseCSV(text);$(which+'-file-label').textContent=file.name;calculate();}
  catch(error){if(request!==requests[which])return;resultVisibility(false);$('change-status').textContent='Не удалось прочитать окно: '+error.message;}
 }
 function example(){requests.before++;requests.after++;inputs={before:math.parseCSV(exampleBefore),after:math.parseCSV(exampleAfter)};$('before-file-label').textContent='Учебный пример · до изменения';$('after-file-label').textContent='Учебный пример · после изменения';calculate();}
 ['before','after'].forEach(which=>$('window-'+which).addEventListener('change',e=>{const file=e.target.files[0];e.target.value='';load(which,file);}));
 $('change-host-search').addEventListener('input',()=>{listLimit=12;if(report)renderList();});$('change-more').addEventListener('click',()=>{listLimit+=12;renderList();});
 $('change-example').addEventListener('click',example);$('change-view-before').addEventListener('click',()=>{period='before';render();});$('change-view-after').addEventListener('click',()=>{period='after';render();});
 let mobile=innerWidth<600;window.addEventListener('resize',()=>{const next=innerWidth<600;if(next!==mobile){mobile=next;if(report)draw();}},{passive:true});
 example();
})();
