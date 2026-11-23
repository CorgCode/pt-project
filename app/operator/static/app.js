/* CorgCode local workspace. Observed connections and profile similarity stay separate. */
(()=>{
 'use strict';
 const $=id=>document.getElementById(id),math=window.PTLab,ns='http://www.w3.org/2000/svg';
 const sample='src,dst,bytes,packets,dst_port,proto\n'+Array.from({length:180},(_,i)=>{const h=i%12,g=Math.floor(h/4);return `device-${String(h+1).padStart(2,'0')},service-${g+1},${400+h*210+i%9*80},${3+h+i%5},${[443,53,445][g]},${g===1?'UDP':'TCP'}`;}).join('\n');
 const incidentTime = '10:42';
 const incidentDay = '2026-11-18';
 const incidentHost = 'device-07';
 const incidentContact = 'service-4';
 const incidentBins = [0, 0, 14, 26, 17, 8];
 const earlierTime = ['10:00', '10:05', '10:10', '10:15', '10:20', '10:25'];
 const laterTime = ['10:30', '10:35', '10:40', '10:45', '10:50', '10:55'];

 function makeIncidentWindow(times, withIncident) {
   const rows = ['src,dst,bytes,packets,dst_port,proto,timestamp'];
   const counts = [];
   for (const [bucket, time] of times.entries()) {
     let outgoing = 0;
     for (let pc = 1; pc <= 12; pc++) {
       const id = `device-${String(pc).padStart(2, '0')}`;
       const group = Math.floor((pc - 1) / 4);
       const records = 3 + (pc + bucket * 3 + Number(withIncident)) % 5;
       for (let row = 0; row < records; row++) {
         const bytes = 480 + pc * 170 + row * 65;
         rows.push(`${id},service-${group + 1},${bytes},${3 + row},${[443, 53, 445][group]},${group === 1 ? 'UDP' : 'TCP'},${incidentDay}T${time}:00Z`);
       }
       if (id === incidentHost) outgoing = records;
     }
     const extra = withIncident ? incidentBins[bucket] : 0;
     for (let row = 0; row < extra; row++) {
       const timestamp = bucket === 2 ? incidentTime : time;
       rows.push(`${incidentHost},${incidentContact},${6400 + row * 160},35,8443,TCP,${incidentDay}T${timestamp}:00Z`);
     }
     counts.push(outgoing + extra);
   }
   return {csv: rows.join('\n'), counts};
 }
 const incidentBaseline = makeIncidentWindow(earlierTime, false);
 const incidentCurrent = makeIncidentWindow(laterTime, true);
 const sampleServiceNames = new Map([
   ['service-1', 'Веб-сервер'],
   ['service-2', 'DNS-сервер'],
   ['service-3', 'Файловый сервер'],
   ['service-4', 'Веб-сервис 8443'],
 ]);
 function hostName(id) {
   const label = workflow.annotation(id).label;
   if (label || !payload.sample) return label || id;
   const computer = /^device-(\d{2})$/.exec(id);
   return computer ? `Рабочий ПК ${computer[1]}` : sampleServiceNames.get(id) || id;
 }
 function hostNameWithId(id) {
   const name = hostName(id);
   return name === id ? id : `${name} · ${id}`;
 }
 const featureNames=['Исходящие записи','Входящие записи','Байты отправлено','Байты получено','Пакеты отправлено','Пакеты получено','Разных контактов','Всего записей'];
 let state=null,selected=null,comparator=null,groups=[],projection=[],currentView='devices',mode='similar',workspaceId=null,payload=null,limit=30,activeGroup=0,zoom=1,pan={x:0,y:0},drag=null,dragged=false,importRequest=0,baselineRequest=0,openRequest=0,revision=0,saving=false,intent=0,workflow=null;
 const element=(tag,text,cls)=>{const e=document.createElement(tag);if(text!==undefined)e.textContent=text;if(cls)e.className=cls;return e;};
 const svg=(tag,attrs,text)=>{const e=document.createElementNS(ns,tag);for(const[k,v]of Object.entries(attrs))e.setAttribute(k,String(v));if(text!==undefined)e.textContent=text;return e;};
 function status(text,error=false){$('status').textContent=text;$('status').dataset.error=String(error);}
 function dirty(){revision++;$('save-state').textContent='Есть несохранённые изменения';}
 const siteMode = Boolean(document.body.dataset.workspaceMode);
 let saveAfterLogin = false;
 function isGuest(){return document.body.dataset.workspaceMode==='guest';}
 function setAccountMode(registering){
  const form=$('account-form'), password=form.elements.password;
  form.action=registering?'/lab/register/':'/lab/login/';
  password.autocomplete=registering?'new-password':'current-password';
  if(registering)password.minLength=8;
  else password.removeAttribute('minlength');
  $('account-title').textContent=registering?'Регистрация':'Вход в рабочее место';
  $('account-purpose').textContent=registering
   ? 'Имя латиницей, пароль от 8 символов.'
   : saveAfterLogin?'После входа текущий набор сохранится.':'';
  form.querySelector('button[type="submit"]').textContent=registering?'Зарегистрироваться':'Войти';
  $('switch-account-mode').textContent=registering?'Уже есть аккаунт':'Создать аккаунт';
  $('account-error').hidden=true;
 }
 function openAccount(afterSave=false,registering=false){
  saveAfterLogin=afterSave;
  setAccountMode(registering);
  if(!$('account-dialog').open)$('account-dialog').showModal();
 }
 async function api(route,options){
  const response=await fetch(route,options);
  const data=await response.json().catch(()=>({}));
  if(response.status===401&&siteMode&&route!=='/lab/login/'){
   document.body.dataset.workspaceMode='guest';
   $('saved-workspaces').replaceChildren();
   $('workspace-dialog-list').replaceChildren();
  }
  if(!response.ok)throw Error(data.error||(response.status===429?'Слишком много запросов. Повторите позже.':'Не удалось выполнить запрос.'));
  return data;
 }
 async function signIn(event){
  event.preventDefault();
  const form=event.currentTarget, submit=form.querySelector('button[type="submit"]');
  const shouldSave=saveAfterLogin;
  const formBody=new URLSearchParams(new FormData(form));
  submit.disabled=true;
  $('switch-account-mode').disabled=true;
  $('account-error').hidden=true;
  try{
   const route=new URL(form.action).pathname;
   await api(route,{method:'POST',headers:{Accept:'application/json'},body:formBody});
   document.body.dataset.workspaceMode='account';
   form.reset();
   $('account-dialog').close();
   const rows=await savedList();
   if(workspaceId&&!rows.some(row=>row.id===workspaceId)){
    workspaceId=null;
    dirty();
    status('В этом аккаунте исходного набора нет. Нажмите «Сохранить», чтобы создать копию.');
   }else if(shouldSave)await save();
   else status('');
  }catch(error){
   if($('account-dialog').open){
    $('account-error').textContent=error.message;
    $('account-error').hidden=false;
   }else status(error.message,true);
  }finally{submit.disabled=false;$('switch-account-mode').disabled=false;}
 }
 function view(name){const previousView=$('view-'+currentView);const moveFocus=currentView!==name&&previousView?.contains(document.activeElement);currentView=name;document.querySelectorAll('.view').forEach(e=>e.hidden=e.id!=='view-'+name);document.querySelectorAll('[data-view]').forEach(e=>{if(e.dataset.view===name){e.setAttribute('aria-current','page');const section=e.closest('.secondary-views');if(section)section.open=true;}else e.removeAttribute('aria-current');});$('view-title').textContent={devices:'Устройства сети',groups:'Группы поведения',contacts:'Реальные контакты',changes:'Изменения во времени',case:'Разбор устройства',watch:'Под наблюдением'}[name];if(name==='groups')drawClusters();if(name==='case')renderCase();if(name==='changes')renderChanges();if(name==='contacts')renderContactTable();if(name==='watch')workflow.renderWatch();if(moveFocus)$('view-title').focus();}
 function applyEmbeddings(raw,result){
  const vectors=new Map(result.dataset.map(row=>[row.id,row]));
  if(vectors.size!==raw.dataset.length||raw.dataset.some(host=>!vectors.has(host.id)))throw Error('Идентификаторы эмбеддингов не совпадают с CSV.');
  return {...raw,model:result.provenance,raw,stats:{...raw.stats,dimensions:result.stats.dimensions},
    dataset:raw.dataset.map(host=>{const {v,norm}=vectors.get(host.id);return {...host,v,norm};})};
 }
 function install(next,data,id=null){const {inference,...saved}=data;state=inference?applyEmbeddings(next,inference):next;
  const modelLabel=$('embedding-model');
  if(modelLabel){modelLabel.hidden=!state.model;modelLabel.textContent=state.model?`Эмбеддинги: ${state.model.model_version} · ${state.stats.dimensions} координат`:'';}
  payload={...saved,notes:{...data.notes},annotations:{...data.annotations},settings:{groupCount:3,...data.settings}};workspaceId=id;$('group-count').value=payload.settings.groupCount;groups=math.group(state.dataset,Number($('group-count').value));projection=math.project(state.dataset);selected=state.dataset[0];comparator=null;mode=!state.model&&!data.sample&&next.kind==='flows'?'traffic':'similar';zoom=1;pan={x:0,y:0};limit=30;activeGroup=groups[0];$('host-search').value='';$('contact-search').value='';$('status-filter').value='';$('host-sort').value='name';$('watch-only').checked=false;$('change-search').value='';$('change-filter').value='all';$('workspace-name').value=data.name;$('mode-traffic').disabled=state.kind!=='flows';$('baseline-name').textContent=data.baseline?'Первое окно сохранено с набором.':'Первое окно не загружено.';$('save-state').textContent=id?'Набор открыт':'';renderGroupOptions();select(selected.id);workflow.renderWatch();renderChanges();status('');view(currentView);}
 function openSample(){intent++;importRequest++;baselineRequest++;openRequest++;install(math.parseCSV(sample),{name:'Знакомство с сетью',csv:sample,baseline:'',notes:{},sample:true});}
 async function openChangeScenario(){
  const request=++intent;
  if(!(await preserve())||request!==intent)return;
  importRequest++;baselineRequest++;openRequest++;
  install(math.parseCSV(incidentCurrent.csv), {
    name: 'Учебный инцидент: новый контакт',
    csv: incidentCurrent.csv, baseline: incidentBaseline.csv, notes: {}, sample: true,
  });
  select(incidentHost);
  view('changes');
  return true;
 }
 function nearest(){return state.dataset.filter(h=>h!==selected).map(host=>({host,score:math.cosine(selected,host)})).sort((a,b)=>b.score-a.score);}
 function contacts(data,id){const found=new Map();for(const edge of data.connections||[]){if(edge.src!==id&&edge.dst!==id)continue;const other=edge.src===id?edge.dst:edge.src;if(other===id)continue;const previous=found.get(other)||{id:other,flows:0,bytes:0};previous.flows+=edge.flows;previous.bytes+=edge.bytes;found.set(other,previous);}return [...found.values()].sort((a,b)=>b.flows-a.flows);}
 function select(id){const host=state.dataset.find(h=>h.id===id);if(!host)return;selected=host;const near=nearest();if(!comparator||comparator===selected)comparator=near[0].host;$('selected-group').textContent=`Группа ${groups[state.dataset.indexOf(host)]+1} · ${state.stats.dimensions} признаков`;$('activity').replaceChildren();$('activity-help').hidden=!host.activity;if(host.activity){for(const[label,value]of [['Исходящих записей',host.activity.outgoing],['Входящих записей',host.activity.incoming],['Разных контактов',host.activity.contacts],['Отправлено байт',host.activity.sentBytes],['Получено байт',host.activity.receivedBytes]]){const row=element('div');row.append(element('span',label),element('b',value.toLocaleString('ru-RU')));$('activity').append(row);}}else $('activity').append(element('p','Векторы описывают поведение, но не содержат исходные записи обмена.','small'));renderList();drawNetwork();renderNeighbors();renderCase();workflow.syncSelected();if(currentView==='groups')drawClusters();window.__corgState={state,selected,comparator,groups,payload};}
 function renderGroupOptions(){const actual=[...new Set(groups)].sort((a,b)=>a-b);$('group-filter').replaceChildren(new Option('Все группы',''),...actual.map(g=>new Option('Группа '+(g+1),g)));}
 function renderList(){const query=$('host-search').value.trim().toLocaleLowerCase('ru'),filter=$('group-filter').value;const rows=workflow.directory(state.dataset.filter((h,i)=>workflow.matches(h,query)&&(filter===''||groups[i]===Number(filter))));$('directory-count').textContent=rows.length;$('host-list').replaceChildren(...rows.slice(0,limit).map(host=>{const button=element('button');button.type='button';button.setAttribute('aria-pressed',String(host===selected));const meta=workflow.annotation(host.id);button.title=[host.id,meta.label,...meta.tags].filter(Boolean).join(' · ');button.dataset.status=meta.status;button.dataset.watch=String(meta.watch);const dot=element('i',undefined,'dot');dot.dataset.group=groups[state.dataset.indexOf(host)];button.append(dot,element('span',(meta.watch?'★ ':'')+hostName(host.id)));button.addEventListener('click',()=>select(host.id));return button;}));if(!rows.length)$('host-list').append(element('p','Устройства не найдены. Измените поиск или группу.','small'));$('more-hosts').hidden=rows.length<=limit;}
 function renderNeighbors(){const traffic=mode==='traffic',rows=traffic?contacts(state,selected.id).slice(0,5).map(c=>({...c,host:state.dataset.find(h=>h.id===c.id)})):nearest().slice(0,5);$('neighbors-title').textContent=traffic?'Наблюдавшиеся контакты':'Рядом по поведению';$('neighbors').replaceChildren(...rows.map(n=>{const b=element('button');b.type='button';b.append(element('span',hostName(n.host.id)),element('span',traffic?n.flows+' зап.':n.score.toFixed(3)));b.addEventListener('click',()=>select(n.host.id));return b;}));if(!rows.length)$('neighbors').append(element('p','В этом окне контактов не наблюдалось.','small'));}
 function keyChoose(node,choose){node.addEventListener('click',()=>{if(!dragged)choose();});node.addEventListener('keydown',e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();choose();}});}
 function fit(text,max=18){return text.length>max?text.slice(0,max-1)+'…':text;}
 function contactRows(){
  const byHost=new Map(), id=selected.id;
  for(const edge of state.connections||[]){
   if(edge.src!==id&&edge.dst!==id)continue;
   const outgoing=edge.src===id, other=outgoing?edge.dst:edge.src;
   if(other===id)continue;
   const row=byHost.get(other)||{id:other,out:0,in:0,sent:0,received:0};
   if(outgoing){row.out+=edge.flows;row.sent+=edge.bytes;}
   else{row.in+=edge.flows;row.received+=edge.bytes;}
   byHost.set(other,row);
  }
  const query=$('contact-search').value.trim().toLocaleLowerCase('ru');
  const order=$('contact-order').value;
  const score=row=>order==='bytes'?row.sent+row.received:row.out+row.in;
  const rows=[...byHost.values()].filter(row=>row.id.toLocaleLowerCase('ru').includes(query));
  rows.sort((a,b)=>(order==='name'?0:score(b)-score(a))||a.id.localeCompare(b.id,'ru'));
  return {rows,total:byHost.size};
 }
 function renderContactTable(){
  if(!selected)return;
  const {rows,total}=contactRows();
  $('contacts-title').textContent='Контакты · '+hostNameWithId(selected.id);
  $('contacts-summary').textContent=state.kind!=='flows'?'Готовые векторы не содержат записей обмена. Загрузите CSV со столбцами src и dst.':total?(rows.length?'Показано '+rows.length+' из '+total+' контактов.':'По этому запросу контакты не найдены.'):'Для выбранного устройства в файле нет записей обмена.';
  $('export-contacts').disabled=!rows.length;
  const formatter=new Intl.NumberFormat('ru');
  $('contact-table-body').replaceChildren(...rows.map(row=>{
   const tr=element('tr'), head=element('th'), button=element('button',hostNameWithId(row.id),'quiet');
   head.scope='row';button.type='button';
   button.addEventListener('click',()=>{select(row.id);$('contact-search').value='';renderContactTable();});
   head.append(button);tr.append(head,...[row.out,row.in,row.sent,row.received].map(value=>element('td',formatter.format(value))));
   return tr;
  }));
 }
 function exportContacts(){
  const {rows}=contactRows();
  if(!rows.length)return;
  const encode=id=>'"'+(/^[=+@\-\s]/.test(id)?"'":"")+id.replaceAll('"','""')+'"';
  const lines=rows.map(row=>[encode(row.id),row.out,row.in,row.sent,row.received].join(','));
  download(['host_id,outgoing_rows,incoming_rows,sent_bytes,received_bytes',...lines].join('\n'),
   'corgcode-contacts.csv','text/csv;charset=utf-8');
 }
 function drawNetwork(){const root=$('network'),mobile=innerWidth<700,width=mobile?360:760,height=mobile?600:470;root.setAttribute('viewBox',`${pan.x} ${pan.y} ${width/zoom} ${height/zoom}`);let center=mobile?[180,280]:[380,225],spots=mobile?[[90,90],[270,90],[90,450],[270,450],[180,550]]:[[140,85],[620,85],[130,350],[630,350],[380,420]];const rows=mode==='traffic'?contacts(state,selected.id).slice(0,5).map(c=>({...c,host:state.dataset.find(h=>h.id===c.id)})):nearest().slice(0,5);
 if(rows.length===1){
   center=mobile?[180,405]:[520,235];
   spots=mobile?[[180,195]]:[[240,235]];
 }else if(rows.length===2){
   center=mobile?[180,300]:[380,255];
   spots=mobile?[[180,125],[180,475]]:[[155,150],[605,150]];
 }
 const fragment=document.createDocumentFragment();rows.forEach((n,i)=>{const p=spots[i],x=(p[0]+center[0])/2,y=(p[1]+center[1])/2;fragment.append(svg('path',{d:`M ${center[0]} ${center[1]} L ${p[0]} ${p[1]}`,class:'edge'}),svg('text',{x,y,'text-anchor':'middle',class:'edge-value'},mode==='traffic'?n.flows+' зап.':n.score.toFixed(3)));});function node(host,p,chosen){const g=svg('g',{class:'node'+(chosen?' selected':''),transform:`translate(${p[0]} ${p[1]})`,role:'button',tabindex:0,'aria-label':`Выбрать ${hostNameWithId(host.id)}`});g.append(svg('title',{},hostNameWithId(host.id)),svg('rect',{x:-70,y:-29,width:140,height:58,rx:13}),svg('text',{x:0,y:-1,'text-anchor':'middle'},fit(hostName(host.id),16)),svg('text',{x:0,y:18,class:'node-sub','text-anchor':'middle'},chosen?'Выбрано':'Группа '+(groups[state.dataset.indexOf(host)]+1)));keyChoose(g,()=>select(host.id));fragment.append(g);}rows.forEach((n,i)=>node(n.host,spots[i],false));node(selected,center,true);if(!rows.length)fragment.append(svg('text',{x:width/2,y:mobile?450:350,'text-anchor':'middle',class:'empty-label'},'Нет контактов в этом окне'));root.replaceChildren(fragment);$('map-help').textContent=mode==='traffic'
   ? 'Здесь видно, с кем общалось устройство. Число на линии — сколько записей обмена есть в файле.'
   : 'Чем ближе число к 1, тем больше похоже поведение устройств. ' +
     (state.kind === 'flows' ? 'Чтобы увидеть, с кем устройство общалось, переключитесь на «Реальные контакты».' : 'В этом файле есть только профили устройств.');$('mode-similar').setAttribute('aria-pressed',String(mode==='similar'));$('mode-traffic').setAttribute('aria-pressed',String(mode==='traffic'));}
 function drawClusters(){const root=$('cluster-map'),xs=projection.map(p=>p.x),ys=projection.map(p=>p.y),minX=Math.min(...xs),minY=Math.min(...ys),dx=Math.max(...xs)-minX,dy=Math.max(...ys)-minY;const fragment=document.createDocumentFragment();fragment.append(svg('path',{d:'M 35 250 H 815 M 425 30 V 470',class:'axis'}));projection.forEach((p,i)=>{const cx=dx?45+(p.x-minX)/dx*760:425,cy=dy?45+(p.y-minY)/dy*410:250;const c=svg('circle',{cx,cy,r:state.dataset[i]===selected?10:6,'data-group':groups[i],class:state.dataset[i]===selected?'selected':'',role:'button',tabindex:0,'aria-label':`${p.hostId}, группа ${groups[i]+1}`});c.append(svg('title',{},p.hostId));keyChoose(c,()=>{select(p.hostId);view('devices');});fragment.append(c);});root.replaceChildren(fragment);const actual=[...new Set(groups)].sort((a,b)=>a-b);if(!actual.includes(activeGroup))activeGroup=actual[0];$('group-list').replaceChildren(...actual.map(g=>{const b=element('button');b.type='button';b.setAttribute('aria-pressed',String(g===activeGroup));const dot=element('i',undefined,'dot');dot.dataset.group=g;b.append(dot,element('span','Группа '+(g+1)),element('span',String(groups.filter(v=>v===g).length)));b.addEventListener('click',()=>{activeGroup=g;drawClusters();});return b;}));$('group-members').replaceChildren(...state.dataset.filter((h,i)=>groups[i]===activeGroup).map(h=>{const b=element('button',hostNameWithId(h.id));b.type='button';b.addEventListener('click',()=>{select(h.id);view('devices');});return b;}));}
 function renderCase(){
  if(!selected)return;
  const rows=math.trafficComparisonRows(selected,comparator),traffic=$('pair-traffic');
  traffic.replaceChildren(...rows.map(row=>element('li',row)));
  traffic.hidden=!rows.length;
  const options=state.dataset.filter(h=>h!==selected);$('compare-host').replaceChildren(...options.map(h=>new Option(hostNameWithId(h.id),h.id)));$('compare-host').value=comparator.id;$('case-title').textContent='Разбор · '+hostNameWithId(selected.id);$('note-host').textContent=hostNameWithId(selected.id);$('operator-note').value=Object.prototype.hasOwnProperty.call(payload.notes,selected.id)?payload.notes[selected.id]:'';const score=math.cosine(selected,comparator);$('comparison-summary').textContent=`${selected.id} и ${comparator.id}: сходство ${score.toFixed(3)}. Ближе к 1 — ближе направления профилей.`;const a=selected.v.map(x=>x/selected.norm),b=comparator.v.map(x=>x/comparator.norm);let ids=a.map((_,i)=>i);if(state.kind==='flows'&&!state.model)ids=ids.slice(0,8);else ids.sort((i,j)=>Math.abs(a[j])+Math.abs(b[j])-Math.abs(a[i])-Math.abs(b[i]));ids=ids.slice(0,8);const max=Math.max(...ids.flatMap(i=>[Math.abs(a[i]),Math.abs(b[i])]),.001);const chart=$('feature-chart');chart.setAttribute('viewBox',`0 0 800 ${ids.length*45+30}`);const fragment=document.createDocumentFragment();ids.forEach((i,row)=>{const y=24+row*45;fragment.append(svg('text',{x:0,y:y+10,class:'feature-label'},state.kind==='flows'&&!state.model?featureNames[i]:'Координата '+i));for(const[value,cls,offset]of [[a[i],'bar-a',0],[b[i],'bar-b',16]]){const w=Math.abs(value)/max*340;fragment.append(svg('rect',{x:240,y:y+offset-2,width:w,height:10,rx:4,class:cls}),svg('text',{x:590,y:y+offset+7,class:'feature-value'},value.toFixed(3)));}});chart.replaceChildren(fragment);$('feature-caption').textContent=`Верхняя полоса — ${selected.id}, нижняя — ${comparator.id}. ${state.kind==='flows'&&!state.model?'Показаны восемь простых признаков':'Показаны восемь наиболее выраженных координат'}. Сравниваются L2-нормированные значения; длина полосы — модуль, знак указан числом.`;const other=new Set(contacts(state,comparator.id).map(c=>c.id)),shared=contacts(state,selected.id).filter(c=>other.has(c.id));$('shared-copy').textContent=state.kind==='flows'?shared.length?`В файле найдено общих собеседников: ${shared.length}. Нажмите, чтобы проверить их окружение.`:'Общих контактов в этих записях нет.':'Векторы не содержат записи обмена. Для проверки общих контактов загрузите CSV трафика.';$('shared-contacts').replaceChildren(...shared.map(c=>{const button=element('button',c.id+' ↗');button.type='button';button.addEventListener('click',()=>{mode='traffic';select(c.id);view('devices');});return button;}));}
 function renderIncident() {
   const panel = $('incident-context');
   panel.hidden = !payload.sample || payload.csv !== incidentCurrent.csv ||
     payload.baseline !== incidentBaseline.csv;
   if (panel.hidden) return;
   const max = Math.max(...incidentBaseline.counts, ...incidentCurrent.counts);
   const timeline = laterTime.map((time, i) => {
     const entry = element('li', undefined, 'incident-bucket');
     const bars = element('div', undefined, 'incident-bars');
     const before = incidentBaseline.counts[i], after = incidentCurrent.counts[i];
     bars.setAttribute('aria-hidden', 'true');
     for (const [count, style] of [[before, 'before'], [after, 'after']]) {
       const bar = element('span', undefined, style);
       bar.style.height = `${100 * count / max}%`;
       bars.append(bar);
     }
     entry.classList.toggle('incident-start', time === '10:40');
     entry.append(
       bars,
       element('span', time, 'incident-time'),
       element('span', `${before} → ${after}`, 'incident-count'),
     );
     entry.setAttribute('aria-label', `${earlierTime[i]}: ${before} записей, ${time}: ${after} записей`);
     return entry;
   });
   $('incident-timeline').replaceChildren(...timeline);
 }
 function renderChanges(){if(!payload)return;renderIncident();$('changes-empty').hidden=!!payload.baseline;$('changes-content').hidden=!payload.baseline;if(!payload.baseline)return;try{const report=math.compareWindows(math.parseCSV(payload.baseline),state.raw||state,Number($('group-count').value));$('changes-summary').textContent=`Общих: ${report.changes.length} · новых: ${report.newHosts.length} · отсутствуют: ${report.missingHosts.length}`;const filtered=workflow.changeRows(report);
 if(report.kind==='flows'){
   const hostsWithNewContacts=report.changes.filter(change=>change.added.length>0).length;
   $('changes-summary').textContent+=` · с новыми контактами: ${hostsWithNewContacts}`;
   filtered.sort((a,b)=>b.added.length-a.added.length||b.removed.length-a.removed.length||a.similarity-b.similarity);
 }$('change-list').replaceChildren(...filtered.map(c=>{const row=element('div',undefined,'change-row');row.append(element('strong',hostNameWithId(c.id)));for(const[value,label]of [[c.similarity.toFixed(3),'сходство до / после'],[`${c.beforeGroup+1} → ${c.afterGroup+1}`,'группа первого окна'],[state.kind==='flows'?`+${c.added.length} / −${c.removed.length}`:'Нет записей','изменение контактов']]){const cell=element('span',value);cell.append(element('small',label));row.append(cell);}const button=element('button','До и после');button.type='button';button.addEventListener('click',()=>{renderChangeDetail(c);$('change-detail').scrollIntoView?.({block:'nearest'});status(state.kind==='flows'?`Новые контакты: ${c.added.join(', ')||'нет'}. Исчезнувшие: ${c.removed.join(', ')||'нет'}.`:`Профиль ${c.id} изменился: сходство ${c.similarity.toFixed(3)}.`);});row.append(button);return row;}));$('change-detail').hidden=!filtered.length;if(filtered.length)renderChangeDetail(filtered[0]);else $('change-list').append(element('p','Нет устройств по выбранным условиям. Измените фильтр.','small'));}catch(error){$('changes-content').hidden=true;$('changes-empty').hidden=false;status('Окна нельзя сравнить: '+error.message,true);}}
 function renderChangeDetail(change){const root=$('change-detail');root.replaceChildren(element('h3',hostNameWithId(change.id)+' · первое → второе окно'));if(state.kind==='flows'){const measures=element('div',undefined,'period-measures');for(const[key,label]of [['outgoing','Исходящих записей'],['incoming','Входящих записей'],['sentBytes','Отправлено байт'],['receivedBytes','Получено байт']]){const row=element('div');row.append(element('span',label),element('strong',change.before.activity[key].toLocaleString('ru-RU')+' → '+change.after.activity[key].toLocaleString('ru-RU')));measures.append(row);}root.append(measures);for(const[label,ids]of [['Новые контакты',change.added],['Больше не наблюдались',change.removed]]){root.append(element('h4',label));const links=element('div',undefined,'chips');if(!ids.length)links.append(element('p','Нет','small'));ids.forEach(id=>{const host=state.dataset.find(h=>h.id===id);if(!host){links.append(element('span',id,'small'));return;}const b=element('button',hostNameWithId(id));b.type='button';b.addEventListener('click',()=>{mode='traffic';select(id);view('devices');});links.append(b);});root.append(links);}
 const inspect=element('button','Проверить контакты '+change.id,'glass');
 inspect.type='button';
 inspect.addEventListener('click',()=>{select(change.id);view('contacts');});
 root.append(inspect);
 }else root.append(element('p',`Сходство двух состояний ${change.similarity.toFixed(3)}. Группа: ${change.beforeGroup+1} → ${change.afterGroup+1}. Готовые векторы не позволяют показать исходные объёмы и контакты.`,'small'));}
 async function savedList(){const rows=await api('/api/workspaces');$('saved-workspaces').replaceChildren(...rows.map(row=>{const button=element('button',row.name);button.type='button';button.title=row.name;button.setAttribute('aria-pressed',String(row.id===workspaceId));button.addEventListener('click',()=>openSaved(row.id));return button;}));if(!rows.length)$('saved-workspaces').append(element('p','Сохранённые наборы появятся здесь.','small'));renderWorkspaceDialog(rows);return rows;}
 function renderWorkspaceDialog(rows){$('workspace-dialog-list').replaceChildren(...rows.map(row=>{const button=element('button',row.name);button.type='button';button.addEventListener('click',()=>{$('workspaces-dialog').close();openSaved(row.id);});return button;}));if(!rows.length)$('workspace-dialog-list').append(element('p','Пока нет сохранённых наборов. Загрузите CSV и нажмите «Сохранить».','small'));}
 async function openSaved(id){if(!(await preserve()))return;const request=++openRequest,currentIntent=++intent;try{const data=await api('/api/workspaces/'+id);if(request!==openRequest||currentIntent!==intent)return;const parsed=math.parseCSV(data.csv);importRequest++;baselineRequest++;install(parsed,data,id);await savedList();}catch(error){if(currentIntent!==intent)return;status('Не удалось открыть набор: '+error.message,true);}}
 async function save(){if(isGuest()){openAccount(true);return false;}if(saving){status('Подождите завершения сохранения.');return false;}saving=true;const currentRevision=revision,oldId=workspaceId,originalPayload=payload,data=JSON.parse(JSON.stringify({...payload,name:$('workspace-name').value}));$('save-state').textContent='Сохраняем…';$('save-workspace').disabled=true;$('save-note').disabled=true;try{const result=await api('/api/workspaces'+(oldId?'/'+oldId:''),{method:oldId?'PUT':'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(data)});if(payload===originalPayload&&workspaceId===oldId){workspaceId=result.id;$('save-state').textContent=revision===currentRevision?'Сохранено на компьютере':'Есть несохранённые изменения';}await savedList();return true;}catch(error){$('save-state').textContent='Есть несохранённые изменения';status('Не удалось сохранить: '+error.message,true);if(isGuest())openAccount(true);return false;}finally{saving=false;$('save-workspace').disabled=false;$('save-note').disabled=false;}}
 async function preserve(){if(isGuest()&&$('save-state').textContent.includes('несохранённые'))return window.confirm('Перейти к другому набору без сохранения текущих изменений?');while($('save-state').textContent.includes('несохранённые'))if(!(await save()))return false;return true;}
 async function importFile(file,baseline=false){if(!file)return;const currentIntent=baseline?intent:++intent,request=baseline?++baselineRequest:++importRequest;try{if(file.size>3*1024*1024)throw Error('Поддерживается CSV до 3 МБ.');const csv=await file.text();if(request!==(baseline?baselineRequest:importRequest)||currentIntent!==intent)return;const data=math.parseCSV(csv);if(baseline){math.compareWindows(data,state.raw||state,Number($('group-count').value));payload.baseline=csv;$('baseline-name').textContent=file.name;dirty();renderChanges();status('Первое окно загружено. Можно сравнивать изменения.');}else{if(!(await preserve()))return;if(currentIntent!==intent)return;
   const stored={name:file.name.replace(/\.csv$/i,''),csv,baseline:'',notes:{},sample:false};
   if(document.body.dataset.mlEnabled==='true'&&!isGuest()&&data.kind==='flows'){
    status('Вычисляем эмбеддинги…');
    const result=await api('/api/inference/embeddings',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({csv})});
    if(currentIntent!==intent||request!==importRequest)return;
    stored.inferenceId=result.jobId;
    stored.inference=result;
   }
   openRequest++;baselineRequest++;install(data,stored);dirty();}}catch(error){if(request===(baseline?baselineRequest:importRequest)&&currentIntent===intent)status('Файл не загружен: '+error.message+' Рабочий набор сохранён.',true);}}
 function download(text,name,type){const blob=new Blob([text],{type}),url=URL.createObjectURL(blob),a=element('a');a.href=url;a.download=name;document.body.append(a);a.click();a.remove();setTimeout(()=>URL.revokeObjectURL(url),1000);}
 function exportCase(){const report={product:'CorgCode',dataset:payload.name,sample:payload.sample,device:selected.id,group:groups[state.dataset.indexOf(selected)]+1,activity:selected.activity||null,neighbors:nearest().map(n=>({id:n.host.id,cosine:n.score})),observedContacts:contacts(state,selected.id),comparedWith:comparator.id,comparisonCosine:math.cosine(selected,comparator),operatorNote:Object.prototype.hasOwnProperty.call(payload.notes,selected.id)?payload.notes[selected.id]:'',annotation:workflow.annotation(selected.id),limitation:'Профиль и группа не определяют роль или наличие атаки. Координаты карты условны.'};download(JSON.stringify(report,null,2),'corgcode-case.json','application/json;charset=utf-8');status('Разбор сохранён в JSON: значения, соседи, контакты и заметка.');}
 const sidebarToggle = $('toggle-sidebar');
 sidebarToggle.addEventListener('click', () => {
   const sidebar = $('workspace-sidebar');
   sidebar.hidden = !sidebar.hidden;
   sidebarToggle.setAttribute('aria-expanded', String(!sidebar.hidden));
   const label = sidebar.hidden ? 'Показать меню' : 'Свернуть меню';
   sidebarToggle.setAttribute('aria-label', label);
   sidebarToggle.title = label;
   sidebarToggle.firstElementChild.textContent = sidebar.hidden ? '›' : '‹';
 });
 const workspaceActions = document.querySelector('.workspace-actions');
 const actionSummary = workspaceActions.querySelector('summary');
 function closeWorkspaceActions() {
   workspaceActions.open = false;
   actionSummary.focus();
 }
 workspaceActions.querySelectorAll('button').forEach(button => {
   button.addEventListener('click', closeWorkspaceActions);
 });
 workspaceActions.addEventListener('keydown', event => {
   if (event.key !== 'Escape' || !workspaceActions.open) return;
   event.preventDefault();
   closeWorkspaceActions();
 });
 document.querySelectorAll('[data-view]').forEach(b=>b.addEventListener('click',()=>view(b.dataset.view)));
 $('host-search').addEventListener('input',()=>{limit=30;renderList();});$('group-filter').addEventListener('change',()=>{limit=30;renderList();});$('more-hosts').addEventListener('click',()=>{limit+=30;renderList();});
 $('group-count').addEventListener('change',()=>{payload.settings.groupCount=Number($('group-count').value);dirty();groups=math.group(state.dataset,Number($('group-count').value));renderGroupOptions();select(selected.id);drawClusters();renderChanges();workflow.renderWatch();});
 for(const name of ['similar','traffic'])$('mode-'+name).addEventListener('click',()=>{mode=name;zoom=1;pan={x:0,y:0};drawNetwork();renderNeighbors();});
 $('compare-host').addEventListener('change',()=>{comparator=state.dataset.find(h=>h.id===$('compare-host').value);renderCase();});$('operator-note').addEventListener('input',()=>{payload.notes={...payload.notes,[selected.id]:$('operator-note').value};dirty();});$('workspace-name').addEventListener('input',()=>{payload.name=$('workspace-name').value;dirty();});
 const helpKey = 'corgcode-context-help';
 function showHelp(visible) {
   document.body.toggleAttribute('data-context-help', visible);
   $('start-tutorial').setAttribute('aria-pressed', String(visible));
   $('start-tutorial').textContent = visible ? 'Скрыть подсказки' : 'Подсказки';
   try { localStorage.setItem(helpKey, visible ? 'on' : 'off'); } catch {}
 }
 $('start-tutorial').addEventListener('click', () => {
   showHelp(!document.body.hasAttribute('data-context-help'));
 });
 try { showHelp(localStorage.getItem(helpKey) !== 'off'); } catch { showHelp(true); }
 $('change-scenario').addEventListener('click',openChangeScenario);
 $('compare-periods').addEventListener('click',()=>view('changes'));
 $('to-contacts').addEventListener('click',()=>view('contacts'));$('contact-search').addEventListener('input',renderContactTable);$('contact-order').addEventListener('change',renderContactTable);$('export-contacts').addEventListener('click',exportContacts);
 $('to-case').addEventListener('click',()=>view('case'));$('save-workspace').addEventListener('click',save);$('save-note').addEventListener('click',save);$('sample-button').addEventListener('click',async()=>{if(await preserve())openSample();});$('import-button').addEventListener('click',()=>$('current-file').click());$('baseline-button').addEventListener('click',()=>$('baseline-file').click());$('current-file').addEventListener('change',e=>{importFile(e.target.files[0]);e.target.value='';});$('baseline-file').addEventListener('change',e=>{importFile(e.target.files[0],true);e.target.value='';});$('export-case').addEventListener('click',exportCase);
 $('open-workspaces').addEventListener('click',()=>{if(isGuest()){openAccount();return;}$('workspaces-dialog').showModal();savedList().catch(error=>status(error.message,true));});$('close-workspaces').addEventListener('click',()=>$('workspaces-dialog').close());$('mobile-help').addEventListener('click',()=>{$('workspaces-dialog').close();$('help-dialog').showModal();});
 $('open-help').addEventListener('click',()=>$('help-dialog').showModal());$('close-help').addEventListener('click',()=>$('help-dialog').close());
 $('zoom-in').addEventListener('click',()=>{zoom=Math.min(4,zoom*1.25);drawNetwork();});$('zoom-out').addEventListener('click',()=>{zoom=Math.max(.5,zoom/1.25);drawNetwork();});$('zoom-reset').addEventListener('click',()=>{zoom=1;pan={x:0,y:0};drawNetwork();});
 const network=$('network');network.addEventListener('pointerdown',e=>{if(e.button!==0||e.pointerType==='touch')return;drag={x:e.clientX,y:e.clientY,pan:{...pan}};dragged=false;});network.addEventListener('pointermove',e=>{if(!drag)return;const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.hypot(dx,dy)>5)dragged=true;if(!dragged)return;network.setPointerCapture?.(e.pointerId);const rect=network.getBoundingClientRect(),width=innerWidth<700?360:760;pan={x:drag.pan.x-dx/rect.width*width/zoom,y:drag.pan.y-dy/rect.width*width/zoom};network.classList.add('is-dragging');drawNetwork();});const endDrag=()=>{drag=null;network.classList.remove('is-dragging');setTimeout(()=>dragged=false,0);};network.addEventListener('pointerup',endDrag);network.addEventListener('pointercancel',endDrag);let mobile=innerWidth<700;window.addEventListener('resize',()=>{const next=innerWidth<700;if(next!==mobile){mobile=next;zoom=1;pan={x:0,y:0};drawNetwork();}});
 window.addEventListener('beforeunload',e=>{if($('save-state').textContent.includes('несохранённые')){e.preventDefault();e.returnValue='';}});
 workflow=window.createCorgWorkflow(()=>({state,selected,comparator,groups,mode,payload,math,element,select,view,dirty,status,download,nearest,contacts,renderList,renderChanges,hostName}));
 if(siteMode){
  $('site-login').addEventListener('click',()=>openAccount());
  $('site-register').addEventListener('click',()=>openAccount(false,true));
  $('switch-account-mode').addEventListener('click',()=>setAccountMode($('account-form').action.endsWith('/lab/login/')));
  $('close-account').addEventListener('click',()=>$('account-dialog').close());
  $('account-form').addEventListener('submit',signIn);
 }
 function followSection(){
  const section=location.hash.slice(1);
  if(section==='incident'){openChangeScenario();return;}
  if(section==='soc-scenarios'){showHelp(true);view('devices');}
  if(['devices','groups','contacts','changes','watch','case'].includes(section))view(section);
 }
 openSample();
 followSection();
 window.addEventListener('hashchange',followSection);
 const startupIntent=intent;
 if(!isGuest())savedList().then(rows=>{
  if(rows.length&&location.hash!=='#incident'&&intent===startupIntent&&revision===0)openSaved(rows[0].id);
 }).catch(error=>{
  if(!isGuest())status('Локальное хранилище недоступно: '+error.message,true);
 });
})();
