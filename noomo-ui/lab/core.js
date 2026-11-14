/* Local CSV/profile math reused from pt-page-lab.js; deterministic cosine grouping. */
(()=>{
function csvRows(text) {
  const first = text.replace(/^\uFEFF/, "").split(/\r?\n/, 1)[0];
  const delimiter = [";", ",", "\t"].sort(
    (a, b) => first.split(b).length - first.split(a).length,
  )[0];
  const rows = [];
  let row = [],
    cell = "",
    quoted = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (ch === '"') {
      if (quoted && text[i + 1] === '"') {
        cell += '"';
        i++;
      } else if (quoted || cell.trim() === "") quoted = !quoted;
      else throw Error("Кавычка внутри незаключённого поля CSV.");
    } else if (ch === delimiter && !quoted) {
      row.push(cell.trim());
      cell = "";
    } else if ((ch === "\n" || ch === "\r") && !quoted) {
      if (ch === "\r" && text[i + 1] === "\n") i++;
      row.push(cell.trim());
      if (row.some((x) => x !== "")) rows.push(row);
      row = [];
      cell = "";
    } else cell += ch;
    if (rows.length > 10001)
      throw Error("Для браузерного примера допустимо до 10 000 строк.");
  }
  if (quoted) throw Error("В CSV не закрыты кавычки.");
  row.push(cell.trim());
  if (row.some((x) => x !== "")) rows.push(row);
  if(rows.length>10001)throw Error("Для браузерного примера допустимо до 10 000 строк.");
  if (rows.length < 2)
    throw Error("Нужны заголовок и хотя бы одна строка данных.");
  rows[0][0] = rows[0][0].replace(/^\uFEFF/, "");
  const columns = rows[0].map((x) => x.trim().toLowerCase());
  if (new Set(columns).size !== columns.length)
    throw Error("Имена столбцов повторяются.");
  for (const r of rows.slice(1))
    if (r.length !== columns.length)
      throw Error("Число полей в строках не совпадает с заголовком.");
  return { columns, rows: rows.slice(1) };
}
function parseCSV(text) {
  if(typeof text!=='string'||text.length>3*1024*1024)throw Error('Поддерживается CSV до 3 МБ.');
  const parsed = csvRows(text),
    { columns, rows } = parsed;
  const hi = columns.indexOf("host_id");
  if (hi >= 0) {
    const dims = columns
      .map((name, i) => ({ name, i }))
      .filter((x) => /^emb_\d+$/.test(x.name))
      .sort((a, b) => Number(a.name.slice(4)) - Number(b.name.slice(4)));
    if (dims.some((d,i)=>d.name !== 'emb_'+i)) throw Error('Координаты должны идти подряд от emb_0.');
    if (!dims.length || dims.length > 1024)
      throw Error("Ожидаются от 1 до 1024 столбцов emb_0, emb_1 и далее.");
    if (rows.length > 500)
      throw Error("Для интерактивного примера допустимо до 500 хостов.");
    const ids = new Set();
    const dataset = rows.map((row) => {
      const id = row[hi];
      if (!id || ids.has(id))
        throw Error("ID хостов должны быть заполнены и уникальны.");
      ids.add(id);
      const v = dims.map((d) => {
        if (row[d.i] === "") throw Error("В векторе есть пустое значение.");
        const x = Number(row[d.i]);
        if (!Number.isFinite(x))
          throw Error("Векторы должны содержать только конечные числа.");
        return x;
      });
      const norm = Math.hypot(...v);
      if (!norm || !Number.isFinite(norm))
        throw Error(
          "Нулевые или слишком большие векторы нельзя сравнить по косинусу.",
        );
      return { id, v, norm };
    });
    if (dataset.length < 2)
      throw Error("Для сравнения нужны хотя бы два хоста.");
    return {
      dataset,
      stats: { hosts: dataset.length, dimensions: dims.length },
      kind: "vectors",
    };
  }
  const si = ["src", "source", "src_host"]
      .map((k) => columns.indexOf(k))
      .find((i) => i >= 0),
    di = ["dst", "target", "dst_host"]
      .map((k) => columns.indexOf(k))
      .find((i) => i >= 0);
  if (si === undefined || di === undefined)
    throw Error("Нужны столбцы src и dst либо host_id и emb_0, emb_1…");
  const bi = columns.indexOf("bytes"),
    pi = columns.indexOf("packets"),
    porti = columns.indexOf("dst_port"),
    proti = columns.indexOf("proto");
  const map = new Map(),links=new Map();
  const get = (id) => {
    if (!id) throw Error("src и dst не должны быть пустыми.");
    if (!map.has(id))
      map.set(id, { id, v: new Array(20).fill(0), peers: new Set() });
    if (map.size > 500) throw Error("В примере поддерживается до 500 хостов.");
    return map.get(id);
  };
  let totalBytes = 0;
  const numeric = (row, index) => {
    if (index < 0) return 0;
    const x = Number(row[index]);
    if (!Number.isFinite(x) || x < 0)
      throw Error("bytes и packets должны быть неотрицательными числами.");
    return x;
  };
  for (const row of rows) {
    const a = get(row[si]),
      b = get(row[di]),
      bytes = numeric(row, bi),
      packets = numeric(row, pi);
    a.v[0]++;
    b.v[1]++;
    a.v[2] += bytes;
    b.v[3] += bytes;
    a.v[4] += packets;
    b.v[5] += packets;
    a.peers.add(b.id);
    b.peers.add(a.id);
    const service =
      (proti >= 0 ? row[proti] : "") + ":" + (porti >= 0 ? row[porti] : "");
    let hash = 0;
    for (const ch of service) hash = (hash * 31 + ch.charCodeAt(0)) >>> 0;
    a.v[8 + (hash % 6)]++;
    b.v[14 + (hash % 6)]++;
    const key=JSON.stringify([a.id,b.id]);
    if(!links.has(key))links.set(key,{src:a.id,dst:b.id,flows:0,bytes:0});
    const link=links.get(key);link.flows++;link.bytes+=bytes;
    totalBytes += bytes;
  }
  if(!Number.isFinite(totalBytes)||[...map.values()].some(h=>h.v.some(x=>!Number.isFinite(x))))throw Error("Суммарные объёмы слишком велики для расчёта.");
  const dataset = [...map.values()].map((h) => {
    const activity={outgoing:h.v[0],incoming:h.v[1],sentBytes:h.v[2],receivedBytes:h.v[3],contacts:h.peers.size};
    h.v[6] = h.peers.size;
    h.v[7] = h.v[0] + h.v[1];
    h.v = h.v.map(Math.log1p);
    return { id: h.id, v: h.v,activity };
  });
  if (dataset.length < 2) throw Error("Для сравнения нужны хотя бы два хоста.");
  const scales = dataset[0].v.map((_, i) =>
    Math.max(1, ...dataset.map((h) => h.v[i])),
  );
  dataset.forEach((h) => {
    h.v = h.v.map((x, i) => x / scales[i]);
    h.norm = Math.hypot(...h.v);
  });
  return {
    dataset,
    stats: {
      hosts: dataset.length,
      flows: rows.length,
      bytes: totalBytes,
      dimensions: 20,
    },
    kind: "flows",
    connections:[...links.values()],
  };
}
function project(dataset) {
  const n = dataset.length,
    d = dataset[0].v.length,
    mean = new Array(d).fill(0);
  dataset.forEach((h) => h.v.forEach((x, j) => (mean[j] += x / n)));
  const x = dataset.map((h) => h.v.map((v, j) => v - mean[j]));
  function axis(previous) {
    let v = Array.from({ length: d }, (_, j) =>
      Math.sin(j * 1.7 + (previous ? 1.1 : 0.4)),
    );
    for (let iter = 0; iter < 18; iter++) {
      let next = new Array(d).fill(0);
      for (const row of x) {
        const dot = row.reduce((s, z, j) => s + z * v[j], 0);
        row.forEach((z, j) => (next[j] += z * dot));
      }
      if (previous) {
        const dot = next.reduce((s, z, j) => s + z * previous[j], 0);
        next = next.map((z, j) => z - dot * previous[j]);
      }
      const norm = Math.hypot(...next);
      if (norm < 1e-12) return new Array(d).fill(0);
      v = next.map((z) => z / norm);
    }
    return v;
  }
  const a = axis(null),
    b = axis(a);
  return x.map((row, i) => ({
    x: row.reduce((s, z, j) => s + z * a[j], 0),
    y: row.reduce((s, z, j) => s + z * b[j], 0),
    hostId: dataset[i].id,
  }));
}

const cosine=(a,b)=>Math.max(-1,Math.min(1,a.v.reduce((sum,x,i)=>sum+x*b.v[i],0)/(a.norm*b.norm)));
function group(dataset,k=3){
 const rows=dataset.map(h=>h.v.map(x=>x/h.norm));k=Math.min(k,rows.length);
 const centers=[rows[0].slice()];
 while(centers.length<k){
  let best=-1,index=0;
  rows.forEach((v,i)=>{const dist=1-Math.max(...centers.map(c=>v.reduce((s,x,j)=>s+x*c[j],0)));if(dist>best){best=dist;index=i;}});
  centers.push(rows[index].slice());
 }
 let labels=new Array(rows.length).fill(0);
 for(let iteration=0;iteration<20;iteration++){
  const next=rows.map(v=>{let best=-Infinity,index=0;centers.forEach((c,i)=>{const score=v.reduce((s,x,j)=>s+x*c[j],0);if(score>best){best=score;index=i;}});return index;});
  const changed=next.some((v,i)=>v!==labels[i]);labels=next;
  for(let c=0;c<k;c++){
   const sum=new Array(rows[0].length).fill(0);rows.forEach((v,i)=>{if(labels[i]===c)v.forEach((x,j)=>sum[j]+=x);});
   const norm=Math.hypot(...sum);if(norm>1e-12)centers[c]=sum.map(x=>x/norm);
  }
  if(!changed&&iteration>0)break;
 }
 return labels;
}
window.PTLab={parseCSV,project,cosine,group};
})();
