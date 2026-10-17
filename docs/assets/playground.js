/* CSV playground. Runs fully in the browser, no external libraries and no upload to any server.
   Metric math (cosine top-k, consistency, Jaccard, nDCG) is a direct port of the previous playground. */
(function () {
  "use strict";
  var N = 300, REF_D = 24, KS = [5, 10, 20], MAX_BYTES = 8 * 1024 * 1024, NS = "http://www.w3.org/2000/svg";
  var statusEl, resultsEl;

  function rng(seed) { var s = (seed || 1) >>> 0; return function () { s = (1664525 * s + 1013904223) >>> 0; return s / 4294967296; }; }
  function normal(r) { var u = Math.max(r(), 1e-12), v = r(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }
  function makeReference() {
    var r = rng(42), rows = [];
    for (var i = 0; i < N; i++) {
      var role = i % 5, x = [];
      for (var j = 0; j < 5; j++) x.push(j === role ? 2.5 : 0);
      x.push(role / 4); x.push((4 - role) / 4); x.push(Math.sin(role * 1.2)); x.push(Math.cos(role * 1.2));
      while (x.length < REF_D) x.push(0.25 * normal(r) + 0.12 * role);
      rows.push(x);
    }
    return rows;
  }
  var REF = makeReference(), HOSTS = [];
  for (var h = 0; h < N; h++) HOSTS.push("h" + h);
  function cosine(a, b) { var d = 0, na = 0, nb = 0; for (var i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; } return (!na || !nb) ? 0 : d / Math.sqrt(na * nb); }
  function topK(X, k) {
    var out = [];
    for (var i = 0; i < X.length; i++) {
      var a = [];
      for (var j = 0; j < X.length; j++) if (i !== j) a.push([j, cosine(X[i], X[j])]);
      a.sort(function (x, y) { return y[1] - x[1]; });
      out.push(a.slice(0, k).map(function (z) { return z[0]; }));
    }
    return out;
  }
  var REF_NN = {}; KS.forEach(function (k) { REF_NN[k] = topK(REF, k); });
  function avg(a) { return a.reduce(function (x, y) { return x + y; }, 0) / a.length; }
  function metrics(X) {
    var res = {};
    KS.forEach(function (k) {
      var e = topK(X, k), cons = [], jac = [], nd = [], disc = [], ideal = 0, i;
      for (i = 0; i < k; i++) disc.push(1 / Math.log2(i + 2));
      for (i = 0; i < k; i++) ideal += (k - i) * disc[i];
      for (i = 0; i < N; i++) {
        var map = new Map(REF_NN[k][i].map(function (hh, rank) { return [hh, k - rank]; })), inter = 0, dcg = 0;
        e[i].forEach(function (hh, rank) { if (map.has(hh)) { inter++; dcg += map.get(hh) * disc[rank]; } });
        cons.push(inter / k); jac.push(inter / (2 * k - inter)); nd.push(dcg / ideal);
      }
      res[k] = { consistency: avg(cons), jaccard: avg(jac), ndcg: avg(nd) };
    });
    return res;
  }
  function randomEmbedding(d) { var r = rng(7), X = []; for (var i = 0; i < N; i++) { var row = []; for (var j = 0; j < d; j++) row.push(normal(r)); X.push(row); } return X; }

  /* minimal RFC 4180 parser: quotes, escaped quotes, CRLF, BOM */
  function parseCsv(text) {
    if (text.charCodeAt(0) === 0xFEFF) text = text.slice(1);
    var rows = [], row = [], f = "", q = false, i = 0, c;
    for (; i < text.length; i++) {
      c = text[i];
      if (q) { if (c === '"') { if (text[i + 1] === '"') { f += '"'; i++; } else q = false; } else f += c; }
      else if (c === '"') q = true;
      else if (c === ",") { row.push(f); f = ""; }
      else if (c === "\n" || c === "\r") { if (c === "\r" && text[i + 1] === "\n") i++; row.push(f); f = ""; if (row.length > 1 || row[0] !== "") rows.push(row); row = []; }
      else f += c;
    }
    if (q) throw new Error("Незакрытая кавычка в CSV.");
    row.push(f); if (row.length > 1 || row[0] !== "") rows.push(row);
    return rows;
  }
  function validate(table) {
    if (table.length < 2) throw new Error("CSV пуст.");
    var head = table[0].map(function (s) { return s.trim(); });
    if (head.indexOf("host_id") < 0) throw new Error("Отсутствует обязательный столбец host_id.");
    var embIdx = [];
    head.forEach(function (c, i) { if (c.indexOf("emb_") === 0) embIdx.push(i); });
    if (!embIdx.length) throw new Error("Нет столбцов emb_*.");
    var hi = head.indexOf("host_id"), by = new Map();
    for (var r = 1; r < table.length; r++) {
      var id = table[r][hi].trim();
      if (by.has(id)) throw new Error("Встречаются повторяющиеся host_id.");
      by.set(id, table[r]);
    }
    if (by.size !== N || HOSTS.some(function (x) { return !by.has(x); })) throw new Error("Для демонстрации нужно ровно " + N + " хостов h0…h" + (N - 1) + ".");
    var X = HOSTS.map(function (x) { return embIdx.map(function (i) { var s = (by.get(x)[i] || "").trim(); return s === "" ? NaN : Number(s); }); });
    if (X.some(function (row) { return row.some(function (v) { return !Number.isFinite(v); }); })) throw new Error("В embeddings обнаружены пропуски, нечисловые значения, NaN или бесконечность.");
    return { X: X, dim: embIdx.length };
  }

  function fmt(v) { return v.toFixed(3).replace(".", ","); }
  function svgEl(t, a, txt) { var n = document.createElementNS(NS, t); for (var k in a) n.setAttribute(k, a[k]); if (txt != null) n.textContent = txt; return n; }
  function css(n) { return getComputedStyle(document.documentElement).getPropertyValue(n).trim(); }

  function renderTable(items) {
    items.sort(function (a, b) { return b.m[10].consistency - a.m[10].consistency; });
    var t = document.getElementById("board"); t.textContent = "";
    var hr = document.createElement("tr");
    ["Источник", "Согл.@5", "Согл.@10", "Согл.@20", "Jaccard@10", "nDCG@10"].forEach(function (s, i) { var th = document.createElement("th"); th.textContent = s; if (i) th.className = "num"; hr.appendChild(th); });
    t.appendChild(hr);
    items.forEach(function (x) {
      var tr = document.createElement("tr");
      [x.name, fmt(x.m[5].consistency), fmt(x.m[10].consistency), fmt(x.m[20].consistency), fmt(x.m[10].jaccard), fmt(x.m[10].ndcg)].forEach(function (s, i) { var td = document.createElement("td"); td.textContent = s; if (i) td.className = "num"; tr.appendChild(td); });
      t.appendChild(tr);
    });
  }
  function renderBars(items) {
    var host = document.getElementById("bars"); host.textContent = "";
    var W = Math.max(280, host.clientWidth || 600), rowH = 22, gap = 18, top = 8, left = 8;
    var H = top + items.length * (3 * rowH + gap) + 24, plotW = W - 80;
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Согласие соседей по значениям k" });
    var accent = css("--accent"), accent2 = css("--accent-2"), warn = css("--warn");
    var cols = [accent, accent2, warn];
    items.forEach(function (x, i) {
      var y = top + i * (3 * rowH + gap);
      svg.appendChild(svgEl("text", { x: left, y: y + 11 }, x.name));
      KS.forEach(function (k, j) {
        var v = x.m[k].consistency, yy = y + 16 + j * 16;
        svg.appendChild(svgEl("rect", { x: left, y: yy, width: Math.max(1, v * plotW), height: 12, fill: cols[j] }));
        svg.appendChild(svgEl("text", { x: left + v * plotW + 6, y: yy + 10 }, "k=" + k + ": " + fmt(v)));
      });
    });
    host.appendChild(svg);
  }
  function renderScatter(X) {
    var host = document.getElementById("scatter"); host.textContent = "";
    var W = Math.max(280, host.clientWidth || 600), H = Math.round(Math.min(420, W * 0.65)), pad = 30;
    var xs = X.map(function (v) { return v[0]; }), ys = X.map(function (v) { return X[0].length > 1 ? v[1] : 0; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var svg = svgEl("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Первые две координаты загруженных эмбеддингов" });
    svg.appendChild(svgEl("rect", { x: pad, y: 6, width: W - pad - 8, height: H - pad - 6, fill: "none", stroke: css("--rule") }));
    X.forEach(function (v, i) {
      var cx = pad + (x1 === x0 ? 0.5 : (xs[i] - x0) / (x1 - x0)) * (W - pad - 14), cy = 12 + (1 - (y1 === y0 ? 0.5 : (ys[i] - y0) / (y1 - y0))) * (H - pad - 18);
      var c = svgEl("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: 3.4, fill: css("--accent"), "fill-opacity": 0.75 });
      c.appendChild(svgEl("title", {}, HOSTS[i])); svg.appendChild(c);
    });
    svg.appendChild(svgEl("text", { x: W / 2, y: H - 8, "text-anchor": "middle", "class": "mut" }, "emb_0"));
    host.appendChild(svg);
  }
  function setStatus(msg, err) { statusEl.textContent = msg; statusEl.className = err ? "err" : ""; }

  function run() {
    var f = document.getElementById("file").files[0];
    resultsEl.hidden = true;
    if (!f) { setStatus("Сначала выберите embeddings.csv.", true); return; }
    if (f.size > MAX_BYTES) { setStatus("Файл больше 8 МБ.", true); return; }
    setStatus("Оценка в браузере…");
    f.text().then(function (text) {
      var v = validate(parseCsv(text));
      /* yield so the status text paints before the O(N^2) computation */
      return new Promise(function (res) { setTimeout(res, 20); }).then(function () {
        var items = [{ name: "Загруженный CSV", m: metrics(v.X) }, { name: "Эталонный максимум (утечка)", m: metrics(REF) }, { name: "Случайные векторы", m: metrics(randomEmbedding(v.dim)) }];
        resultsEl.hidden = false;
        renderTable(items.slice()); renderBars(items); renderScatter(v.X);
        setStatus("Файл проверен: " + N + " хостов × " + v.dim + " измерений.");
      });
    }).catch(function (e) { resultsEl.hidden = true; setStatus("Ошибка: " + e.message, true); });
  }
  function sample() {
    var r = rng(123), d = 16, lines = ["host_id," + Array.from({ length: d }, function (_, j) { return "emb_" + j; }).join(",")];
    for (var i = 0; i < N; i++) {
      var role = i % 5, row = [];
      for (var j = 0; j < d; j++) row.push(((j < 5 && j === role ? 2.2 : 0) + 0.35 * normal(r)).toFixed(6));
      lines.push("h" + i + "," + row.join(","));
    }
    var a = document.createElement("a");
    a.href = URL.createObjectURL(new Blob([lines.join("\n")], { type: "text/csv" }));
    a.download = "embeddings.csv"; document.body.appendChild(a); a.click(); a.remove(); URL.revokeObjectURL(a.href);
  }
  window.__pg = { parseCsv: parseCsv, validate: validate, metrics: metrics, REF: REF };
  document.addEventListener("DOMContentLoaded", function () {
    statusEl = document.getElementById("status"); resultsEl = document.getElementById("results");
    document.getElementById("run").addEventListener("click", run);
    document.getElementById("sample").addEventListener("click", sample);
  });
})();
