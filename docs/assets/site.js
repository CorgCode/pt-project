/* Графики сайта без зависимостей. Все числа читаются из data/results.json (собирает docs/build_data.py);
   здесь ничего не считается, кроме форматирования. Графики — inline SVG. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  /* Названия методов и метрик не переводим: так они называются в коде и статьях. */
  var TASKS = ["Поведение", "Будущие связи", "Общие собеседники", "Узнаваемость"];   // ключи групп в results.json
  var TASK_EN = { "Поведение": "Behavior", "Будущие связи": "Future links", "Общие собеседники": "Shared partners", "Узнаваемость": "Identity" };
  var SHORT = { "Поведение": "Beh.", "Будущие связи": "Links", "Общие собеседники": "Pairs", "Узнаваемость": "Ident." };
  var METRIC_EN = { "consistency@10_mean": "consistency@10", "link_learned_roc_auc": "link ROC-AUC",
    "partner_new_mrr": "new-partner MRR", "pair_roc_auc": "pair ROC-AUC", "identity_top1": "identity top-1" };
  var METRICS = {
    neighbors: {
      series: [["consistency@5_mean", "k = 5 neighbors"], ["consistency@10_mean", "k = 10"], ["consistency@20_mean", "k = 20"]],
      title: "Neighborhood consistency@k",
      what: "Из k ближайших соседей хоста — доля тех же, что у эталона поведения в test.",
      caveat: "behavioral и эталон строятся из похожих признаков; выше ≈ 0,31 не прыгнуть."
    },
    links: {
      series: [["link_learned_roc_auc", "link ROC-AUC"], ["link_new_roc_auc", "new links ROC-AUC"], ["partner_new_mrr", "new-partner MRR"]],
      fixedMax: 1, ref: 0.5, refLabel: "ROC-AUC 0,5",
      title: "Future link prediction",
      what: "Предсказываются ли связи test-окна по эмбеддингам train.",
      caveat: "88% связей уже были в train («уже общались» — 0,939), поэтому важнее new links и new partners."
    },
    pairs: {
      series: [["pair_roc_auc", "pair ROC-AUC"]], bar: true, fixedMax: 1, ref: 0.5, refLabel: "0,5",
      title: "Pair ROC-AUC",
      what: "Отличает ли расстояние пары с общими собеседниками от пар без них.",
      caveat: "Спорные пары исключены — задача лёгкая."
    },
    identity: {
      series: [["identity_top1", "identity top-1"]], bar: true, chanceRef: true, refLabel: "random",
      title: "Identity top-1",
      what: "Ближайший вектор в test-окне — тот же хост?",
      caveat: "Нужен test.csv; окна по 30 минут короткие."
    }
  };
  var state = { data: null, metric: "neighbors", weights: {} };

  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs || {}) if (attrs[k] != null) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function html(tag, attrs, text) {
    var n = document.createElement(tag);
    for (var k in attrs || {}) n.setAttribute(k, attrs[k]);
    if (text != null) n.textContent = text;
    return n;
  }
  function css(name) { return getComputedStyle(document.documentElement).getPropertyValue(name).trim(); }
  function comma(s) { return String(s).replace(".", ","); }
  function fmt(v) { return v == null ? "—" : comma(v.toFixed(3)); }
  function name(m) { return m; }
  function kind(m) {
    if (m === "random" || m === "reference_ceiling") return "baseline";
    if (m.indexOf("hybrid") === 0) return "hybrid";
    return m.indexOf("behavioral") === 0 ? "behavior" : "graph";
  }
  function rows() { return state.data.team.rows.filter(function (r) { return r.method !== "reference_ceiling"; }); }
  function w(t) { return state.weights[t] == null ? 1 : state.weights[t]; }
  function equalWeights() { return TASKS.every(function (t) { return w(t) === w(TASKS[0]); }); }
  /* Взвешенное среднее геометрическое по задачам (как в evaluation/scoreboard.py, пол 1). При равных весах — score из файла. */
  function score(r) {
    if (equalWeights()) return r.score;
    var s = 0, n = 0;
    TASKS.forEach(function (t) { if (r.tasks[t] != null && w(t) > 0) { s += w(t) * Math.log(Math.max(r.tasks[t], 1)); n += w(t); } });
    return n ? Math.exp(s / n) : r.score;
  }
  function ranked() {
    var base = {}, i = 0;
    rows().forEach(function (r) { if (!r.service) base[r.method] = ++i; });
    var list = rows().slice().sort(function (a, b) { if (a.service !== b.service) return a.service - b.service;
      var d = score(b) - score(a);   // разница меньше округления — порядок как при равных весах
      return Math.abs(d) < 0.05 ? (base[a.method] || 99) - (base[b.method] || 99) : d; });
    return { list: list, base: base };
  }

  /* Ползунки весов задач: проверить, меняется ли порядок. Официальный score — равные веса. */
  function weightsUI(host) {
    host.textContent = "";
    host.appendChild(html("span", { "class": "wl" }, "Task weights"));
    TASKS.forEach(function (t) {
      var lab = html("label"), inp = html("input", { type: "range", min: "0", max: "3", step: "1", value: String(w(t)), "aria-label": TASK_EN[t] + " weight" });
      var out = html("output", null, "×" + w(t));
      inp.addEventListener("input", function () { state.weights[t] = Number(inp.value); out.textContent = "×" + inp.value; reset.hidden = equalWeights(); leaderboard(document.getElementById("leaderboard")); });
      lab.appendChild(document.createTextNode(TASK_EN[t])); lab.appendChild(inp); lab.appendChild(out); host.appendChild(lab);
    });
    var reset = html("button", { type: "button", "class": "reset" }, "Equal weights");
    reset.hidden = equalWeights();
    reset.addEventListener("click", function () { state.weights = {}; weightsUI(host); leaderboard(document.getElementById("leaderboard")); });
    host.appendChild(reset);
  }
  function real() { return state.data.team.rows.filter(function (r) { return !r.service; }); }

  /* Главная таблица: общий балл полоской и балл каждой задачи тепловой ячейкой. */
  function leaderboard(host) {
    host.textContent = "";
    var tb = html("table", { "class": "board" }), head = html("tr");
    head.appendChild(html("th", null, "#"));
    head.appendChild(html("th", null, "Method"));
    head.appendChild(html("th", null, "Score"));
    TASKS.forEach(function (t) {
      var th = html("th", { "class": "heat", title: TASK_EN[t] });
      th.appendChild(html("span", { "class": "full" }, TASK_EN[t])); th.appendChild(html("span", { "class": "short" }, SHORT[t]));
      head.appendChild(th);
    });
    tb.appendChild(head);
    var place = 0, rk = ranked();
    rk.list.forEach(function (r) {
      var tr = html("tr", r.service ? { "class": "service" } : null);
      if (!r.service && place === 0) tr.className = "lead";
      var pl = html("td", { "class": "place" }, r.service ? "" : String(++place));
      var moved = r.service ? 0 : rk.base[r.method] - place;
      if (moved) pl.appendChild(html("span", { "class": moved > 0 ? "up" : "down", title: "was #" + rk.base[r.method] }, (moved > 0 ? "▲" : "▼") + Math.abs(moved)));
      tr.appendChild(pl);
      var nm = html("td", { "class": "name" }, name(r.method));
      nm.appendChild(html("small", null, kind(r.method) + (r.dim ? " · dim " + r.dim : "")));
      tr.appendChild(nm);
      var sc = html("td"), bar = html("div", { "class": "scorebar" }), fill = html("i");
      var v0 = score(r);
      fill.style.width = Math.max(2, v0) * 2 + "px";
      bar.appendChild(html("b", null, comma(v0.toFixed(1)))); bar.appendChild(fill); sc.appendChild(bar);
      tr.appendChild(sc);
      TASKS.forEach(function (t) {
        var v = r.tasks[t], td = html("td", { "class": "heat" }, v == null ? "—" : String(Math.round(v)));
        if (v != null) {
          var p = Math.round(Math.min(100, v) * 0.85);
          td.style.background = "color-mix(in srgb, var(--heat) " + p + "%, transparent)";
          if (p > 48) td.style.color = "var(--on-accent)";
          td.title = TASK_EN[t] + ": " + v.toFixed(1) + " / 100";
        }
        tr.appendChild(td);
      });
      tb.appendChild(tr);
    });
    var wrap = html("div", { "class": "tablewrap" }); wrap.appendChild(tb); host.appendChild(wrap);
    host.appendChild(rawTable());
  }

  /* Исходные метрики — под спойлером. */
  function rawTable() {
    var t = state.data.team, keys = Object.keys(t.columns);
    var det = html("details"); det.appendChild(html("summary", null, "Raw metrics"));
    var tb = html("table"), head = html("tr");
    head.appendChild(html("th", null, "Method"));
    keys.forEach(function (k) { head.appendChild(html("th", { "class": "num" }, METRIC_EN[k] || k)); });
    tb.appendChild(head);
    rows().forEach(function (r) {
      var tr = html("tr"); tr.appendChild(html("td", null, name(r.method)));
      keys.forEach(function (k) { tr.appendChild(html("td", { "class": "num" }, fmt(r.values[k]))); });
      tb.appendChild(tr);
    });
    var tr = html("tr"); tr.appendChild(html("td", null, "random level"));
    keys.forEach(function (k) { var c = t.chance[k]; tr.appendChild(html("td", { "class": "num" }, !c ? "—" : (c[1] - c[0] < 1e-9 ? fmt(c[0]) : fmt(c[0]) + "–" + fmt(c[1])))); });
    tr.style.color = "var(--muted)"; tb.appendChild(tr);
    var wrap = html("div", { "class": "tablewrap" }); wrap.appendChild(tb); det.appendChild(wrap);
    return det;
  }

  /* Точечный или столбчатый график: строка на метод, общая ось. */
  function draw(host, cfg) {
    host.textContent = "";
    var data = rows().map(function (r) { return { key: r.method, values: cfg.series.map(function (s) { return r.detail[s[0]]; }) }; })
      .filter(function (r) { return r.values.some(function (v) { return v != null; }); });
    var all = [].concat.apply([], data.map(function (r) { return r.values; })).filter(function (v) { return v != null; });
    var hi = cfg.fixedMax || Math.min(1, Math.ceil(Math.max.apply(null, all) * 1.15 * 10) / 10);
    var step = hi <= 0.5 ? 0.1 : 0.25, ticks = [];
    for (var t = 0; t <= hi + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100);
    var ref = cfg.ref, chance = state.data.team.chance.identity_top1;
    if (cfg.chanceRef && chance) ref = chance[0];

    var W = Math.max(280, Math.floor(host.clientWidth || 700)), narrow = W < 560;
    var labelW = narrow ? 0 : Math.min(250, Math.round(W * 0.33));
    var rowH = narrow ? 44 : 28, top = 24, right = 44;
    var plotX = labelW + (narrow ? 4 : 12), plotW = W - plotX - right;
    var H = top + data.length * rowH + 24;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": cfg.title });
    function X(v) { return plotX + v / hi * plotW; }
    var rule = css("--rule"), muted = css("--muted");
    var colors = [css("--s1"), css("--s2"), css("--s3")];
    ticks.forEach(function (tk) {
      svg.appendChild(el("line", { x1: X(tk), x2: X(tk), y1: top - 8, y2: H - 22, stroke: rule }));
      svg.appendChild(el("text", { x: X(tk), y: H - 6, "text-anchor": "middle", "class": "mut" }, comma(tk)));
    });
    if (ref != null) {
      svg.appendChild(el("line", { x1: X(ref), x2: X(ref), y1: top - 8, y2: H - 22, stroke: muted, "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
      svg.appendChild(el("text", { x: X(ref) + 4, y: top - 11, "class": "mut" }, cfg.refLabel));
    }
    data.forEach(function (r, i) {
      var y0 = top + i * rowH, cy = y0 + (narrow ? 30 : rowH / 2);
      var g = el("g", { "class": "hit" });
      g.appendChild(el("title", null, name(r.key) + "\n" + cfg.series.map(function (s, j) { return s[1] + ": " + fmt(r.values[j]); }).join("\n")));
      g.appendChild(el("rect", { x: 0, y: y0, width: W, height: rowH, fill: "transparent" }));
      g.appendChild(el("text", { x: 0, y: narrow ? y0 + 12 : cy + 4, "class": r.key === "random" ? "mut" : null }, name(r.key)));
      if (cfg.bar) {
        var v = r.values[0];
        g.appendChild(el("rect", { "class": "mark", x: plotX, y: cy - 7, width: Math.max(2, X(v) - plotX), height: 14, rx: 4, fill: colors[0] }));
        g.appendChild(el("text", { x: X(v) + 6, y: cy + 4 }, fmt(v)));
      } else {
        var xs = r.values.filter(function (v) { return v != null; }).map(X);
        if (xs.length > 1) g.appendChild(el("line", { x1: Math.min.apply(null, xs), x2: Math.max.apply(null, xs), y1: cy, y2: cy, stroke: rule, "stroke-width": 2 }));
        r.values.forEach(function (v, s) {
          if (v == null) return;
          var x = X(v), c = colors[s];
          if (s === 0) g.appendChild(el("circle", { "class": "mark", cx: x, cy: cy, r: 5.5, fill: c, stroke: css("--paper"), "stroke-width": 2 }));
          else if (s === 1) g.appendChild(el("rect", { "class": "mark", x: x - 5, y: cy - 5, width: 10, height: 10, fill: c, stroke: css("--paper"), "stroke-width": 2 }));
          else g.appendChild(el("path", { "class": "mark", d: "M" + x + " " + (cy - 6.5) + "L" + (x + 6.5) + " " + cy + "L" + x + " " + (cy + 6.5) + "L" + (x - 6.5) + " " + cy + "Z", fill: c, stroke: css("--paper"), "stroke-width": 2 }));
        });
      }
      svg.appendChild(g);
    });
    host.appendChild(svg);
    if (cfg.series.length > 1) {
      var lg = html("div", { "class": "legend" });
      cfg.series.forEach(function (s, i) {
        var item = html("span"), sw = html("span", { "class": "sw" });
        sw.style.background = colors[i];
        sw.style.borderRadius = i === 0 ? "50%" : "0";
        if (i === 2) sw.style.transform = "rotate(45deg) scale(.8)";
        item.appendChild(sw); item.appendChild(document.createTextNode(s[1])); lg.appendChild(item);
      });
      host.appendChild(lg);
    }
  }

  /* Карта хостов: PCA эмбеддинга behavioral, цвет — доля соседей, сохранившихся в test. */
  function hostMap(host) {
    host.textContent = "";
    var pr = state.data.projection, pts = pr.points.slice().sort(function (a, b) { return a.agreement - b.agreement; });
    var W = Math.max(280, Math.floor(host.clientWidth || 700)), H = Math.round(Math.min(500, Math.max(300, W * 0.55))), pad = 14;
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var k = Math.min((W - 2 * pad) / (x1 - x0), (H - 2 * pad) / (y1 - y0));
    var ox = (W - k * (x1 - x0)) / 2, oy = (H - k * (y1 - y0)) / 2;
    function X(v) { return ox + (v - x0) * k; } function Y(v) { return H - oy - (v - y0) * k; }
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img",
      "aria-label": pts.length + " hosts; mean share of neighbors kept " + comma(pr.per_host_mean_agreement.toFixed(2)) });
    svg.appendChild(el("rect", { x: 0.5, y: 0.5, width: W - 1, height: H - 1, rx: 8, fill: "none", stroke: css("--rule") }));
    var lo = css("--heat-lo"), hi = css("--heat"), r = W < 560 ? 2.2 : 2.8;
    pts.forEach(function (p) {
      svg.appendChild(el("circle", { cx: X(p.x).toFixed(1), cy: Y(p.y).toFixed(1), r: r,
        fill: "color-mix(in srgb, " + hi + " " + Math.round(15 + p.agreement * 85) + "%, " + lo + ")" }));
    });
    var named = pts.filter(function (p) { return !/^Comp/.test(p.id); });
    named.forEach(function (p) {
      var cx = X(p.x), cy = Y(p.y), right = cx > W * 0.6;
      svg.appendChild(el("circle", { cx: cx, cy: cy, r: r + 3, fill: "none", stroke: css("--ink"), "stroke-width": 1.2 }));
      svg.appendChild(el("text", { x: right ? cx - 8 : cx + 8, y: cy + 4, "text-anchor": right ? "end" : "start", "class": "lbl" }, p.id));
    });
    var ring = el("circle", { r: r + 4, fill: "none", stroke: css("--ink"), "stroke-width": 2, visibility: "hidden" });
    svg.appendChild(ring);
    var tip = html("div", { "class": "tip", hidden: "" });
    svg.addEventListener("pointermove", function (e) {
      var b = svg.getBoundingClientRect(), mx = (e.clientX - b.left) * W / b.width, my = (e.clientY - b.top) * H / b.height, best = null, bd = 400;
      pts.forEach(function (p) { var d = Math.pow(X(p.x) - mx, 2) + Math.pow(Y(p.y) - my, 2); if (d < bd) { bd = d; best = p; } });
      if (!best) { ring.setAttribute("visibility", "hidden"); tip.hidden = true; return; }
      ring.setAttribute("cx", X(best.x)); ring.setAttribute("cy", Y(best.y)); ring.setAttribute("visibility", "visible");
      tip.textContent = ""; tip.appendChild(html("b", null, best.id));
      tip.appendChild(html("span", null, Math.round(best.agreement * 10) + " / 10 neighbors kept"));
      tip.hidden = false;
      var f = host.parentElement.getBoundingClientRect();   // подсказка позиционируется внутри .fig
      var tx = b.left - f.left + (X(best.x) / W) * b.width + 12, ty = b.top - f.top + (Y(best.y) / H) * b.height + 12;
      if (tx > f.width - 190) tx -= 200;
      tip.style.left = tx + "px"; tip.style.top = ty + "px";
    });
    svg.addEventListener("pointerleave", function () { ring.setAttribute("visibility", "hidden"); tip.hidden = true; });
    host.appendChild(svg); host.appendChild(tip);
    var lg = html("div", { "class": "legend ramp" });
    lg.appendChild(html("span", null, "0 / 10"));
    var bar = html("i"); bar.style.background = "linear-gradient(90deg, color-mix(in srgb, " + hi + " 15%, " + lo + "), " + hi + ")";
    lg.appendChild(bar); lg.appendChild(html("span", null, "10 / 10 neighbors kept"));
    lg.appendChild(html("span", { "class": "mean" }, "mean " + comma((pr.per_host_mean_agreement * 10).toFixed(1)) + " / 10"));
    host.appendChild(lg);
  }

  function renderMetric() {
    var cfg = METRICS[state.metric], host = document.getElementById("chartMetric");
    if (host) draw(host, cfg);
    document.querySelector("[data-explain=what]").textContent = cfg.what;
    document.querySelector("[data-explain=caveat]").textContent = cfg.caveat;
    document.querySelectorAll(".tabs [data-metric]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.metric === state.metric)); });
  }

  function fill() {
    var t = state.data.team, ds = state.data.dataset, r = real();
    function set(sel, txt) { document.querySelectorAll(sel).forEach(function (n) { n.textContent = txt; }); }
    function num(v) { return v == null ? "—" : Number(v).toLocaleString("ru-RU"); }
    function bestOf(k) { return Math.max.apply(null, r.map(function (x) { return x.values[k] == null ? -1 : x.values[k]; })); }
    var pct = function (v) { return Math.round(v * 100) + "%"; };
    set("[data-kpi=best-score]", comma(r[0].score.toFixed(1)));
    set("[data-kpi=best-method]", name(r[0].method));
    set("[data-kpi=link]", fmt(bestOf("link_learned_roc_auc")));
    set("[data-kpi=persistence]", fmt(t.persistence_auc));
    set("[data-kpi=partners]", "×" + comma((bestOf("partner_new_mrr") / t.chance.partner_new_mrr[0]).toFixed(1)));
    set("[data-kpi=identity]", pct(bestOf("identity_top1")));
    set("[data-kpi=identity-chance]", comma((t.chance.identity_top1[0] * 100).toFixed(1)) + "%");
    set("[data-ds=rows]", num(ds.valid_rows)); set("[data-ds=train-min]", ds.windows.train.minutes);
    set("[data-ds=test-min]", ds.windows.test.minutes); set("[data-ds=evaluated]", num(ds.evaluated));
    set("[data-meta=commit]", state.data.meta.source_commit.slice(0, 8));
  }

  function renderAll() {
    if (!state.data) return;
    weightsUI(document.getElementById("weights"));
    leaderboard(document.getElementById("leaderboard"));
    renderMetric();
    hostMap(document.getElementById("hostMap"));
  }

  function init() {
    document.querySelectorAll(".tabs [data-metric]").forEach(function (b) {
      b.addEventListener("click", function () { state.metric = b.dataset.metric; renderMetric(); });
    });
    var timer;
    window.addEventListener("resize", function () { clearTimeout(timer); timer = setTimeout(function () { renderMetric(); if (state.data) hostMap(document.getElementById("hostMap")); }, 120); });
    var links = [].slice.call(document.querySelectorAll("nav.main a[href^='#']"));
    if ("IntersectionObserver" in window && links.length) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) links.forEach(function (a) { if (a.getAttribute("href") === "#" + e.target.id) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); }); });
      }, { rootMargin: "-20% 0px -70% 0px" });
      links.forEach(function (a) { var t = document.querySelector(a.getAttribute("href")); if (t) io.observe(t); });
    }
    fetch("./data/results.json", { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(r.status); return r.json();
    }).then(function (d) {
      state.data = d; fill(); renderAll();
      document.documentElement.setAttribute("data-charts", "ready");
    }).catch(function (e) {
      document.querySelectorAll(".fig").forEach(function (n) { n.textContent = "Не удалось загрузить data/results.json (" + e.message + ")."; });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
