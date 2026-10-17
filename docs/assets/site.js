/* Графики сайта без зависимостей: данные из data/results.json (собирает docs/build_data.py),
   charts are inline SVG. Every number shown is read from that file; nothing is computed or invented here
   except the arithmetic mean already verified by docs/check_pages.py. */
(function () {
  "use strict";
  var NS = "http://www.w3.org/2000/svg";
  var LABEL = {
    behavioral: "Поведенческий профиль",
    behavioral_pca: "Поведенческий + PCA",
    behavioral_svd: "Поведенческий + SVD",
    graph_spectral: "Спектральный граф",
    random: "Случайный (нижняя граница)",
    reference_ceiling: "Потолок эталона (утечка)",
    deepwalk: "DeepWalk", node2vec: "node2vec", graphsage: "GraphSAGE",
    struc2vec: "struc2vec", rolx: "RolX"
  };
  var PENDING_TEXT = "Ожидаются результаты эксперимента";
  var state = { data: null, split: "test" };

  function el(tag, attrs, text) {
    var n = document.createElementNS(NS, tag);
    for (var k in attrs || {}) n.setAttribute(k, attrs[k]);
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
  function fmt(v) { return v == null ? "—" : v.toFixed(3).replace(".", ","); }
  function name(m) { return LABEL[m] || m; }

  /* rows: [{key, series:[value|null,...]}]; pending rows have all-null values */
  function rowsFor(getter, nSeries) {
    var d = state.data, out = [];
    d.meta.missing.slice().sort().forEach(function (m) {
      out.push({ key: m, pending: true, values: new Array(nSeries).fill(null) });
    });
    d.meta.methods.forEach(function (m) { out.push({ key: m, pending: false, values: getter(m) }); });
    return out;
  }

  function draw(host, rows, opts) {
    host.textContent = "";
    var W = Math.max(280, Math.floor(host.clientWidth || 700));
    var narrow = W < 540;
    var labelW = narrow ? 0 : Math.min(230, Math.round(W * 0.32));
    var rowH = narrow ? 46 : 32, top = 22, right = 14;
    var plotX = labelW + (narrow ? 16 : 14), plotW = W - plotX - right - (narrow ? 6 : 0);
    var H = top + rows.length * rowH + 26;
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": opts.title });
    var lo = opts.min, hi = opts.max;
    function X(v) { return plotX + (v - lo) / (hi - lo) * plotW; }
    var ink = css("--ink"), rule = css("--rule"), accent = css("--accent"), accent2 = css("--accent-2"), muted = css("--muted"), warn = css("--warn");
    var ticks = opts.ticks;
    ticks.forEach(function (t) {
      svg.appendChild(el("line", { x1: X(t), x2: X(t), y1: top - 6, y2: H - 22, stroke: rule, "stroke-width": 1 }));
      svg.appendChild(el("text", { x: X(t), y: H - 6, "text-anchor": "middle", "class": "mut" }, String(t).replace(".", ",")));
    });
    if (opts.ref != null) {
      svg.appendChild(el("line", { x1: X(opts.ref), x2: X(opts.ref), y1: top - 6, y2: H - 22, stroke: muted, "stroke-width": 1.2, "stroke-dasharray": "4 3" }));
      svg.appendChild(el("text", { x: X(opts.ref) + 4, y: top - 9, "class": "mut" }, opts.refLabel));
    }
    var colors = [accent, accent2, warn];
    rows.forEach(function (r, i) {
      var y0 = top + i * rowH, cy = y0 + (narrow ? 30 : rowH / 2);
      var t = el("text", { x: narrow ? 4 : 0, y: narrow ? y0 + 12 : cy + 4 }, name(r.key));
      if (r.pending) t.setAttribute("class", "mut");
      svg.appendChild(t);
      if (r.pending) {
        svg.appendChild(el("rect", { x: plotX, y: cy - 9, width: plotW, height: 18, fill: "none", stroke: rule, "stroke-dasharray": "3 3" }));
        svg.appendChild(el("text", { x: plotX + 8, y: cy + 4, "class": "mut" }, PENDING_TEXT));
        return;
      }
      if (opts.kind === "bar") {
        var v = r.values[0];
        var leak = r.key === "reference_ceiling";
        svg.appendChild(el("rect", { x: plotX, y: cy - 8, width: Math.max(1, X(v) - plotX), height: 16, fill: leak ? "none" : accent, stroke: leak ? warn : "none", "stroke-dasharray": leak ? "4 2" : "none" }));
        svg.appendChild(el("text", { x: Math.min(X(v) + 6, W - 40), y: cy + 4 }, fmt(v)));
      } else {
        var xs = [];
        r.values.forEach(function (v, s) {
          if (v == null) return;
          var x = X(v), c = colors[s % 3];
          if (s === 0) svg.appendChild(el("circle", { cx: x, cy: cy, r: 5.5, fill: c }));
          else if (s === 1) svg.appendChild(el("rect", { x: x - 5, y: cy - 5, width: 10, height: 10, fill: c }));
          else svg.appendChild(el("path", { d: "M" + x + " " + (cy - 6.5) + "L" + (x + 6.5) + " " + cy + "L" + x + " " + (cy + 6.5) + "L" + (x - 6.5) + " " + cy + "Z", fill: c }));
          xs.push(x);
        });
        if (xs.length > 1) svg.insertBefore(el("line", { x1: Math.min.apply(null, xs), x2: Math.max.apply(null, xs), y1: cy, y2: cy, stroke: rule, "stroke-width": 2 }), svg.firstChild);
      }
    });
    host.appendChild(svg);
    if (opts.legend) {
      var lg = html("div", { "class": "legend" });
      opts.legend.forEach(function (t, s) {
        var item = html("span");
        var sw = html("span", { "class": "sw" });
        sw.style.background = colors[s]; if (s === 1) sw.style.borderRadius = "0"; if (s === 0) sw.style.borderRadius = "50%"; if (s === 2) sw.style.transform = "rotate(45deg) scale(.8)";
        item.appendChild(sw); item.appendChild(document.createTextNode(t)); lg.appendChild(item);
      });
      host.appendChild(lg);
    }
    /* accessible table twin */
    var det = html("details"); det.appendChild(html("summary", null, "Таблица значений"));
    var wrap = html("div", { "class": "tablewrap" }), tb = html("table");
    var head = html("tr"); head.appendChild(html("th", null, "Метод"));
    (opts.legend || [opts.valueLabel]).forEach(function (t) { head.appendChild(html("th", { "class": "num" }, t)); });
    tb.appendChild(head);
    rows.forEach(function (r) {
      var tr = html("tr"); tr.appendChild(html("td", null, name(r.key)));
      r.values.forEach(function (v) { tr.appendChild(html("td", { "class": "num" }, r.pending ? "ожидается" : fmt(v))); });
      tb.appendChild(tr);
    });
    wrap.appendChild(tb); det.appendChild(wrap); host.appendChild(det);
  }

  function scatter(host, pts) {
    host.textContent = "";
    var W = Math.max(280, Math.floor(host.clientWidth || 700)), H = Math.round(Math.min(440, Math.max(260, W * 0.6)));
    var pad = 30;
    var xs = pts.map(function (p) { return p.x; }), ys = pts.map(function (p) { return p.y; });
    var x0 = Math.min.apply(null, xs), x1 = Math.max.apply(null, xs), y0 = Math.min.apply(null, ys), y1 = Math.max.apply(null, ys);
    var svg = el("svg", { viewBox: "0 0 " + W + " " + H, role: "img", "aria-label": "Двумерная проекция PCA хостов; цвет показывает долю совпавших ближайших соседей" });
    svg.appendChild(el("rect", { x: pad, y: 6, width: W - pad - 8, height: H - pad - 6, fill: "none", stroke: css("--rule") }));
    var lowC = css("--rule"), hiC = css("--accent");
    function mix(a, b, t) {
      var pa = parseInt(a.slice(1), 16), pb = parseInt(b.slice(1), 16), o = "#";
      [16, 8, 0].forEach(function (s) { var ca = (pa >> s) & 255, cb = (pb >> s) & 255; o += ("0" + Math.round(ca + (cb - ca) * t).toString(16)).slice(-2); });
      return o;
    }
    var sorted = pts.slice().sort(function (a, b) { return a.agreement - b.agreement; });
    sorted.forEach(function (p) {
      var cx = pad + (p.x - x0) / (x1 - x0) * (W - pad - 14), cy = 12 + (1 - (p.y - y0) / (y1 - y0)) * (H - pad - 18);
      var c = el("circle", { cx: cx.toFixed(1), cy: cy.toFixed(1), r: 3.4, fill: mix(lowC, hiC, p.agreement), "fill-opacity": 0.9 });
      c.appendChild(el("title", {}, p.id + ": доля совпавших соседей " + fmt(p.agreement)));
      svg.appendChild(c);
    });
    svg.appendChild(el("text", { x: W / 2, y: H - 8, "text-anchor": "middle", "class": "mut" }, "Главная компонента 1"));
    var yl = el("text", { x: 14, y: H / 2, "text-anchor": "middle", "class": "mut", transform: "rotate(-90 14 " + H / 2 + ")" }, "Главная компонента 2");
    svg.appendChild(yl);
    host.appendChild(svg);
    var lg = html("div", { "class": "legend" });
    [["Доля совпавших соседей: 0", lowC], ["1", hiC]].forEach(function (a) {
      var s = html("span"), sw = html("span", { "class": "sw" }); sw.style.background = a[1]; sw.style.borderRadius = "50%";
      s.appendChild(sw); s.appendChild(document.createTextNode(a[0])); lg.appendChild(s);
    });
    host.appendChild(lg);
  }

  function leaderboard(host) {
    host.textContent = "";
    var d = state.data, res = d.results[state.split];
    var cols = [["consistency@10_mean", "Согласие соседей @10"], ["link_roc_auc", "ROC-AUC будущих связей"], ["pair_roc_auc", "ROC-AUC пар"], ["triplet_accuracy", "Точность триплетов"]];
    var tb = html("table"), head = html("tr");
    head.appendChild(html("th", null, "Метод"));
    cols.forEach(function (c) { head.appendChild(html("th", { "class": "num" }, c[1])); });
    head.appendChild(html("th", null, "Статус"));
    tb.appendChild(head);
    var ms = d.meta.methods.slice().sort(function (a, b) { return res[b]["consistency@10_mean"] - res[a]["consistency@10_mean"]; });
    ms.forEach(function (m) {
      var tr = html("tr"); tr.appendChild(html("td", null, name(m)));
      cols.forEach(function (c) { tr.appendChild(html("td", { "class": "num" }, fmt(res[m][c[0]]))); });
      var st = m === "reference_ceiling" ? "утечка: не метод" : m === "random" ? "нижняя граница" : d.meta.label;
      var td = html("td"); td.appendChild(html("span", { "class": "tag syn" }, st)); tr.appendChild(td); tb.appendChild(tr);
    });
    d.meta.missing.slice().sort().forEach(function (m) {
      var tr = html("tr"); tr.appendChild(html("td", { "class": "mut" }, name(m)));
      var td = html("td", { colspan: String(cols.length) }, PENDING_TEXT); td.style.color = css("--muted"); tr.appendChild(td);
      var st = html("td"); st.appendChild(html("span", { "class": "tag pend" }, "не запускался")); tr.appendChild(st); tb.appendChild(tr);
    });
    var wrap = html("div", { "class": "tablewrap" }); wrap.appendChild(tb); host.appendChild(wrap);
  }

  function renderAll() {
    var d = state.data; if (!d) return;
    var res = d.results[state.split];
    var kEl = document.getElementById("chartConsistency");
    if (kEl) draw(kEl, rowsFor(function (m) { return [res[m]["consistency@5_mean"], res[m]["consistency@10_mean"], res[m]["consistency@20_mean"]]; }, 3),
      { kind: "dot", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], title: "Согласие ближайших соседей по методам", legend: ["k = 5", "k = 10", "k = 20"] });
    var lEl = document.getElementById("chartLinks");
    if (lEl) draw(lEl, rowsFor(function (m) { return [res[m].link_roc_auc, res[m].link_learned_roc_auc, res[m].link_new_roc_auc]; }, 3),
      { kind: "dot", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], ref: 0.5, refLabel: "случайный уровень 0,5", title: "ROC-AUC прогноза будущих связей",
        legend: ["косинус", "обученный классификатор", "только новые связи"] });
    var pEl = document.getElementById("chartPairs");
    if (pEl) draw(pEl, rowsFor(function (m) { return [res[m].pair_roc_auc]; }, 1),
      { kind: "bar", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], ref: 0.5, refLabel: "случайный уровень 0,5", title: "ROC-AUC различения похожих и непохожих пар", valueLabel: "ROC-AUC пар" });
    var iEl = document.getElementById("chartIdentity");
    if (iEl) {
      var ids = d.identity;
      draw(iEl, rowsFor(function (m) { var r = ids[m]; return r ? [r.raw_top1, r.aligned_top1] : [null, null]; }, 2).filter(function (r) { return r.pending || ids[r.key]; }),
        { kind: "dot", min: 0, max: 1, ticks: [0, 0.25, 0.5, 0.75, 1], ref: ids.behavioral.chance_top1, refLabel: "случайное угадывание", title: "Точность top-1 при поиске того же хоста в следующем окне", legend: ["без выравнивания", "с выравниванием (Прокруст)"] });
    }
    var sEl = document.getElementById("chartHosts"); if (sEl) scatter(sEl, d.projection.points);
    var bEl = document.getElementById("leaderboard"); if (bEl) leaderboard(bEl);
    document.querySelectorAll("[data-split-label]").forEach(function (n) { n.textContent = state.split === "test" ? "тестовое окно" : "валидационное окно"; });
    document.querySelectorAll(".tabs [data-split]").forEach(function (b) { b.setAttribute("aria-pressed", String(b.dataset.split === state.split)); });
  }

  function fillMeta() {
    var m = state.data.meta;
    document.querySelectorAll("[data-meta=hosts]").forEach(function (n) { n.textContent = m.hosts; });
    document.querySelectorAll("[data-meta=commit]").forEach(function (n) { n.textContent = m.source_commit.slice(0, 8); });
    document.querySelectorAll("[data-meta=pending]").forEach(function (n) { n.textContent = m.missing.map(name).join(", "); });
    document.querySelectorAll("[data-meta=meanagree]").forEach(function (n) { n.textContent = fmt(state.data.projection.per_host_mean_agreement); });
  }

  function init() {
    document.querySelectorAll(".tabs [data-split]").forEach(function (b) {
      b.addEventListener("click", function () { state.split = b.dataset.split; renderAll(); });
    });
    var timer;
    window.addEventListener("resize", function () { clearTimeout(timer); timer = setTimeout(renderAll, 120); });
    var ids = [].slice.call(document.querySelectorAll("nav.main a[href^='#']"));
    if ("IntersectionObserver" in window && ids.length) {
      var io = new IntersectionObserver(function (es) {
        es.forEach(function (e) { if (e.isIntersecting) ids.forEach(function (a) { if (a.getAttribute("href") === "#" + e.target.id) a.setAttribute("aria-current", "true"); else a.removeAttribute("aria-current"); }); });
      }, { rootMargin: "-20% 0px -70% 0px" });
      ids.forEach(function (a) { var t = document.querySelector(a.getAttribute("href")); if (t) io.observe(t); });
    }
    if (!document.getElementById("chartConsistency")) return;
    fetch("./data/results.json", { cache: "no-cache" }).then(function (r) {
      if (!r.ok) throw new Error(r.status); return r.json();
    }).then(function (d) {
      state.data = d; fillMeta(); renderAll();
      document.documentElement.setAttribute("data-charts", "ready");
    }).catch(function (e) {
      document.querySelectorAll(".fig").forEach(function (n) { n.textContent = "Не удалось загрузить data/results.json (" + e.message + ")."; });
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init); else init();
})();
