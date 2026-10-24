// Таблица результатов: scoreboard/README.md из main (его пересобирает CI) показывается здесь как HTML.
(function () {
  "use strict";
  var SRC = "https://raw.githubusercontent.com/CorgCode/pt-project/main/scoreboard/README.md";
  var GH = "https://github.com/CorgCode/pt-project/blob/main/scoreboard/README.md";

  function esc(s) { return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;"); }

  // Строчная разметка: `код`, [текст](ссылка), **жирный**, *курсив*
  function inline(s) {
    var codes = [];
    s = esc(s).replace(/`([^`]+)`/g, function (_, c) { codes.push("<code>" + c + "</code>"); return "\u0000" + (codes.length - 1) + "\u0000"; });
    s = s.replace(/\[([^\]]+)\]\(([^)\s]+)\)/g, function (_, t, u) {
      if (!/^(https?:|#)/.test(u)) u = "https://github.com/CorgCode/pt-project/blob/main/scoreboard/" + u;
      return '<a href="' + u + '">' + t + "</a>";
    });
    s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>").replace(/(^|[^*])\*([^*]+)\*/g, "$1<em>$2</em>");
    return s.replace(/\u0000(\d+)\u0000/g, function (_, i) { return codes[+i]; });
  }

  function slug(t) { return t.toLowerCase().replace(/[^\p{L}\p{N}\s-]/gu, "").trim().replace(/\s+/g, "-"); }

  function cells(line) { return line.trim().replace(/^\||\|$/g, "").split("|").map(function (c) { return c.trim(); }); }

  function render(md) {
    var out = [], lines = md.split("\n"), i = 0, para = [];
    function flush() { if (para.length) out.push("<p>" + inline(para.join(" ")) + "</p>"); para = []; }
    while (i < lines.length) {
      var l = lines[i];
      var h = /^(#{1,3})\s+(.*)$/.exec(l);
      if (h) {
        flush();
        if (h[1].length > 1) {                          // # — заголовок страницы, он уже есть
          var n = Math.min(h[1].length, 3);
          out.push("<h" + n + ' id="' + slug(h[2]) + '">' + inline(h[2]) + "</h" + n + ">");
        }
        i++; continue;
      }
      if (/^\s*\|/.test(l) && i + 1 < lines.length && /^\s*\|?\s*:?-{3}/.test(lines[i + 1])) {
        flush();
        var head = cells(l), rows = [];
        i += 2;
        while (i < lines.length && /^\s*\|/.test(lines[i])) rows.push(cells(lines[i++]));
        out.push('<div class="tablewrap"><table><thead><tr>' + head.map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") +
          "</tr></thead><tbody>" + rows.map(function (r) {
            return "<tr>" + r.map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>";
          }).join("") + "</tbody></table></div>");
        continue;
      }
      if (!l.trim()) flush(); else para.push(l.trim());
      i++;
    }
    flush();
    return out.join("\n");
  }

  fetch(SRC, { cache: "no-cache" }).then(function (r) {
    if (!r.ok) throw new Error("GitHub ответил " + r.status);
    return r.text();
  }).then(function (md) {
    document.getElementById("board").innerHTML = render(md);
    document.getElementById("updated").textContent = "Таблица из репозитория (main), обновляется автоматически при каждой новой отправке.";
    if (location.hash) { var t = document.getElementById(decodeURIComponent(location.hash.slice(1))); if (t) t.scrollIntoView(); }
  }).catch(function (e) {
    document.getElementById("updated").textContent = "";
    var err = document.getElementById("err");
    err.style.display = "block";
    err.innerHTML = "Не удалось загрузить таблицу (" + esc(e.message) + "). Её можно посмотреть на <a href=\"" + GH + "\">GitHub</a>.";
  });
})();
