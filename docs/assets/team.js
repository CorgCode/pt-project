// Команда и сроки: issues, milestones и PR берутся из открытого GitHub API при открытии страницы.
(function () {
  "use strict";
  var API = "https://api.github.com/repos/CorgCode/pt-project";
  var GH = "https://github.com/CorgCode/pt-project";
  var DAY = 864e5;
  var now = Date.now();

  function get(path) {
    return fetch(API + path, { headers: { Accept: "application/vnd.github+json" } }).then(function (r) {
      if (r.status === 403 || r.status === 429) throw new Error("GitHub временно ограничил запросы (60 в час с одного адреса). Обновите страницу позже.");
      if (!r.ok) throw new Error("GitHub ответил " + r.status + " на " + path);
      return r.json();
    });
  }

  function all(path, page, acc) {            // постранично, по 100
    page = page || 1; acc = acc || [];
    return get(path + (path.indexOf("?") < 0 ? "?" : "&") + "per_page=100&page=" + page).then(function (rows) {
      acc = acc.concat(rows);
      return rows.length === 100 && page < 10 ? all(path, page + 1, acc) : acc;
    });
  }

  // Срок milestone: «срок: ГГГГ-ММ-ДД ЧЧ:ММ» (Москва, UTC+3) в описании, иначе конец дня due_on по Москве.
  function deadline(m) {
    if (!m || !m.due_on) return null;
    var x = /срок:\s*(\d{4})-(\d{2})-(\d{2})\s+(\d{1,2}):(\d{2})/i.exec(m.description || "");
    if (x) return Date.UTC(+x[1], +x[2] - 1, +x[3], +x[4] - 3, +x[5]);
    var d = m.due_on.slice(0, 10).split("-");
    return Date.UTC(+d[0], +d[1] - 1, +d[2], 20, 59, 59);
  }

  function fmt(t) {
    return new Date(t).toLocaleString("ru-RU", { timeZone: "Europe/Moscow", day: "numeric", month: "short",
      hour: "2-digit", minute: "2-digit" });
  }

  function span(ms) {
    var d = Math.floor(ms / DAY);
    return d >= 1 ? d + " дн." : Math.max(1, Math.round(ms / 36e5)) + " ч";
  }

  // Статус задачи: kind — для подсчёта, text — для таблицы
  function status(issue, due) {
    var closed = issue.closed_at ? Date.parse(issue.closed_at) : null;
    if (closed && issue.state_reason === "not_planned") return { kind: "none", text: "закрыта без результата" };
    if (closed) {
      if (due === null) return { kind: "ok", text: "сдано" };
      var at = issue.delivered;                        // когда сдал исполнитель (см. delivery)
      if (at === null) return { kind: "ok", text: "сдано, время сдачи неизвестно (закрыл @" + issue.closer + ")" };
      if (at > due) return { kind: "late", text: "сдано с опозданием на " + span(at - due) };
      return { kind: "ok", text: "сдано в срок" };
    }
    if (due !== null && now > due) return { kind: "over", text: "просрочено на " + span(now - due) };
    return { kind: "work", text: due !== null ? "в работе, осталось " + span(due - now) : "в работе" };
  }

  // Время сдачи задачи со сроком. Закрыл сам исполнитель — время закрытия. Закрыл другой (например, тимлид
  // утром) — время закрытия не время сдачи: берём последний комментарий исполнителя до закрытия, иначе неизвестно.
  function delivery(issue) {
    var mine = (issue.assignees || []).map(function (a) { return a.login; });
    return get("/issues/" + issue.number).then(function (full) {
      issue.closer = full.closed_by ? full.closed_by.login : "?";
      if (!mine.length || mine.indexOf(issue.closer) >= 0) { issue.delivered = Date.parse(issue.closed_at); return; }
      return all("/issues/" + issue.number + "/comments").then(function (cs) {
        var t = cs.filter(function (c) { return mine.indexOf(c.user.login) >= 0 && c.created_at <= issue.closed_at; })
          .map(function (c) { return Date.parse(c.created_at); });
        issue.delivered = t.length ? Math.max.apply(null, t) : null;
      });
    });
  }

  function el(tag, attrs, kids) {
    var e = document.createElement(tag);
    Object.keys(attrs || {}).forEach(function (k) { e.setAttribute(k, attrs[k]); });
    [].concat(kids === undefined ? [] : kids).forEach(function (c) { e.append(c && c.nodeType ? c : document.createTextNode(c == null ? "" : String(c))); });
    return e;
  }
  function who(login) {
    return login ? el("a", { href: "https://github.com/" + login }, "@" + login) : el("span", { "class": "note" }, "не назначено");
  }
  function issueLink(i) { return el("a", { href: i.html_url }, "#" + i.number + " " + i.title); }
  function chip(s) { return el("span", { "class": "st " + s.kind }, s.text); }

  // По неделям: сколько коммитов в main, PR и комментариев в issues у каждого. Неделя — от предыдущего срока до своего.
  function weekly(milestones, issues, commits, pulls, comments) {
    var weeks = milestones.filter(function (m) { return m.due_on; }).map(function (m) { return { m: m, due: deadline(m) }; })
      .sort(function (a, b) { return a.due - b.due; });
    var shown = weeks.filter(function (w, i) { return i === 0 || weeks[i - 1].due < now; });
    var box = document.getElementById("weeks");
    if (!shown.length) { box.append(el("p", {}, "Сроков пока нет.")); return; }
    function bucket(t) {
      for (var i = 0; i < shown.length; i++) if (t <= shown[i].due && (i === 0 || t > shown[i - 1].due)) return i;
      return -1;
    }
    var act = {};
    function row(key) { return act[key] || (act[key] = shown.map(function () { return { c: 0, p: 0, m: 0 }; })); }
    function add(key, i, what) { if (i >= 0) row(key)[i][what]++; }
    issues.forEach(function (i) { (i.assignees || []).forEach(function (a) { row(a.login); }); });   // и те, у кого ноль
    commits.forEach(function (c) {
      var key = c.author && c.author.login ? c.author.login : "?" + c.commit.author.name;
      add(key, bucket(Date.parse(c.commit.author.date)), "c");
    });
    pulls.forEach(function (pr) { add(pr.user.login, bucket(Date.parse(pr.created_at)), "p"); });
    comments.forEach(function (c) { if (c.user.type !== "Bot") add(c.user.login, bucket(Date.parse(c.created_at)), "m"); });
    var head = el("tr", {}, [el("th", {}, "Участник")].concat(shown.map(function (w) {
      return el("th", {}, [el("a", { href: w.m.html_url }, "до " + fmt(w.due)), w.due > now ? el("br") : "", w.due > now ? el("span", { "class": "note" }, "идёт") : ""]);
    })));
    var tb = el("tbody");
    Object.keys(act).sort(function (a, b) { return (a[0] === "?") - (b[0] === "?") || a.localeCompare(b); }).forEach(function (k) {
      var name = k[0] === "?" ? el("span", {}, [k.slice(1) + " ", el("span", { "class": "note" }, "(git без привязки к GitHub)")]) : who(k);
      tb.append(el("tr", {}, [el("td", {}, name)].concat(act[k].map(function (x, i) {
        if (!x.c && !x.p && !x.m) return el("td", {}, shown[i].due > now ? el("span", { "class": "st work" }, "пока нет") : el("span", { "class": "st over" }, "ничего"));
        var parts = [];
        if (x.c) parts.push(x.c + " коммит.");
        if (x.p) parts.push(x.p + " PR");
        if (x.m) parts.push(x.m + " комм. в issues");
        return el("td", {}, el("span", { "class": "st ok" }, parts.join(" · ")));
      }))));
    });
    box.append(el("div", { "class": "tablewrap" }, el("table", {}, [el("thead", {}, head), tb])));
  }

  function render(milestones, issues, pulls) {
    issues = issues.filter(function (i) {
      return !i.pull_request && !(i.labels || []).some(function (l) { return l.name === "outdated"; });
    });
    var people = {};
    function person(login) {
      var k = login || "";
      return people[k] || (people[k] = { login: login, tasks: 0, done: 0, late: 0, over: 0, work: 0, merged: 0, openPr: 0 });
    }
    var rows = issues.map(function (i) {
      var due = deadline(i.milestone), s = status(i, due);
      var logins = (i.assignees || []).map(function (a) { return a.login; });
      (logins.length ? logins : [null]).forEach(function (l) {
        var p = person(l);
        p.tasks++;
        if (s.kind === "ok" || s.kind === "late") p.done++;
        if (s.kind === "late") p.late++;
        if (s.kind === "over") p.over++;
        if (s.kind === "work") p.work++;
      });
      return { issue: i, due: due, s: s, logins: logins };
    });
    pulls.forEach(function (pr) {
      if (pr.merged_at) person(pr.user.login).merged++;
      else if (pr.state === "open") person(pr.user.login).openPr++;
    });

    var tb = document.querySelector("#tPeople tbody");
    Object.keys(people).map(function (k) { return people[k]; })
      .sort(function (a, b) { return b.done - a.done || b.merged - a.merged || (a.login || "~").localeCompare(b.login || "~"); })
      .forEach(function (p) {
        tb.append(el("tr", {}, [el("td", {}, who(p.login))].concat(
          [p.tasks, p.done, p.late, p.over, p.work, p.merged, p.openPr].map(function (v) { return el("td", { "class": "num" }, v); }))));
      });

    var box = document.getElementById("milestones");
    var withDue = milestones.filter(function (m) { return m.due_on; })
      .sort(function (a, b) { return deadline(b) - deadline(a); });
    if (!withDue.length) box.append(el("p", {}, "Сроков пока нет: ни у одного milestone не указана дата."));
    withDue.forEach(function (m) {
      var due = deadline(m);
      var mine = rows.filter(function (r) { return r.issue.milestone && r.issue.milestone.number === m.number; });
      box.append(el("h3", {}, [el("a", { href: m.html_url }, m.title), " — срок " + fmt(due)]));
      if (!mine.length) { box.append(el("p", { "class": "note" }, "Задач в этом сроке нет.")); return; }
      var t = el("tbody");
      mine.forEach(function (r) {
        t.append(el("tr", {}, [el("td", {}, issueLink(r.issue)), el("td", {}, (r.logins.length ? r.logins : [null]).map(who)),
          el("td", {}, chip(r.s)), el("td", {}, r.issue.closed_at ? fmt(Date.parse(r.issue.closed_at)) : "—")]));
      });
      box.append(el("div", { "class": "tablewrap" }, el("table", {}, [el("thead", {}, el("tr", {},
        ["Задача", "Исполнитель", "Статус", "Закрыта"].map(function (h) { return el("th", {}, h); }))), t])));
    });

    var ob = document.querySelector("#tOpen tbody");
    rows.filter(function (r) { return r.issue.state === "open"; })
      .sort(function (a, b) { return (a.due === null) - (b.due === null) || a.due - b.due || a.issue.number - b.issue.number; })
      .forEach(function (r) {
        ob.append(el("tr", {}, [el("td", {}, issueLink(r.issue)), el("td", {}, (r.logins.length ? r.logins : [null]).map(who)),
          el("td", {}, r.due === null ? "—" : fmt(r.due)), el("td", {}, chip(r.s))]));
      });
    document.getElementById("updated").textContent = "Данные GitHub на " + fmt(now) + " (по Москве). Задач: " + issues.length +
      ", сроков: " + withDue.length + ".";
  }

  Promise.all([all("/milestones?state=all"), all("/issues?state=all"), all("/pulls?state=all"),
               all("/commits?since=2026-09-01T00:00:00Z"), all("/issues/comments?since=2026-09-01T00:00:00Z")])
    .then(function (r) {
      var dated = r[1].filter(function (i) { return !i.pull_request && i.closed_at && i.milestone && i.milestone.due_on; });
      return Promise.all(dated.map(delivery)).then(function () {
        render(r[0], r[1], r[2]);
        weekly(r[0], r[1].filter(function (i) { return !i.pull_request; }), r[3], r[2], r[4]);
      });
    })
    .catch(function (e) {
      document.getElementById("updated").textContent = "";
      var err = document.getElementById("err");
      err.style.display = "block";
      err.append(e.message + " Данные можно посмотреть прямо на GitHub: ", el("a", { href: GH + "/issues" }, "issues"), ".");
    });
})();
