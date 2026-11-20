/* Operator-owned labels and decisions. No automatic incident verdicts. */
window.createCorgWorkflow = function (context) {
  const $ = id => document.getElementById(id);
  const labels = {new: 'Не разобрано', review: 'В работе', done: 'Проверено'};
  const own = (object, key) => Object.prototype.hasOwnProperty.call(object, key);
  let report = null;

  function annotation(id) {
    const data = context().payload.annotations;
    const item = own(data, id) ? data[id] : {};
    return {label: '', status: 'new', tags: [], watch: false, ...item};
  }

  function setAnnotation(id, patch) {
    const c = context();
    c.payload.annotations = {...c.payload.annotations, [id]: {...annotation(id), ...patch}};
    c.dirty();
    syncSelected();
    c.renderList();
    renderWatch();
  }

  function syncSelected() {
    const c = context();
    if (!c.selected) return;
    const meta = annotation(c.selected.id);
    $('toggle-watch').textContent = meta.watch ? '★ Под наблюдением' : '☆ Под наблюдение';
    $('toggle-watch').setAttribute('aria-pressed', String(meta.watch));
    $('selected-status').textContent = labels[meta.status];
    $('host-label').value = meta.label;
    $('host-status').value = meta.status;
    $('host-tags').value = meta.tags.join(', ');
  }

  function matches(host, query) {
    const meta = annotation(host.id);
    return [host.id, meta.label, ...meta.tags].join(' ').toLocaleLowerCase('ru').includes(query);
  }

  function directory(rows) {
    const c = context(), status = $('status-filter').value, watched = $('watch-only').checked;
    rows = rows.filter(host => {
      const meta = annotation(host.id);
      return (!status || meta.status === status) && (!watched || meta.watch);
    });
    const order = $('host-sort').value;
    return rows.sort((a, b) => {
      if (order === 'similarity') return c.math.cosine(c.selected, b) - c.math.cosine(c.selected, a) || a.id.localeCompare(b.id);
      if (order === 'activity') return (b.activity?.outgoing + b.activity?.incoming || 0) - (a.activity?.outgoing + a.activity?.incoming || 0) || a.id.localeCompare(b.id);
      return a.id.localeCompare(b.id, 'ru', {numeric: true});
    });
  }

  function renderWatch() {
    const c = context(), rows = c.state.dataset.filter(host => annotation(host.id).watch);
    $('watch-count').textContent = `Под наблюдением: ${rows.length}. В работе: ${rows.filter(h => annotation(h.id).status === 'review').length}.`;
    $('export-watch').disabled = !rows.length;
    $('watch-list').replaceChildren(...rows.map(host => {
      const meta = annotation(host.id), row = c.element('article', undefined, 'panel watch-row');
      const heading = c.element('div');
      heading.append(c.element('h3', meta.label || host.id), c.element('p', host.id, 'small'));
      const badge = c.element('span', labels[meta.status], 'badge');
      const copy = c.element('p', meta.tags.length ? meta.tags.join(' · ') : 'Без меток', 'small');
      const actions = c.element('div', undefined, 'report-actions');
      const open = c.element('button', 'Открыть разбор ↗', 'glass');
      open.type = 'button';
      open.addEventListener('click', () => { c.select(host.id); c.view('case'); });
      const remove = c.element('button', 'Снять отметку', 'quiet');
      remove.type = 'button';
      remove.addEventListener('click', () => setAnnotation(host.id, {watch: false}));
      actions.append(open, remove);
      row.append(heading, badge, copy, actions);
      return row;
    }));
    if (!rows.length) {
      const empty = c.element('div', undefined, 'panel empty');
      empty.append(c.element('h3', 'Нужное устройство легко сохранить.'), c.element('p', 'Выберите его в каталоге и нажмите «Под наблюдение». Метки и заметки остаются в рабочем наборе.'));
      $('watch-list').append(empty);
    }
  }

  function changeRows(next) {
    report = next;
    const c = context(), query = $('change-search').value.trim().toLocaleLowerCase('ru');
    const type = $('change-filter').value;
    let threshold = Number($('change-threshold').value);
    if (!Number.isFinite(threshold)) threshold = .95;
    threshold = Math.max(-1, Math.min(1, threshold));
    $('change-threshold-label').hidden = type !== 'profile';
    $('change-filter').querySelector('option[value="contacts"]').disabled = next.kind !== 'flows';
    const rows = next.changes.filter(change => matches({id: change.id}, query) && (
      type === 'all' || type === 'contacts' && change.added.length + change.removed.length > 0 ||
      type === 'group' && change.beforeGroup !== change.afterGroup || type === 'profile' && change.similarity < threshold
    ));
    $('change-filter-count').textContent = `Показано ${rows.length} из ${next.changes.length} общих устройств.`;
    $('change-cohort').replaceChildren();
    for (const [title, ids] of [['Появились во втором окне', next.newHosts], ['Отсутствуют во втором окне', next.missingHosts]]) {
      const group = c.element('details');
      group.append(c.element('summary', `${title} · ${ids.length}`));
      const list = c.element('div', undefined, 'chips');
      if (!ids.length) list.append(c.element('p', 'Нет', 'small'));
      ids.forEach(id => {
        if (c.state.dataset.some(host => host.id === id)) {
          const button = c.element('button', id + ' ↗');
          button.type = 'button';
          button.addEventListener('click', () => { c.select(id); c.view('devices'); });
          list.append(button);
        } else list.append(c.element('span', id, 'badge'));
      });
      group.append(list);
      $('change-cohort').append(group);
    }
    return rows;
  }

  function documentReport(title) {
    const doc = document.implementation.createHTMLDocument('CorgCode · ' + title);
    doc.documentElement.lang = 'ru';
    const charset = doc.createElement('meta'); charset.setAttribute('charset', 'utf-8'); doc.head.append(charset);
    const meta = doc.createElement('meta');
    meta.name = 'viewport'; meta.content = 'width=device-width,initial-scale=1'; doc.head.append(meta);
    const style = doc.createElement('style');
    style.textContent = ':root{--ink:#181520;--muted:#626671;--paper:#f5f5fb;--accent:#8231fe;--line:#18152025;--glass:#ffffff;--font:Arial,sans-serif;}*{box-sizing:border-box;min-width:0;}html,body{overflow-x:clip;}body{max-width:1050px;margin:auto;padding:40px 24px;background:var(--paper);color:var(--ink);font:15px/1.7 var(--font);}h1,h2,h3{line-height:1.2;overflow-wrap:anywhere;}h2{margin-top:38px;}p,td,th{overflow-wrap:anywhere;}small,figcaption{color:var(--muted);}table{width:100%;border-collapse:collapse;font-size:13px;}th,td{text-align:left;padding:12px 8px;border-bottom:1px solid var(--line);}th:first-child,td:first-child{padding-left:0;}svg{display:block;width:100%;height:auto;max-height:540px;}svg text{font:13px var(--font);fill:var(--ink);}svg .node-sub,svg .edge-value{font-size:10px;}svg .edge{stroke:var(--line);fill:none;}svg .node rect{fill:var(--glass);stroke:var(--line);}svg .node.selected rect{fill:var(--ink);}svg .node.selected text{fill:var(--paper);}blockquote{margin:20px 0;padding:20px;border-left:3px solid var(--accent);background:var(--glass);white-space:pre-wrap;overflow-wrap:anywhere;}figure{margin:24px 0;}@media print{body{padding:0;background:var(--glass);}h2{break-after:avoid;}tr{break-inside:avoid;}svg{max-height:360px;}}';
    doc.head.append(style);
    const add = (tag, text, parent = doc.body) => { const node = doc.createElement(tag); node.textContent = text; parent.append(node); return node; };
    const table = (headers, rows) => {
      const table = doc.createElement('table'), head = doc.createElement('thead'), body = doc.createElement('tbody');
      const tr = doc.createElement('tr'); headers.forEach(text => add('th', text, tr)); head.append(tr);
      rows.forEach(row => { const tr = doc.createElement('tr'); row.forEach(text => add('td', String(text), tr)); body.append(tr); });
      table.append(head, body); doc.body.append(table);
    };
    add('small', 'CorgCode · Рабочее место SOC'); add('h1', title);
    const c = context();
    add('p', `Набор: ${c.payload.name}. ${c.payload.sample ? 'Учебный пример.' : 'Данные пользователя.'} Создано: ${new Date().toLocaleString('ru-RU')}.`);
    return {doc, add, table, finish(name) {
      add('p', 'Сходство и изменения помогают выбрать направление разбора. Они не доказывают одинаковую роль устройства или наличие атаки. Статусы, подписи и метки назначены оператором.');
      c.download('<!doctype html>\n' + doc.documentElement.outerHTML, name, 'text/html;charset=utf-8');
      c.status('Читаемый HTML-отчёт сохранён. Его можно открыть без сервера и распечатать в PDF.');
    }};
  }

  function exportReadable() {
    const c = context(), host = c.selected, meta = annotation(host.id);
    const out = documentReport(meta.label ? `${meta.label} · ${host.id}` : `Разбор устройства ${host.id}`);
    out.add('p', `Статус: ${labels[meta.status]}. Под наблюдением: ${meta.watch ? 'да' : 'нет'}. Метки: ${meta.tags.join(', ') || 'нет'}.`);
    out.add('h2', 'Наблюдение оператора');
    const note = own(c.payload.notes, host.id) ? c.payload.notes[host.id] : '';
    out.add('blockquote', note || 'Заметка не добавлена.');
    if (host.activity) {
      out.add('h2', 'Активность в текущем окне');
      out.table(['Показатель', 'Значение'], [['Исходящих записей', host.activity.outgoing], ['Входящих записей', host.activity.incoming], ['Контактов', host.activity.contacts], ['Отправлено байт', host.activity.sentBytes], ['Получено байт', host.activity.receivedBytes]]);
    }
    out.add('h2', c.mode === 'traffic' ? 'Наблюдавшиеся контакты' : 'Похожие профили');
    const figure = out.doc.createElement('figure'), diagram = $('network').cloneNode(true);
    diagram.setAttribute('viewBox', innerWidth < 700 ? '0 0 360 600' : '0 0 760 470');
    diagram.querySelectorAll('[role="button"]').forEach(node => { node.removeAttribute('role'); node.removeAttribute('tabindex'); });
    figure.append(diagram); out.add('figcaption', $('map-help').textContent + ' Положение узлов условно.', figure); out.doc.body.append(figure);
    out.add('h2', 'Ближайшие по всем координатам');
    out.table(['Устройство', 'Косинусное сходство'], c.nearest().slice(0, 10).map(n => [n.host.id, n.score.toFixed(4)]));
    out.add('p', `Сравниваемая пара: ${host.id} и ${c.comparator.id}. Сходство: ${c.math.cosine(host, c.comparator).toFixed(4)}.`);
    if (c.state.kind === 'flows') {
      out.add('h2', 'Контакты из исходных записей');
      out.table(['Устройство', 'Записей обмена', 'Байт в обоих направлениях'], c.contacts(c.state, host.id).map(contact => [contact.id, contact.flows, contact.bytes]));
    } else out.add('p', 'Готовые векторы не содержат исходный обмен; контакты по ним не восстановлены.');
    out.finish('corgcode-report.html');
  }

  $('toggle-watch').addEventListener('click', () => { const id = context().selected.id; setAnnotation(id, {watch: !annotation(id).watch}); });
  $('host-label').addEventListener('input', () => setAnnotation(context().selected.id, {label: $('host-label').value}));
  $('host-status').addEventListener('change', () => setAnnotation(context().selected.id, {status: $('host-status').value}));
  $('host-tags').addEventListener('input', () => context().dirty());
  $('host-tags').addEventListener('change', () => {
    const tags = [...new Set($('host-tags').value.split(',').map(s => s.trim()).filter(Boolean))];
    if (tags.length > 8 || tags.some(tag => tag.length > 32)) {
      context().status('До 8 меток, каждая не длиннее 32 символов. Прежние метки сохранены.', true); syncSelected(); return;
    }
    setAnnotation(context().selected.id, {tags});
  });
  ['status-filter', 'host-sort', 'watch-only'].forEach(id => $(id).addEventListener('change', () => context().renderList()));
  ['change-search', 'change-filter', 'change-threshold'].forEach(id => $(id).addEventListener(id === 'change-filter' ? 'change' : 'input', () => context().renderChanges()));
  $('export-readable').addEventListener('click', exportReadable);
  $('export-watch').addEventListener('click', () => {
    const c = context(), out = documentReport('Устройства под наблюдением');
    out.table(['Устройство', 'Подпись', 'Статус', 'Метки', 'Заметка'], c.state.dataset.filter(h => annotation(h.id).watch).map(host => {
      const meta = annotation(host.id); return [host.id, meta.label, labels[meta.status], meta.tags.join(', '), own(c.payload.notes, host.id) ? c.payload.notes[host.id] : ''];
    })); out.finish('corgcode-watchlist.html');
  });
  $('export-changes').addEventListener('click', () => {
    if (!report) return;
    const out = documentReport('Изменения между двумя окнами');
    out.add('p', `Общих устройств: ${report.changes.length}. Новых: ${report.newHosts.length}. Отсутствуют во втором окне: ${report.missingHosts.length}. Отчёт содержит все общие устройства независимо от фильтра на экране.`);
    out.table(['Устройство', 'Сходство', 'Группа: до → после', 'Новые контакты', 'Больше не наблюдались'], report.changes.map(c => [c.id, c.similarity.toFixed(4), `${c.beforeGroup + 1} → ${c.afterGroup + 1}`, report.kind === 'flows' ? c.added.join(', ') || 'Нет' : 'Нет записей', report.kind === 'flows' ? c.removed.join(', ') || 'Нет' : 'Нет записей']));
    out.add('h2', 'Появились во втором окне'); out.add('p', report.newHosts.join(', ') || 'Нет');
    out.add('h2', 'Отсутствуют во втором окне'); out.add('p', report.missingHosts.join(', ') || 'Нет');
    out.add('p', 'Группы второго окна назначены относительно центров первого. Для трафика используется общая шкала признаков. Отсутствие устройства или контакта в CSV не доказывает отключение.'); out.finish('corgcode-changes.html');
  });
  return {annotation, syncSelected, matches, directory, renderWatch, changeRows};
};
