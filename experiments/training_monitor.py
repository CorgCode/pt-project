"""Read-only интерактивный монитор фоновых экспериментов, без новых зависимостей."""
from __future__ import annotations

import argparse
import curses
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


DEFAULT_WORK = Path('/Volumes/KINGSTON/PT_PROJECT_ML/fingerprint_expanded')


def current_work():
    candidates = [DEFAULT_WORK, DEFAULT_WORK.parent/'fingerprint_memory', DEFAULT_WORK.parent/'fingerprint_quick',
                  DEFAULT_WORK.parent/'fingerprint_donors']
    existing = [p for p in candidates if (p/'state.json').exists()]
    return max(existing,key=lambda p:(p/'state.json').stat().st_mtime) if existing else DEFAULT_WORK


STAGES = {
    'donor_inputs': 'Подготовка исторических переходов и полезных соседей',
    'donor_training': 'Обучение прогноза новых собеседников',
    'waiting_for_download13': 'Ожидание скачивания итогового дня 13',
    'pca': 'Подготовка сжатых представлений без обучения нейросети',
    'export_model': 'Сохранение выбранной модели',
    'download13': 'Скачивание нового итогового дня 13',
    'prepare_day13': 'Подготовка итогового дня 13',
    'extracting_32_hours': 'Извлечение 32 часов исторического трафика',
    'fit_bank': 'Подготовка признаков и TF-IDF',
    'training_inputs': 'Подготовка обучающих признаков',
    'teacher': 'Расчёт признаков прежней модели',
    'parallel_training': 'Обучение нейросетей',
    'reference_validation': 'Подготовка эталона validation',
    'search': 'Сравнение вариантов на validation',
    'validation': 'Сравнение вариантов на validation',
    'winner_frozen': 'Победитель зафиксирован',
    'waiting_for_download12': 'Ожидание скачивания дня 12',
    'prepare_day11': 'Подготовка итогового дня 11',
    'prepare_day12': 'Подготовка итогового дня 12',
    'reference_test': 'Подготовка итогового эталона',
    'final_evaluation': 'Итоговая проверка на отложенном дне',
    'complete': 'Прогон завершён',
    'failed': 'Прогон остановлен с ошибкой',
}


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text())
    except (OSError, ValueError):
        return default


def tail(path, limit=96_000):
    try:
        with Path(path).open('rb') as stream:
            stream.seek(0, os.SEEK_END)
            end = stream.tell()
            start = max(0, end-limit)
            stream.seek(start)
            raw = stream.read()
        if start:
            raw = raw.partition(b'\n')[2]
        return raw.decode('utf-8', errors='replace').splitlines()
    except OSError:
        return []


def last_record(path, field):
    for line in reversed(tail(path)):
        try:
            value = json.loads(line)
            if isinstance(value, dict) and field in value:
                return value
        except ValueError:
            pass
    return {}


def duration(seconds):
    seconds = max(0, int(seconds))
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    return f'{hours}ч {minutes:02d}м' if hours else f'{minutes}м {seconds:02d}с'


def bar(value, total, width=20):
    fraction = max(0.0, min(1.0, value / total)) if total else 0
    filled = round(width*fraction)
    return '[' + '#' * filled + '.' * (width-filled) + f'] {fraction:5.1%}'


class Resources:
    def __init__(self):
        self.processes = {}
        self.warmed = False
        psutil.cpu_percent(percpu=True)

    def sample(self, work):
        root = None
        try:
            pid_path = work / 'coordinator.pid'
            candidate = psutil.Process(int(pid_path.read_text().strip()))
            if candidate.create_time() <= pid_path.stat().st_mtime + 5 and candidate.status() != psutil.STATUS_ZOMBIE:
                root = candidate
        except (OSError, ValueError, psutil.Error):
            pass
        found = []
        if root:
            try:
                found = [root, *root.children(recursive=True)]
            except psutil.Error:
                found = [root]
        live, cpu, rss = {}, 0.0, 0
        for process in found:
            try:
                old = self.processes.get(process.pid)
                if old and old.create_time() == process.create_time():
                    process = old
                cpu += process.cpu_percent()
                rss += process.memory_info().rss
                live[process.pid] = process
            except psutil.Error:
                pass
        measured = self.warmed
        self.processes = live
        self.warmed = True
        memory, swap = psutil.virtual_memory(), psutil.swap_memory()
        return {'running': bool(root), 'pids': set(live), 'processes': live,
                'cores': psutil.cpu_percent(percpu=True), 'job_cpu': cpu if measured else None,
                'rss': rss/2**30, 'ram_total': memory.total/2**30,
                'ram_available': memory.available/2**30, 'swap': swap.used/2**30}


def snapshot(work, sampler):
    protocol = read_json(work/'protocol.json', {})
    state = read_json(work/'state.json', {})
    results = read_json(work/'validation_results.json', [])
    resources = sampler.sample(work)
    historical = read_json(work/'bank_metadata.json', {}).get('training_days', [])
    days = []
    for day in protocol.get('train_days', range(2,10)):
        complete = read_json(work/f'day{day:02d}/complete.json', {})
        if day in historical and (work/'bank.pkl').exists():
            complete = complete or {'day':day}
        record = last_record(work/f'extract{day:02d}.log', 'raw_rows')
        pid = state.get('workers', {}).get(f'extract{day:02d}')
        status = 'готово' if complete else ('обрабатывается' if pid in resources['pids'] else 'ожидает')
        if not complete and record and not resources['running']:
            status = 'прервано'
        days.append({'day':day, 'status':status, **(complete or record)})
    heads = []
    labels = protocol.get('head_labels')
    for dim in (labels if labels is not None else protocol.get('dimensions', [128,256])):
        label = str(dim) if labels is not None else f'bag{dim}'
        record = last_record(work/f'{label}.log', 'epoch')
        complete = read_json(work/f'{label}/complete.json', {})
        checkpoints = complete.get('epochs') or protocol.get('checkpoints', [40,120])
        target = max(checkpoints)
        epoch = target if complete else record.get('epoch',0)
        pid = state.get('workers',{}).get(label,state.get('workers',{}).get(str(dim)))
        if not pid:
            # Поддержка координатора, который продолжает уже работающие головы.
            for candidate in state.get('adopted_workers',[]):
                try:
                    args = resources['processes'][candidate].cmdline()
                    if '--dim' in args and args[args.index('--dim')+1] == str(dim):
                        pid = candidate
                except (KeyError, ValueError, IndexError, psutil.Error):
                    pass
        status = 'готово' if complete else ('обучается' if pid in resources['pids'] else 'ожидает')
        eta = None
        if epoch and not complete and pid in resources['processes']:
            elapsed = time.time()-resources['processes'][pid].create_time()
            eta = elapsed/epoch*(target-epoch)
        heads.append({'dim':dim,'epoch':epoch,'target':target,'loss':record.get('loss'), 'status':status,'eta':eta})
    logs = sorted([p for p in work.glob('*.log') if not p.name.startswith(('network','download'))])
    preferred = work / f"{state.get('stage','coordinator')}.log"
    if not preferred.exists():
        preferred = work / ('search.log' if state.get('stage')=='validation' else 'coordinator.log')
    # Новые worker-логи полезнее редко обновляемого coordinator.log.
    if state.get('stage') in {'extracting_32_hours','parallel_training','donor_training'}:
        active_logs = [work/f'{name}.log' for name in state.get('workers',{})]
        active_logs = [p for p in active_logs if p.exists()]
        if active_logs:
            preferred = max(active_logs, key=lambda p:p.stat().st_mtime)
    return {'work':work,'protocol':protocol,'state':state,'resources':resources,
            'days':days,'heads':heads,'results':results,'logs':logs,'preferred':preferred,
            'final':read_json(work/'final/summary.json',{}),
            'inputs':read_json(work/'inputs_ready.json',{}),
            'input_record':last_record(work/'donor_inputs.log','prepared_peer_day')}


def overview(s):
    r, state, protocol = s['resources'], s['state'], s['protocol']
    stage=state.get('stage','нет state.json')
    status = 'РАБОТАЕТ' if r['running'] else ('ЗАВЕРШЁН' if stage=='complete' else 'НЕ ЗАПУЩЕН / ОСТАНОВЛЕН')
    lines = [f"{status}  |  {STAGES.get(stage,stage)}",
             f"Стадия обновлена: {state.get('at','—')}",
             f"Цель: {protocol.get('target_score',80)}  |  validation: день {protocol.get('validation_day','—')}  |  test: день {protocol.get('test_day',protocol.get('final_test_day','—'))}", '']
    if state.get('error'): lines += [f"ОШИБКА: {state['error']}",'']
    done = sum(d['status']=='готово' for d in s['days'])
    if s['days'] and not protocol.get('head_labels'):
        lines += [f"Дни подготовлены: {bar(done,len(s['days']))}  {done}/{len(s['days'])}"]
    if protocol.get('head_labels'):
        if s['inputs']:
            lines.append(f"Исторические переходы готовы: {len(s['inputs'].get('pairs',[]))}; хостов {s['inputs'].get('n_hosts','—')}")
        elif s['input_record']:
            lines.append(f"Профили контактов подготовлены по день {s['input_record']['prepared_peer_day']:02d}")
    for day in s['days']:
        if day['status']=='обрабатывается':
            lines.append(f"  Сейчас день {day['day']:02d}: прочитано {day.get('raw_rows',0):,} строк")
    for head in s['heads']:
        loss = f"  loss {head['loss']:.4f}" if head['loss'] is not None else ''
        eta = f"  осталось ≈{duration(head['eta'])}" if head['eta'] is not None else ''
        lines.append(f"Голова {head['dim']}: {bar(head['epoch'],head['target'])}  {head['epoch']}/{head['target']}  {head['status']}{loss}{eta}")
    count, total=len(s['results']), protocol.get('candidates',20)
    lines += [f'Validation: {bar(count,total)}  {count}/{total}']
    if state.get('candidate'): lines.append(f"Сейчас проверяется: {state['candidate']}")
    if s['results']:
        best = max(s['results'],key=lambda x:x['score'])
        lines.append(f"Лучший validation: {best['score']:.3f}  {best['name']}")
    if s['final']:
        final=s['final']; lines += ['',f"ИТОГОВЫЙ TEST: {final['scores']['winner']:.3f}  |  прирост к исходной модели {final['gain']:+.3f}"]
    if state.get('result')=='no_material_validation_gain':
        lines += ['',f"Значимого для продолжения прироста на validation нет: {state.get('validation_gain',0):+.3f}.",
                  'Новый test не открывался. Подробности: selection_summary.json.']
    job_cpu = (f"{r['job_cpu']/max(1,len(r['cores'])):.0f}% мощности CPU" if r['job_cpu'] is not None
               else ('первый замер…' if r['running'] else 'нет активных процессов'))
    lines += ['',f"CPU всего: {sum(r['cores'])/max(1,len(r['cores'])):.0f}%  |  процессы прогона: {job_cpu}",
              'Ядра: '+'  '.join(f'{i+1}:{load:.0f}%' for i,load in enumerate(r['cores'])),
              f"RAM свободно: {r['ram_available']:.1f}/{r['ram_total']:.1f} ГБ  |  RSS прогона: {r['rss']:.1f} ГБ  |  swap: {r['swap']:.2f} ГБ"]
    try:
        disk=psutil.disk_usage(s['work'])
        lines.append(f'На диске с данными свободно: {disk.free/2**30:.1f} ГБ')
    except OSError: pass
    lines += ['', 'Баллы многодневного протокола не равны баллам исходной таблицы.']
    return lines


def training(s):
    lines=['ПОДГОТОВКА ДАННЫХ', '']
    if s['protocol'].get('head_labels'):
        for pair in s['inputs'].get('pairs',[]):
            lines.append(f"Дни {pair['source_day']:02d} → {pair['target_day']:02d}: {pair['active_queries']:,} запросов с полезными соседями")
        if not s['inputs']: lines.append('Готовятся исторические признаки и цели; подробности на вкладке4.')
    else:
        for day in s['days']:
            lines.append(f"День {day['day']:02d}: {day['status']:<16}  прочитано {day.get('raw_rows',0):,} строк  |  в нужных часах {day.get('kept',0):,}")
    lines += ['', 'ОБУЧЕНИЕ', '']
    for h in s['heads']:
        lines.append(f"Голова {h['dim']}: {h['epoch']}/{h['target']} эпох; {h['status']}; loss: {h['loss'] if h['loss'] is not None else '—'}")
    lines += ['', 'Эпоха и loss обновляются раз в 10 эпох.', 'Оценка оставшегося времени приблизительная; она не включает validation/test.']
    return lines


def results(s):
    lines=['VALIDATION: подбор модели, не итоговый test', '', 'Балл      Identity    Модель', '-'*78]
    for row in sorted(s['results'],key=lambda x:-x['score']):
        identity=f"{row['identity']:.1%}" if row.get('identity') is not None else '—'
        lines.append(f"{row['score']:7.3f}   {identity:>7}    {row['name']}")
    if not s['results']: lines.append('Пока нет оценённых вариантов.')
    if s['final']:
        lines += ['',f"ИТОГОВЫЙ TEST: {s['final'].get('test','')}"]
        for name, score in s['final']['scores'].items(): lines.append(f'{name:<15} {score:.3f}')
        lines.append(f"Отчёт: {s['work']/'final/REPORT.md'}")
    return lines


def interactive(screen, work, interval):
    try: curses.curs_set(0)
    except curses.error: pass
    screen.keypad(True); screen.timeout(150)
    sampler=Resources()
    page, paused, offset, selected_log = 0, False, 0, None
    last_update, data = 0, None
    while True:
        now=time.monotonic()
        if data is None or (not paused and now-last_update>=interval):
            data=snapshot(work,sampler); last_update=now
        height,width=screen.getmaxyx()
        screen.erase()
        def put(y,text,style=0):
            if 0<=y<height and width>1:
                # Удаляет управляющие символы из логов; UTF-8 русский текст сохраняется.
                text=''.join(c if c.isprintable() else ' ' for c in str(text))
                try: screen.addnstr(y,0,text,max(1,width-1),style)
                except curses.error: pass
        put(0, f"ОБУЧЕНИЕ · {datetime.now():%H:%M:%S}  {'[ОБНОВЛЕНИЕ НА ПАУЗЕ]' if paused else f'каждые {interval:g} с'}",curses.A_BOLD)
        put(1,'  '.join(f"{'>' if i==page else ' '}{i+1} {name}" for i,name in enumerate(['Обзор','Дни / эпохи','Результаты','Логи'])),curses.A_REVERSE)
        put(2,str(work),curses.A_DIM)
        if page==3:
            logs=data['logs']
            log=selected_log if selected_log in logs else data['preferred']
            lines=[f'Лог: {log.name}  |  [ и ] сменить лог; ↑/↓ прокрутка','']
            content=tail(log)
            room=max(1,height-9)
            end=max(0,len(content)-offset)
            lines+=content[max(0,end-room):end]
        else:
            lines=[overview,training,results][page](data)
            lines=lines[offset:]
        for i,line in enumerate(lines[:max(0,height-6)]): put(4+i,line)
        put(height-2,'1–4 / Tab: раздел   r: обновить   Space: пауза монитора   ↑/↓: строки   q: выход',curses.A_REVERSE)
        put(height-1,'Монитор только читает файлы. Выход и пауза не останавливают обучение.',curses.A_DIM)
        screen.refresh()
        key=screen.getch()
        if key in (ord('q'),ord('Q'),27): break
        if key in (ord('1'),ord('2'),ord('3'),ord('4')):
            page=key-ord('1'); offset=0
        elif key==9: page=(page+1)%4; offset=0
        elif key==ord(' '): paused=not paused
        elif key in (ord('r'),ord('R')): data=None; offset=0
        elif key==curses.KEY_UP: offset=max(0,offset-1) if page!=3 else offset+1
        elif key==curses.KEY_DOWN: offset=offset+1 if page!=3 else max(0,offset-1)
        elif key in (curses.KEY_HOME,ord('g')): offset=0
        elif key in (ord('['),ord(']')) and data['logs']:
            logs=data['logs']; current=selected_log if selected_log in logs else data['preferred']
            index=logs.index(current) if current in logs else 0
            selected_log=logs[(index+(-1 if key==ord('[') else 1))%len(logs)]; offset=0


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work',type=Path,default=current_work(),help='Каталог прогона (по умолчанию самый свежий)')
    ap.add_argument('--interval',type=float,default=2,help='Обновление, секунды (по умолчанию 2)')
    ap.add_argument('--once',action='store_true',help='Один текстовый снимок вместо интерактивного экрана')
    args=ap.parse_args()
    if args.interval<0.2: ap.error('--interval должен быть не меньше 0.2 секунды')
    if not args.work.is_dir(): ap.error(f'Нет каталога прогона: {args.work}')
    if args.once or not (sys.stdin.isatty() and sys.stdout.isatty()):
        data=snapshot(args.work,Resources())
        print('\n'.join([str(args.work),'',*overview(data),'',*training(data),'',*results(data)]))
    else:
        try: curses.wrapper(interactive,args.work,args.interval)
        except KeyboardInterrupt: pass


if __name__=='__main__':
    main()
