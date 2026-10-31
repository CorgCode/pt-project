"""Две параллельные raw-token модели; выбор на дне10, однократный test11."""
from __future__ import annotations

import argparse
import copy
import gc
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import cloudpickle
import numpy as np
import psutil
import scipy.sparse as sp

from experiments.multiday.bag import BagHybrid, hash_sparse
from experiments.multiday.memory import FingerprintHybrid
from experiments.multiday.finish import freeze_selection
from experiments.multiday.gain import evaluate
from experiments.multiday.reference import build_reference
from experiments.overnight.model import T
from pipeline.data import ProcessedData


# корень рабочих данных: архивы, подготовленные окна и итоги этапов (по умолчанию — SSD)
SOURCE = Path(os.environ.get('PT_ML_ROOT', '/Volumes/KINGSTON/PT_PROJECT_ML'))
BASE = SOURCE / 'multiday/processed'
INCUMBENT = SOURCE / 'multiday_gain/extended/final/model.pkl'
ROOT = Path(__file__).resolve().parents[2]


def load(path):
    with Path(path).open('rb') as stream:
        return cloudpickle.load(stream)


def dump(value, path):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    with temp.open('wb') as stream:
        cloudpickle.dump(value, stream)
    temp.replace(path)


def write_json(path, value):
    path = Path(path)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def state(work, stage, **details):
    value = {'stage': stage, 'at': datetime.now().astimezone().isoformat(), **details}
    write_json(work / 'state.json', value)
    print(json.dumps(value, ensure_ascii=False), flush=True)


def slim_base(model):
    for name in ['_xs', '_ok', '_active', '_training_counts']:
        model.__dict__.pop(name, None)
    model._parts_cache = {}
    model._evaluation_cache, model._behavior_cache = {}, {}
    return model


class ArrayEmbeddings:
    """Тот же вывод модели, заранее рассчитанный для общего validation."""
    def __init__(self, train, later):
        self.train, self.later = train, later

    def embed(self, data, window, hosts):
        return (self.train if window == 'train' else self.later).loc[hosts]


def prepare_inputs(work):
    if (work / 'inputs_ready.json').exists():
        return
    bank = load(work / 'bank.pkl')
    rng = np.random.default_rng(883)
    buckets = rng.integers(0, 131072, size=bank.memory.shape[1], dtype=np.int32)
    np.save(work / 'buckets.npy', buckets)
    sp.save_npz(work / 'training.npz', hash_sparse(bank.memory, buckets, 131072))
    write_json(work / 'hosts.json', bank.hosts)
    model = slim_base(load(INCUMBENT))
    data = ProcessedData(BASE)
    teacher = model.embed(data, 'train', bank.hosts).to_numpy()
    np.save(work / 'teacher.npy', teacher)
    write_json(work / 'inputs_ready.json', {'train_days': list(range(2, 10)),
        'observations': bank.memory.shape[0], 'raw_tokens': bank.memory.shape[1],
        'teacher': str(INCUMBENT), 'validation_used_for_training': False})


def score_candidates(work):
    data = ProcessedData(work / 'paired_validation')
    ref = work / 'reference_validation/validation'
    base = slim_base(load(INCUMBENT))
    bank = load(work / 'bank.pkl')
    buckets = np.load(work / 'buckets.npy')
    hosts, later_hosts = data.cohort, data.evaluable('validation')
    # Все кандидаты используют один и тот же исторический компонент.
    train_base = base.embed(data, 'train', hosts)
    later_base = base.embed(data, 'validation', later_hosts)
    candidates, records = [], []

    def assess(name, model, arrays=None):
        out = work / 'candidates' / name
        out.mkdir(parents=True, exist_ok=True)
        candidates.append(out)
        if (out / 'best.json').exists():
            records.append(json.loads((out / 'best.json').read_text()))
            return
        if psutil.virtual_memory().available < 2 * 1024**3:
            raise MemoryError('Перед оценкой осталось менее 2 ГБ доступной RAM')
        state(work, 'validation', candidate=name, completed=len(records), total=20)
        score, result = evaluate(arrays or model, data, 'validation', ref, out, name)
        record = {'name': name, 'score': score, 'identity': result['metrics']['identity_top1']}
        # Сохраняется только текущий лучший checkpoint; результаты всех вариантов остаются.
        if not records or score > max(r['score'] for r in records):
            for previous in candidates[:-1]:
                (previous / 'best.pkl').unlink(missing_ok=True)
            if name == 'incumbent':
                import shutil
                shutil.copy2(INCUMBENT, out / 'best.pkl')
            else:
                model._encoded = {}
                slim_base(model.base) if hasattr(model, 'base') else slim_base(model)
                dump(model, out / 'best.pkl')
        write_json(out / 'best.json', record)
        records.append(record)
        write_json(work / 'validation_results.json', records)
        print(json.dumps(record), flush=True)
        gc.collect()

    # Контроль участвует в выборе: новая архитектура должна его обойти.
    assess('incumbent', base, ArrayEmbeddings(train_base, later_base))
    baseline = T.TemporalEncoder().fit(data)
    assess('baseline', baseline)
    del baseline
    for dim in [128, 256]:
        for epoch in [40, 120]:
            head = load(work / f'bag{dim}/epoch{epoch}.pkl')
            model = BagHybrid(base, bank, work / 'count_cache', buckets, head)
            for mode in ['neural', 'posterior']:
                model.mode, model.weight = mode, 1.0
                # Одна инференс-проходка на голову, затем меняется только вес слияния.
                train = model.embed(data, 'train', hosts)
                later = model.embed(data, 'validation', later_hosts)
                for weight in [0.5, 1.0]:
                    model.weight = weight
                    a, b = train.copy(), later.copy()
                    a.iloc[:, train_base.shape[1]:] *= weight
                    b.iloc[:, later_base.shape[1]:] *= weight
                    assess(f'bag{dim}_e{epoch}_{mode}_w{weight}', model, ArrayEmbeddings(a, b))
            del model, head
            gc.collect()
    # Контроль: память только повторяющихся признаков; весь отбор по train.
    stable = copy.copy(bank).set_stability(2)
    stable.class_projection = stable.class_projection[:, :128]
    model = FingerprintHybrid(base, stable, work / 'count_cache')
    for weight in [0.25, 0.5]:
        model.weight = weight
        assess(f'stable_memory_w{weight}', model)
    # freeze_selection читает все scores, но checkpoint нужен лишь у победителя.
    freeze_selection(candidates, work / 'final')


def run(work, stage, module, *args):
    state(work, stage)
    with (work / f'{stage}.log').open('a') as log:
        subprocess.run([sys.executable, '-m', module, *map(str, args)], cwd=ROOT,
                       env=dict(os.environ, PT_COMPACT='1'),
                       stdout=log, stderr=subprocess.STDOUT, check=True)


def train_parallel(work):
    jobs = []
    try:
        for dim in [128, 256]:
            out = work / f'bag{dim}'
            if (out / 'complete.json').exists():
                continue
            env = dict(os.environ, PT_COMPACT='1', PT_TORCH_THREADS='4', OMP_NUM_THREADS='4',
                       OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
            log = (work / f'bag{dim}.log').open('a')
            process = subprocess.Popen([sys.executable, '-m', 'experiments.multiday.bag',
                '--features', str(work / 'training.npz'), '--teacher', str(work / 'teacher.npy'),
                '--hosts', str(work / 'hosts.json'), '--dim', str(dim), '--out', str(out)],
                env=env, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            jobs.append((process, log, dim))
        state(work, 'parallel_training', workers={str(dim): p.pid for p, _, dim in jobs}, threads_per_worker=4)
        while any(p.poll() is None for p, _, _ in jobs):
            if psutil.virtual_memory().available < 2 * 1024**3:
                raise MemoryError('Доступная RAM меньше 2 ГБ: параллельное обучение остановлено')
            for p, _, dim in jobs:
                if p.poll() not in (None, 0):
                    raise RuntimeError(f'bag{dim}: exit {p.returncode}; см. bag{dim}.log')
            time.sleep(5)
        for p, _, dim in jobs:
            if p.returncode:
                raise RuntimeError(f'bag{dim}: exit {p.returncode}')
    finally:
        for p, log, _ in jobs:
            if p.poll() is None:
                os.killpg(p.pid, signal.SIGTERM)
                p.wait()
            log.close()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', required=True, type=Path)
    ap.add_argument('--stage', choices=['all', 'inputs', 'search'], default='all')
    ap.add_argument('--wait-workers', nargs='*', type=int, default=[])
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    if args.wait_workers:
        state(work, 'parallel_training', adopted_workers=args.wait_workers, threads_per_worker=4)
        def active_worker(pid):
            try:
                process = psutil.Process(pid)
                return process.is_running() and process.status() != psutil.STATUS_ZOMBIE and 'experiments.multiday.bag' in process.cmdline()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        while any(active_worker(pid) for pid in args.wait_workers):
            if psutil.virtual_memory().available < 2 * 1024**3:
                for pid in args.wait_workers:
                    if active_worker(pid):
                        os.killpg(pid, signal.SIGTERM)
                raise MemoryError('Меньше2 ГБ RAM во время продолжения обучения')
            time.sleep(5)
        if not all((work / f'bag{dim}/complete.json').exists() for dim in [128, 256]):
            raise RuntimeError('Обучение завершилось без complete.json; автоматического перезапуска нет')
    if args.stage == 'inputs':
        prepare_inputs(work)
        return
    if args.stage == 'search':
        score_candidates(work)
        return
    if (work / 'final/REPORT.md').exists():
        state(work, 'complete', **json.loads((work / 'final/summary.json').read_text()))
        return
    run(work, 'training_inputs', 'experiments.multiday.search', '--work', work, '--stage', 'inputs')
    train_parallel(work)
    # subprocess изолирует большие промежуточные массивы и освобождает RAM между стадиями.
    run(work, 'reference_validation', 'experiments.multiday.reference', '--base', BASE,
        '--target', work / 'day10', '--out', work / 'paired_validation',
        '--reference', work / 'reference_validation', '--window', 'validation')
    run(work, 'search', 'experiments.multiday.search', '--work', work, '--stage', 'search')
    if not (work / 'final/frozen_selection.json').exists():
        raise RuntimeError('Победитель не зафиксирован; день11 не открывается')
    if not (work / 'day11/manifest.json').exists():
        run(work, 'prepare_day11', 'experiments.multiday.prepare', '--raw',
            SOURCE / 'multiday/raw/netflow_day-11.bz2', '--base', BASE, '--out', work / 'day11', '--day', 11)
    run(work, 'reference_test', 'experiments.multiday.reference', '--base', BASE,
        '--target', work / 'day11', '--out', work / 'paired_test',
        '--reference', work / 'reference_test', '--window', 'test')
    candidates = sorted((work / 'candidates').iterdir())
    run(work, 'final_evaluation', 'experiments.multiday.finish', '--candidates', *candidates,
        '--data', work / 'paired_test', '--reference', work / 'reference_test/test', '--out', work / 'final',
        '--incumbent', INCUMBENT, '--test-label', 'day11, 12:30–13:30',
        '--protocol-text', 'Train: дни02–09, validation:10, test:11. Предыдущий энкодер обучен на02–07; новая голова на02–09. Когорта дня02.')
    state(work, 'complete', report=str(work / 'final/REPORT.md'), **json.loads((work / 'final/summary.json').read_text()))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        if '--work' in sys.argv:
            state(Path(sys.argv[sys.argv.index('--work')+1]), 'failed', error=f'{type(error).__name__}: {error}')
        raise
