"""Оценка сохранённого temporal_stable_hybrid по окнам общей таблицы.

python experiments/temporal_stable_score.py --data data/scoreboard_exact/processed \
    --model runs/temporal_stable_hybrid_v2_validation/model.joblib --output runs/temporal_stable_hybrid_v2_exact
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scoreboard'))
from temporal_stable_hybrid import TemporalStableHybrid  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402
from evaluation import scoreboard, submit  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--data', type=Path, required=True)
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--bootstrap', type=int, default=200)
    args = ap.parse_args()
    official = ROOT / 'scoreboard/data/processed'
    data = ProcessedData(args.data)
    expected = json.loads((official / 'manifest.json').read_text())
    digest = lambda p: hashlib.sha256(p.read_bytes()).digest()
    if (data.manifest['windows'] != expected['windows'] or
            digest(args.data / 'cohort.csv') != digest(official / 'cohort.csv') or
            data.manifest['inputs'] != expected['inputs']):
        raise ValueError('манифест или когорта не соответствуют общей подготовке')
    for window in data.windows:
        if not (args.data / 'flows' / f'{window}.csv').is_file():
            raise FileNotFoundError(f'нет потоков окна {window}')
    model = TemporalStableHybrid.load(args.model)
    if model.hosts != data.cohort:
        raise ValueError('когорта обученной модели отличается от общей')
    args.output.mkdir(parents=True, exist_ok=False)
    params = args.model.parent / 'params.json'
    if params.exists():
        shutil.copyfile(params, args.output / 'params.json')
    # Кэш чтения сохраняется в runs, подготовленные данные не изменяются.
    local = args.output / 'processed'
    (local / 'flows').mkdir(parents=True)
    for name in ('cohort.csv', 'manifest.json'):
        shutil.copyfile(args.data / name, local / name)
    for window in data.windows:
        (local / 'flows' / f'{window}.csv').symlink_to((args.data / 'flows' / f'{window}.csv').resolve())
    data = ProcessedData(local)
    jobs = [('train', data.cohort, 'embeddings.csv')]
    jobs += [(w, data.evaluable(w), submit.window_file(w)) for w in data.test_windows]
    for window, hosts, filename in jobs:
        print(f'Векторы {window}...', flush=True)
        model.embed(data, window, hosts).to_csv(args.output / filename, float_format='%.8g')
        if window == 'train':
            previous = args.model.parent / 'embeddings.csv'
            if previous.exists() and digest(previous) != digest(args.output / filename):
                raise ValueError('train-векторы после загрузки модели изменились')
    (args.output / 'SHARED_COORDINATES').touch()
    submit.N_BOOT = args.bootstrap
    results = {}
    for w in data.test_windows:
        print(f'Оценка {w}...', flush=True)
        results[w] = submit._score_window(args.output, w)
    result = submit.combine_windows(results)
    cols = {s['metric']: s['title'] for s in result['summary']}
    score = scoreboard.overall_score([dict(result, method='temporal_stable_hybrid')], cols).iloc[0]
    per_window = {w: scoreboard._one_score(r, cols, window=w) for w, r in results.items()}
    if not all(scoreboard.complete(r, cols) for r in results.values()):
        raise ValueError('неполная оценка')
    out = {'score': score, 'per_window': per_window, 'result': result}
    (args.output / 'result.json').write_text(json.dumps(out, ensure_ascii=False, indent=2) + '\n')
    print(f'Score: {score:.3f}; по окнам: {per_window}', flush=True)


if __name__ == '__main__':
    main()
