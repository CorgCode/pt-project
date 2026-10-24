"""Запустить зафиксированный эксперимент v5/v6 в отдельной рабочей папке."""
import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

PACKAGE = Path(__file__).resolve().parent
ROOT = PACKAGE.parents[2]


def materialize(work):
    work = Path(work)
    if work.exists() and any(work.iterdir()):
        raise ValueError(f'{work}: выберите пустую папку; существующие модели не перезаписываются')
    work.mkdir(parents=True, exist_ok=True)
    for source in (PACKAGE / 'templates').rglob('*'):
        if source.is_file() and '__pycache__' not in source.parts:
            destination = work / source.relative_to(PACKAGE / 'templates')
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)


def check_dataset(dataset):
    manifest = json.loads((dataset / 'processed/manifest.json').read_text())
    expected = {'train': [126000, 129600], 'validation': [129600, 131400], 'test': [131400, 133200]}
    if manifest['config']['windows'] != expected or manifest['config']['cohort_policy'] != 'train_only':
        raise ValueError('Этот эксперимент зафиксирован для исходного протокола day02; окна или когорта отличаются')
    if manifest['config']['min_events_per_window'] != 300:
        raise ValueError('Ожидается порог 300 событий')
    for window in expected:
        if not (dataset / 'processed/flows' / f'{window}.csv').is_file():
            raise FileNotFoundError(f'Нет потоков {window}')
    for window in ('validation', 'test'):
        if not (dataset / 'reference' / window / 'manifest.json').is_file():
            raise FileNotFoundError(f'Нет эталона {window}')


def run_stage(script, work, env):
    print(f'\nЗапуск: {script}', flush=True)
    subprocess.run([sys.executable, '-u', str(script)], cwd=ROOT, env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True, help='Папка с processed/ и reference/ исходного day02')
    parser.add_argument('--workdir', type=Path, default=ROOT / 'runs/fast_temporal_reproduction')
    parser.add_argument('--with-holdout', action='store_true', help='Проверить новое окно 15:00–15:30')
    parser.add_argument('--raw', type=Path, help='netflow_day-02.bz2; нужен для --with-holdout')
    args = parser.parse_args()
    dataset, work = args.dataset.resolve(), args.workdir.resolve()
    check_dataset(dataset)
    if args.with_holdout and (args.raw is None or not args.raw.is_file()):
        parser.error('--with-holdout требует доступный --raw netflow_day-02.bz2')
    materialize(work)
    env = dict(os.environ, PT_REPO_ROOT=str(ROOT), PT_RUN_ROOT=str(work), PT_LANL_DIR=str(dataset),
               PYTHONUNBUFFERED='1', OPENBLAS_NUM_THREADS='2', OMP_NUM_THREADS='2', VECLIB_MAXIMUM_THREADS='2',
               MPLBACKEND='Agg', MPLCONFIGDIR=str(work / 'mpl-cache'))
    if args.raw:
        env['PT_LANL_RAW'] = str(args.raw.resolve())
    (work / 'execution.json').write_text(json.dumps({'dataset': str(dataset), 'workdir': str(work),
        'python': sys.executable, 'with_holdout': args.with_holdout}, indent=2)+'\n')
    run_stage(PACKAGE / 'prepare_features.py', work, env)
    run_stage(work / 'hybrid_v5/train.py', work, env)
    # v6 использует фиксированные блоки победителя исходного эксперимента.
    # При изменении победителя не смешиваем его test с другой базовой моделью.
    selected = json.loads((work / 'hybrid_v5/selected.json').read_text())
    expected_recipe = {'behavioral': 2.0, 'peers': 1.0, 'services': 1.0, 'diffusion': 0.5, 'neural': 1.0}
    if selected['name'] != 'v5_candidate_07' or selected['checkpoint'] != 20 or selected['recipe'] != expected_recipe:
        raise RuntimeError('v5 выбрала другой рецепт. Результаты сохранены; фиксированный адаптер v6 нельзя применять к другим блокам.')
    run_stage(work / 'hybrid_v6_fast/train.py', work, env)
    if args.with_holdout:
        run_stage(work / 'holdout_day02_1500/prepare.py', work, env)
        run_stage(work / 'holdout_day02_1500/evaluate_frozen.py', work, env)
    print(f'\nГотово: {work / "hybrid_v6_fast/comparison.md"}', flush=True)


if __name__ == '__main__':
    main()
