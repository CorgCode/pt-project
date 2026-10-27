"""Подбор параметров GNN V1.1 через Optuna. Цель — ОБЩИЙ БАЛЛ проекта на validation (test не трогается).

    pip install optuna
    python scoreboard/tune_gnn.py --trials 5            # проверка, что всё работает (около 10-15 минут)
    python scoreboard/tune_gnn.py --trials 40           # полный подбор; можно прервать (Ctrl+C) и продолжить той же командой
    python scoreboard/tune_gnn.py --trials 40 --timeout 3600   # остановиться через час

Результат: runs/best_params.json. Потом:
    python scoreboard/gnn_v1.py --params-json runs/best_params.json --seed 0 --name gnn_v1_s0 --score

Исключённых метрик нет. Если понадобится исключить — только здесь, в EXCLUDED_METRICS (сейчас пусто, см. objective).
Размерность dim=32 зафиксирована (как у остальных методов проекта, для честного сравнения), mu зафиксирован в 1.0:
Adam почти не чувствителен к общему масштабу потери, значит значение имеет только отношение lam/mu, а его задаёт lam.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import optuna

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnn_v1 as G                      # noqa: E402  (добавляет корень репозитория в sys.path)
from evaluation import submit           # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_METRICS: list[str] = []        # сейчас ничего не исключаем
T_START = time.time()
NARROW = False                          # узкий режим: подбираем только concat_k, beta, lam (для малого числа проб)


def objective(trial: optuna.Trial) -> float:
    a = G.get_parser().parse_args([])
    a.out, a.name = str(ROOT / "runs" / "tune"), f"trial_{trial.number}"
    a.only_windows = ["validation"]     # test-окна в подборе не строим: быстрее и честнее
    a.seed = 0
    if NARROW:                          # остальные параметры — значения по умолчанию из gnn_v1.py
        a.concat_k = trial.suggest_categorical("concat_k", [16, 32])
        a.beta = trial.suggest_float("beta", 0.5, 4.0)
        a.lam = trial.suggest_float("lam", 0.2, 5.0, log=True)
    else:
        a.hidden = trial.suggest_categorical("hidden", [64, 128])
        a.lr = trial.suggest_float("lr", 1e-3, 2e-2, log=True)
        a.epochs = trial.suggest_int("epochs", 80, 250, step=10)
        a.dropout = trial.suggest_float("dropout", 0.0, 0.3)
        a.tau = trial.suggest_float("tau", 0.05, 0.5, log=True)
        a.hold = trial.suggest_float("hold", 0.05, 0.3)
        a.kneg = trial.suggest_int("kneg", 2, 10)
        a.hub_pow = trial.suggest_float("hub_pow", 0.0, 1.0)
        a.lam = trial.suggest_float("lam", 0.05, 20.0, log=True)
        a.concat_k = trial.suggest_categorical("concat_k", [0, 8, 16, 24, 32])   # 0 — чистый V1
        if a.concat_k > 0:
            a.beta = trial.suggest_float("beta", 0.0, 2.0)

    out = Path(a.out) / a.name
    try:
        out = G.build_submission(a)
        score, tasks = submit.validation_score(out / "embeddings.csv", out / "validation.csv")
    finally:
        shutil.rmtree(out, ignore_errors=True)                               # экономим диск
    trial.set_user_attr("tasks", tasks)
    tasks_txt = ", ".join(f"{k} {v:.0f}" for k, v in tasks.items())
    print(f"[{time.time() - T_START:.0f} с] проба {trial.number}: балл на validation {score:.2f} | {tasks_txt}", flush=True)
    return score


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--timeout", type=int, default=None, help="секунд; остановка по времени")
    ap.add_argument("--study", default=None, help="имя подбора; при смене пространства поиска задайте новое имя")
    ap.add_argument("--narrow", action="store_true",
                    help="узкий режим для 5-10 проб: только concat_k, beta, lam; первой идёт проба «16, 1.0, 1.0» (то, что уже запускали)")
    a = ap.parse_args()
    global NARROW
    NARROW = a.narrow
    a.study = a.study or ("gnn_v1_narrow" if a.narrow else "gnn_v1_a")
    G.QUIET = True                                                           # без построчного хода обучения
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    (ROOT / "runs").mkdir(exist_ok=True)
    study = optuna.create_study(direction="maximize", study_name=a.study,
                                storage=f"sqlite:///{ROOT / 'runs' / 'optuna.db'}", load_if_exists=True,
                                sampler=optuna.samplers.TPESampler(seed=0, n_startup_trials=4 if a.narrow else 10))
    if a.narrow and not study.trials:
        study.enqueue_trial({"concat_k": 16, "beta": 1.0, "lam": 1.0})      # контрольная проба: сравнимый с остальными балл на validation
    print(f"подбор «{a.study}»: уже проб {len(study.trials)}; запускаю до {a.trials} новых", flush=True)
    study.optimize(objective, n_trials=a.trials, timeout=a.timeout)

    bar = "=" * 14
    print(f"\n{bar} КРАТКИЙ ИТОГ (скопируйте всё отсюда и пришлите мне) {bar}")
    print(f"Проб всего: {len(study.trials)}. ЛУЧШИЙ балл на validation: {study.best_value:.2f} (проба {study.best_trial.number})")
    print("Баллы по задачам в лучшей пробе:", study.best_trial.user_attrs.get("tasks"))
    print(json.dumps(study.best_params, indent=1))
    top = sorted((t for t in study.trials if t.value is not None), key=lambda t: -t.value)[:5]
    print("Пять лучших проб (балл, concat_k, beta):",
          [(round(t.value, 1), t.params.get("concat_k"), round(t.params.get("beta", 0.0), 2)) for t in top])
    print("=" * 83)
    (ROOT / "runs" / "best_params.json").write_text(json.dumps(study.best_params, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
