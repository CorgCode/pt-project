"""Ночной перебор гипотез на одной машине: подбор только по validation, test — один раз в конце.

    PT_DATA=/путь/processed python -m experiments.overnight.search --hours 8
    PT_DATA=... python -m experiments.overnight.search --final      # только итог по уже сделанным пробам

Каждая проба — строка в runs/overnight/trials.jsonl; повторный запуск продолжает с места остановки.
Между пробами — пауза и проверки: свободная память, место на диске, перегрев (macOS: pmset, Linux: thermal_zone).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from evaluation import scoreboard, submit  # noqa: E402
from experiments.overnight.model import FanEncoder  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402

OUT = ROOT / "runs" / "overnight"
LOG = OUT / "trials.jsonl"
BASE = {"behavioral": 2.0, "peers": 1.0, "services": 1.0, "diffusion": 0.5, "neural": 1.0}   # temporal_encoder (№1)
BEST_SO_FAR = dict(BASE, behavioral=4.0, neural=2.0, fan=0.5)                               # temporal_encoder_fan

# что обучается (каждое сочетание — одно обучение), от дешёвого к дорогому
TRAIN = [{"splits": s, "epochs": e, "seeds": sd} for s, e, sd in
         [(2, 20, (17,)), (2, 5, (17,)), (2, 10, (17,)), (4, 10, (17,)), (4, 20, (17,)), (3, 20, (17,)),
          (2, 40, (17,)), (2, 10, (17, 1, 2)), (2, 20, (17, 1, 2)), (4, 20, (17, 1, 2))]]
# 5 и 10 эпох — после проверки: 2 эпохи дали на validation больше, чем 20 (59.4 против 58.9), но на test меньше
# веса частей и длина PCA — без переобучения
SPACE = {"behavioral": (3.0, 3.5, 4.0, 4.5, 5.0), "neural": (1.5, 2.0, 2.5, 3.0), "fan": (0.0, 0.25, 0.5, 0.75),
         "diffusion": (0.0, 0.25, 0.5), "context": (0.0, 0.25), "peers": (0.75, 1.0), "pca_dim": (256, 384, 512)}
RECIPES_PER_TRAIN = 24

MIN_FREE_GB, MIN_DISK_GB, PAUSE_S = 2.0, 5.0, 20


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_memory_gb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 2**30
    except ImportError:
        pass
    if Path("/proc/meminfo").exists():
        m = dict(re.findall(r"(\w+):\s+(\d+)", Path("/proc/meminfo").read_text()))
        return int(m.get("MemAvailable", 0)) / 2**20
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
        page = int(re.search(r"page size of (\d+)", out).group(1))
        pages = sum(int(re.search(rf"{k}:\s+(\d+)", out).group(1)) for k in ("Pages free", "Pages inactive", "Pages speculative"))
        return pages * page / 2**30
    except Exception:
        return 99.0                                          # не смогли узнать — не мешаем


def overheated() -> bool:
    """macOS снижает частоту сама; если лимит < 80% — даём остыть. Linux: зона > 85 °C."""
    try:
        out = subprocess.run(["pmset", "-g", "therm"], capture_output=True, text=True, timeout=10).stdout
        m = re.search(r"CPU_Speed_Limit\s*=\s*(\d+)", out)
        if m:
            return int(m.group(1)) < 80
    except Exception:
        pass
    temps = [int(p.read_text()) / 1000 for p in Path("/sys/class/thermal").glob("thermal_zone*/temp") if p.exists()]
    return bool(temps) and max(temps) > 85


def safe_to_continue() -> bool:
    for _ in range(30):                                      # до 15 минут ожидания, потом стоп
        disk = shutil.disk_usage(OUT).free / 2**30
        if disk < MIN_DISK_GB:
            log(f"мало места на диске: {disk:.1f} ГБ — останавливаюсь")
            return False
        mem, hot = free_memory_gb(), overheated()
        if mem >= MIN_FREE_GB and not hot:
            return True
        log(f"пауза 30 с: свободно памяти {mem:.1f} ГБ, перегрев {hot}")
        time.sleep(30)
    log("условия не улучшились за 15 минут — останавливаюсь")
    return False


def done_keys() -> set[str]:
    if not LOG.exists():
        return set()
    return {json.loads(l)["key"] for l in LOG.read_text().splitlines() if l.strip()}


def key(train: dict, recipe: dict) -> str:
    return json.dumps({"train": train, "recipe": recipe}, sort_keys=True)


def sample_recipes(seed: int) -> list[dict]:
    rng = random.Random(seed)
    grid = [dict(zip(SPACE, v)) for v in itertools.product(*SPACE.values())]
    out = [dict(BEST_SO_FAR, pca_dim=384), dict(BASE, pca_dim=384)]
    out += rng.sample(grid, RECIPES_PER_TRAIN - len(out))
    return [{**BASE, **r} for r in out]


def evaluate(model: FanEncoder, data: ProcessedData, recipe: dict) -> tuple[float, dict]:
    r = dict(recipe)
    model.pca_dim = r.pop("pca_dim")
    model.recipe = {k: v for k, v in r.items()}
    model.fit_pca(data)
    d = OUT / "candidate"
    d.mkdir(parents=True, exist_ok=True)
    for window, hosts, fname in (("train", data.cohort, "embeddings.csv"),
                                 ("validation", data.evaluable("validation"), "validation.csv")):
        model.embed(data, window, hosts).to_csv(d / fname, float_format="%.5g")
    return submit.validation_score(d / "embeddings.csv", d / "validation.csv")


def search(data: ProcessedData, deadline: float) -> None:
    done = done_keys()
    log(f"уже сделано проб: {len(done)}")
    for i, train in enumerate(TRAIN):
        recipes = [r for r in sample_recipes(i) if key(train, r) not in done]
        if not recipes:
            continue
        if time.time() > deadline or not safe_to_continue():
            return
        t = time.time()
        model = FanEncoder(recipe=BASE, splits=train["splits"], epochs=train["epochs"], encoder="mlp",
                           seeds=tuple(train["seeds"])).fit(data)
        log(f"обучение {train}: {time.time() - t:.0f} с")
        for recipe in recipes:
            if time.time() > deadline or not safe_to_continue():
                return
            score, tasks = evaluate(model, data, recipe)
            row = {"key": key(train, recipe), "train": train, "recipe": recipe, "score": round(score, 3),
                   **{k: round(v, 2) for k, v in tasks.items()}, "at": time.strftime("%F %T")}
            with LOG.open("a") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            log(f"{score:6.2f}  {train}  {recipe}")
            time.sleep(PAUSE_S)                              # процессору — передышка между пробами
        del model


CARD = """# overnight_best

## Кратко

Лучшая проба ночного перебора: `temporal_encoder` + «веер по сервису», веса частей и обучение подобраны на validation.

## Архитектура

Части (каждая нормирована и умножена на вес): {parts}. Итог сжимается PCA до {pca}.
Веер — для каждого сервиса число разных собеседников как клиент и как сервер, TF-IDF → SVD 64.
Нейросеть «тот же хост в разное время»: {seeds}.

## Данные

Поля src, dst, dst_port, proto, счётчики потоков. Всё обучается только на train-окне; другие окна —
через те же преобразования (общие координаты).

## Обучение и подбор

Train режется на {splits} куска(ов), {epochs} эпох. Перебрано {n} проб (`experiments/overnight/search.py`):
{n_train} вариантов обучения × до {per} наборов весов. Выбор — по баллу на validation, test не смотрели.

## Ограничения

Много проб на одном окне validation: часть прироста на validation может не повториться на test.
Победитель выбран без учёта test, поэтому итог на test — честная проверка.
"""


def final(data: ProcessedData) -> None:
    rows = [json.loads(l) for l in LOG.read_text().splitlines() if l.strip()]
    rows.sort(key=lambda r: -r["score"])
    best = rows[0]
    log(f"лучшая на validation: {best['score']} {best['train']} {best['recipe']}")
    train, recipe = best["train"], dict(best["recipe"])
    model = FanEncoder(recipe=BASE, splits=train["splits"], epochs=train["epochs"], encoder="mlp",
                       seeds=tuple(train["seeds"])).fit(data)
    model.pca_dim = recipe.pop("pca_dim")
    model.recipe = recipe
    model.fit_pca(data)
    sub = ROOT / "scoreboard" / "submissions" / "overnight_best"
    sub.mkdir(parents=True, exist_ok=True)
    jobs = [("train", data.cohort, "embeddings.csv")] + [(w, data.evaluable(w), f"{w}.csv") for w in submit.test_windows()]
    for w, hosts, fname in jobs:
        model.embed(data, w, hosts).to_csv(sub / fname, float_format="%.5g")
    (sub / "SHARED_COORDINATES").touch()
    (sub / "params.json").write_text(json.dumps(best, ensure_ascii=False, indent=1))
    parts = ", ".join(f"{k} ×{v:g}" for k, v in recipe.items() if v)
    (sub / "README.md").write_text(CARD.format(parts=parts, pca=model.pca_dim, splits=train["splits"],
                                               epochs=train["epochs"], n=len(rows), n_train=len(TRAIN),
                                               per=RECIPES_PER_TRAIN,
                                               seeds=f"ансамбль из {len(train['seeds'])} зёрен" if len(train["seeds"]) > 1
                                               else "одно зерно"), encoding="utf-8")
    submit.score(sub)
    results = scoreboard.load_results(Path(submit.RESULTS))
    _, _, order = scoreboard.ranking(results)
    lines = ["# Ночной перебор: итог", "", f"Проб: {len(rows)}. Лучшая на validation: **{best['score']}**.", "",
             "## Таблица на test (три дня)", "", "| # | method | score |", "|---|---|---|"]
    lines += [f"| {i} | `{m}` | {s:.2f} |" for i, (m, s) in enumerate(order.head(8).items(), 1)]
    lines += ["", "## Десять лучших проб на validation", "", "| validation | обучение | веса |", "|---|---|---|"]
    lines += [f"| {r['score']} | {r['train']} | {r['recipe']} |" for r in rows[:10]]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    log(f"итог: {OUT / 'REPORT.md'}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hours", type=float, default=8.0, help="сколько часов перебирать, потом итог")
    ap.add_argument("--final", action="store_true", help="только итог по сделанным пробам")
    ap.add_argument("--smoke", action="store_true", help="проверка за ~10 минут: 2 эпохи, 2 пробы, отдельная папка")
    a = ap.parse_args()
    global OUT, LOG, TRAIN, RECIPES_PER_TRAIN, PAUSE_S
    if a.smoke:
        OUT, TRAIN, RECIPES_PER_TRAIN, PAUSE_S = OUT.parent / "overnight_smoke", [{"splits": 2, "epochs": 2, "seeds": (17,)}], 2, 1
        LOG = OUT / "trials.jsonl"
    OUT.mkdir(parents=True, exist_ok=True)
    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    if not a.final:
        search(data, time.time() + a.hours * 3600)
    if LOG.exists():
        final(data)


if __name__ == "__main__":
    main()
