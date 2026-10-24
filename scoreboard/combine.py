"""Гибрид: склеить эмбеддинги нескольких методов из таблицы в один.

    python scoreboard/combine.py behavioral_pca node2vec --name hybrid_behavioral_node2vec
    python scoreboard/combine.py behavioral node2vec --name hybrid --tune-weights   # вес частей подобрать на validation

Каждая часть нормируется (среднее 0, общая норма 1 на хост), чтобы ни одна не перевешивала из-за масштаба,
затем векторы склеиваются. Берутся train (embeddings.csv) и test-окна (test.csv, test_<метка>.csv), если они есть у всех частей.
Результат — обычная отправка в scoreboard/submissions/<name>/, она сразу оценивается.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from evaluation.io import load_embeddings  # noqa: E402


def _block(x: pd.DataFrame) -> pd.DataFrame:
    x = x - x.mean()
    norm = np.sqrt((x.to_numpy() ** 2).sum(axis=1).mean())
    return x / (norm if norm > 0 else 1.0)


def combine(parts: list[str], name: str, out_root: Path | None = None, weights: list[float] | None = None) -> Path:
    subs = Path(submit.SUBMISSIONS)                       # части берём из таблицы
    out = Path(out_root or subs) / name
    out.mkdir(parents=True, exist_ok=True)
    tests = sorted({f.name for p in parts for f in (subs / p).glob("test*.csv")})   # test.csv, test_day03.csv, ...
    for fname in ["embeddings.csv"] + tests + ["validation.csv"]:
        files = [subs / p / fname for p in parts]
        if not all(f.exists() for f in files):
            continue
        w = weights or [1.0] * len(files)
        blocks = [_block(load_embeddings(f)) * wi for f, wi in zip(files, w)]
        hosts = sorted(set.intersection(*(set(b.index) for b in blocks)))
        x = np.hstack([b.loc[hosts].to_numpy() for b in blocks])
        pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(out / fname, float_format="%.7g")
    (out / "SHARED_COORDINATES").unlink(missing_ok=True)       # части могут быть в разных координатах
    if not (out / "README.md").exists():
        (out / "README.md").write_text(_card(parts, name, weights), encoding="utf-8")
    return out


def _card(parts: list[str], name: str, weights: list[float] | None) -> str:
    """Карточка гибрида по рецепту; при желании её можно дописать руками."""
    w = weights or [1.0] * len(parts)
    tuning = ("Вес первой части подобран по баллу на validation (`scoreboard/combine.py --tune-weights`), "
              "test при этом не используется." if weights else "Веса равные, не подбирались.")
    recipe = " + ".join(f"`{p}` ×{wi:g}" for p, wi in zip(parts, w))
    links = ", ".join(f"[`{p}`](../{p}/README.md)" for p in parts)
    return f"""# {name}

## Кратко

Гибрид: {recipe}. Части ловят разное, вместе — лучше, чем по отдельности.

## Архитектура

Векторы частей ({links}) склеиваются (конкатенация). Перед склейкой каждая часть нормируется:
среднее 0, средняя длина вектора 1 — чтобы ни одна не перевешивала из-за масштаба. Затем умножается на свой вес.

## Данные

Те же, что у частей: всё строится только по train-окну. Вектор test-окна — склейка test-векторов частей.

## Обучение и подбор

Своего обучения нет. {tuning}

## Ограничения

Наследует ограничения частей. Размерность — сумма размерностей частей, поэтому сравнение с одиночными методами
не при равной размерности.
"""


def tune_weights(parts: list[str], grid=(1.0, 2.0, 3.0, 4.0, 6.0, 10.0)) -> tuple[list[float], list[dict]]:
    """Вес первой части (остальные = 1) по общему баллу на validation. Test не используется.
    Узнаваемость тоже учитывается, если у всех частей есть validation.csv (эмбеддинг validation-окна)."""
    import tempfile

    rows = []
    with tempfile.TemporaryDirectory() as tmp:
        for w0 in grid:
            weights = [w0] + [1.0] * (len(parts) - 1)
            d = combine(parts, "w", Path(tmp), weights)
            score, tasks = submit.validation_score(d / "embeddings.csv", d / "validation.csv")
            rows.append({"weight": w0, "score": score, "tasks": tasks})
            print(f"  вес {parts[0]} = {w0}: балл на validation {score:.1f}  "
                  + ", ".join(f"{k} {v:.0f}" for k, v in tasks.items()))
    best = max(rows, key=lambda x: x["score"])
    return [best["weight"]] + [1.0] * (len(parts) - 1), rows


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("parts", nargs="+", help="имена методов из scoreboard/submissions")
    ap.add_argument("--name", required=True)
    ap.add_argument("--tune-weights", action="store_true", help="подобрать вес первой части на validation")
    a = ap.parse_args()
    weights = None
    if a.tune_weights:
        weights, _ = tune_weights(a.parts)
        print(f"выбран вес {a.parts[0]}: {weights[0]}")
    out = combine(a.parts, a.name, weights=weights)
    r = submit.score(out)
    submit.render()
    print(f"{a.name}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
