"""Поведенческий эталон из ОТЛОЖЕННОГО окна для оценки соседства.

Проверяет, сохраняется ли поведенческое соседство во времени, а не правильность ролей.
На каких данных учился эмбеддинг, должен проверить автор эксперимента.

 python -m evaluation.temporal_reference \
   --train-flows data/train.csv --test-flows data/test.csv \
   --embeddings submissions/method/embeddings.csv \
   --output-dir reference/heldout
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.validate import validate_submission
from evaluation.knn_consistency import build_reference_features, evaluate_embedding
from evaluation.frozen_neighbors import save_frozen_neighbors

REQUIRED = {"src", "dst", "dst_port", "proto", "bytes", "packets", "ts"}


def load_flows(path: Path) -> pd.DataFrame:
    """Загрузить нормализованные потоки; время — Unix-секунды или ISO-8601."""
    f = pd.read_csv(
        path, dtype={"src": "string", "dst": "string",
                     "dst_port": "string", "proto": "string"}
    )
    missing = REQUIRED - set(f.columns)
    if missing:
        raise ValueError(f"{path}: нет колонок: {sorted(missing)}")
    if f.empty:
        raise ValueError(f"{path}: пустой файл потоков")

    for col in ("src", "dst", "dst_port", "proto"):
        if f[col].isna().any() or f[col].str.strip().eq("").any():
            raise ValueError(f"{path}: пустые значения в {col}")
        f[col] = f[col].str.strip()

    for col in ("bytes", "packets"):
        f[col] = pd.to_numeric(f[col], errors="coerce")
        if (not np.isfinite(f[col].to_numpy(dtype=float)).all()
                or (f[col] < 0).any()):
            raise ValueError(f"{path}: {col} должен быть конечным и неотрицательным")

    raw = f["ts"]
    numeric = pd.api.types.is_numeric_dtype(raw)
    if not numeric:
        numeric = raw.astype(str).str.fullmatch(r"[+-]?\d+(\.\d+)?").all()
    if numeric:
        ts = pd.to_datetime(pd.to_numeric(raw, errors="coerce"),
                            unit="s", errors="coerce", utc=True)
    else:
        ts = pd.to_datetime(raw, errors="coerce", utc=True)
    if ts.isna().any():
        raise ValueError(f"{path}: некорректное время")
    # build_reference_features ждёт обычный datetime64
    f["ts"] = ts.dt.tz_convert("UTC").dt.tz_localize(None)
    return f


def _hosts_from_embeddings(path: Path) -> list[str]:
    validate_submission(str(path))
    hosts = pd.read_csv(path, dtype={"host_id": "string"})["host_id"]
    if hosts.isna().any() or hosts.str.strip().eq("").any():
        raise ValueError("в эмбеддинге пустые host_id")
    return hosts.astype(str).tolist()


def generate_reference(
    train: pd.DataFrame,
    test: pd.DataFrame,
    hosts: list[str],
    output_dir: Path,
    ks: tuple[int, ...] = (5, 10, 20),
    split: str = "test",
) -> dict:
    """Записать features.csv и manifest.json для строгого отложенного окна.

    Хосты должны быть активны в ОБОИХ периодах. Нулевые профили для отсутствующих хостов не подставляем:
    это дало бы случайных соседей по косинусу.
    """
    if not hosts or len(set(hosts)) != len(hosts):
        raise ValueError("список хостов должен быть непустым и без повторов")
    if not ks or any(k < 1 or k >= len(hosts) for k in ks):
        raise ValueError("каждое k должно быть от 1 до числа хостов - 1")

    train_end, test_start = train["ts"].max(), test["ts"].min()
    if not train_end < test_start:
        raise ValueError(
            f"окна пересекаются или соприкасаются: train_end={train_end}, test_start={test_start}. "
            "Test-события должны быть строго позже."
        )

    for name, flows in (("train", train), ("test", test)):
        active = set(flows["src"]) | set(flows["dst"])
        absent = sorted(set(hosts) - active)
        if absent:
            raise ValueError(
                f"в окне {name} неактивны {len(absent)} хостов эмбеддинга "
                f"(например {absent[:5]}). Выберите общую когорту активных хостов "
                "заранее, одинаково для ВСЕХ методов."
            )

    train_ref = build_reference_features(train, hosts=hosts)
    test_ref = build_reference_features(test, hosts=hosts)
    # у каждого периода своя похожесть внутри окна; размерности признаков могут различаться,
    # потому что сравниваются СПИСКИ СОСЕДЕЙ, а не векторы
    summary = evaluate_embedding(train_ref, test_ref, ks=ks)
    stability = {
        f"{metric}@{k}": float(row["mean"])
        for (k, metric), row in summary.iterrows()
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    reference = test_ref.copy()
    reference.columns = [f"ref_{j}" for j in range(reference.shape[1])]
    reference.index.name = "host_id"
    reference.to_csv(output_dir / "features.csv")
    save_frozen_neighbors(
        reference, output_dir / "neighbors.csv",
        reference_window=test_start.isoformat() + "Z"
        + "/" + test["ts"].max().isoformat() + "Z",
        k_max=max(ks), split=split,
    )

    manifest = {
        "kind": "heldout_temporal_behavior_proxy",
        "ground_truth": False,
        "provenance_verified": False,
        "provenance_warning": (
            "Nonoverlap of supplied flow files is verified, but whether the "
            "submitted embeddings were trained exclusively on the train window "
            "cannot be established automatically."
        ),
        "purpose": "cross-window behavioral neighbor preservation, not host role labels",
        "train_events": int(len(train)),
        "test_events": int(len(test)),
        "train_end_utc": train_end.isoformat() + "Z",
        "test_start_utc": test_start.isoformat() + "Z",
        "split": split,
        "n_hosts": len(hosts),
        "reference_dimensions": reference.shape[1],
        "ks": list(ks),
        "chance_consistency_at_k": {
            str(k): k / (len(hosts) - 1) for k in ks
        },
        "reference_neighbor_stability": stability,
        "note": (
            "High consistency alone does not prove functional roles are learned. "
            "Report feature choice, baseline results and window drift."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-flows", type=Path, required=True)
    parser.add_argument("--test-flows", type=Path, required=True)
    parser.add_argument("--embeddings", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ks", nargs="+", type=int, default=[5, 10, 20])
    args = parser.parse_args()
    hosts = _hosts_from_embeddings(args.embeddings)
    manifest = generate_reference(
        load_flows(args.train_flows), load_flows(args.test_flows),
        hosts, args.output_dir, tuple(args.ks),
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
