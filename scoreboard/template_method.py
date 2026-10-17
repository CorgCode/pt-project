"""Шаблон своего метода. Скопируйте файл, поменяйте NAME и две функции — fit и embed.

    cp scoreboard/template_method.py scoreboard/my_method.py
    PT_DATA=/путь/к/lanl/processed python scoreboard/my_method.py   # обучить, сохранить, увидеть место

Потоки LANL (≈ 1 ГБ) в git не хранятся: их даёт experiments/lanl/README.md (скачать и подготовить)
или копия с общего диска. PT_DATA — путь к папке processed с подпапкой flows/.

Скрипт сам:
  1. берёт общие данные и отдаёт вашему методу ТОЛЬКО train-окно;
  2. сохраняет эмбеддинги в scoreboard/submissions/<NAME>/ (train и test);
  3. считает метрики и печатает строку таблицы.
Потом commit и push — CI добавит метод в scoreboard/README.md.

Пример ниже — простой метод: хост описывается тем, на какие порты он ходит, затем SVD.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402

NAME = "template_ports_svd"     # имя метода: латиница, без пробелов
DIM = 16                        # размерность эмбеддинга
SHARED_COORDINATES = True       # True, если fit один раз, а embed применяет ту же модель к любому окну


# ------------------------------------------------------------------ ваш метод: меняйте только это

def fit(train_flows: pd.DataFrame, hosts: list[str]):
    """Обучить модель. train_flows — потоки train-окна: src, dst, dst_port, proto, bytes, packets, ts."""
    counts = _port_counts(train_flows, hosts)
    top_ports = counts.sum().sort_values(ascending=False).index[:200]
    x = np.log1p(counts[top_ports].to_numpy(float))
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return {"ports": top_ports, "mean": mean, "components": vt[:DIM]}


def embed(model, flows: pd.DataFrame, hosts: list[str]) -> np.ndarray:
    """Эмбеддинг хостов hosts по потокам flows: массив (len(hosts), DIM), строки в порядке hosts."""
    counts = _port_counts(flows, hosts).reindex(columns=model["ports"], fill_value=0)
    x = np.log1p(counts.to_numpy(float)) - model["mean"]
    return x @ model["components"].T


def _port_counts(flows: pd.DataFrame, hosts: list[str]) -> pd.DataFrame:
    f = flows[flows["src"].astype(str).isin(hosts)]
    return pd.crosstab(f["src"].astype(str), f["dst_port"].astype(str)).reindex(hosts, fill_value=0)


# ------------------------------------------------------------------ дальше ничего менять не нужно

def main() -> None:
    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    model = fit(data.training_flows(), data.cohort)                  # учимся только на train
    out = ROOT / "scoreboard" / "submissions" / NAME
    out.mkdir(parents=True, exist_ok=True)
    for window, hosts, fname in (("train", data.cohort, "embeddings.csv"), ("test", data.evaluable("test"), "test.csv")):
        x = np.asarray(embed(model, data.flows(window), hosts), dtype=float)
        if x.shape != (len(hosts), x.shape[1]) or not np.isfinite(x).all():
            raise SystemExit(f"embed вернул некорректный массив для окна {window}: {x.shape}")
        pd.DataFrame(x, index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(out / fname)
    flag = out / "SHARED_COORDINATES"
    flag.touch() if SHARED_COORDINATES else flag.unlink(missing_ok=True)
    r = submit.score(out)
    submit.render()
    print(f"\n{NAME}: " + ", ".join(f"{s['title']} = {s['value']:.3f}" for s in r["summary"] if s["value"] is not None))
    print("Таблица: scoreboard/README.md. Теперь commit и push.")


if __name__ == "__main__":
    main()
