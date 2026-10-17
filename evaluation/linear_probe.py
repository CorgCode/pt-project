"""Линейный зонд на замороженных эмбеддингах с ВНЕШНИМИ метками (reference/labels.csv).

StandardScaler + LogisticRegression(C=1, seed 0) учатся только на split=train и оцениваются на test
(или validation). Эмбеддинги не меняются, на оценочном split ничего не подбирается. Выводятся accuracy,
macro-F1, weighted-F1, balanced accuracy и бейзлайн «самый частый класс».

Зонд имеет смысл, только если метки — независимое внешнее свойство хоста. Код это проверить не может,
поэтому в манифесте эталона ОБЯЗАТЕЛЕН labels_source, иначе зонд недоступен. Метки из тех же логов,
из эмбеддинга или от LLM не годятся. Хост с меткой, но без эмбеддинга — ошибка.

Первая версия — PR #39 (Даша); проверки, правило источника меток, бейзлайны и CI добавлены.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from evaluation.io import ReferenceError, read_reference_table
from evaluation.uncertainty import bootstrap_ci

SPLITS = {"train", "validation", "test"}


def load_labels(path) -> pd.DataFrame:
    lab = read_reference_table(path, {"host_id", "target", "split"}, split_values=SPLITS)
    if lab["host_id"].duplicated().any():
        raise ReferenceError(f"{path}: повторяющийся host_id (у хоста может быть только один split)")
    return lab


def linear_probe(emb: pd.DataFrame, labels: pd.DataFrame, *, labels_source: str | None,
                 eval_split: str = "test", n_boot: int = 0, seed: int = 0, max_iter: int = 2000) -> dict:
    if not labels_source or not str(labels_source).strip():
        raise ReferenceError("у labels.csv не указан внешний источник (labels_source в манифесте); "
                             "непроверенные метки за правду не принимаем")
    if eval_split == "train":
        raise ValueError("оценивать нужно на validation или test, а не на train")
    unknown = sorted(set(labels.host_id) - set(emb.index))
    if unknown:
        raise ReferenceError(f"в labels.csv есть хосты, которых нет в эмбеддинге: {unknown[:5]}")
    tr, te = labels[labels.split == "train"], labels[labels.split == eval_split]
    if tr.empty or te.empty:
        raise ReferenceError(f"нужны размеченные хосты в split 'train' и {eval_split!r}")
    if tr.target.nunique() < 2:
        raise ReferenceError("в train нужно хотя бы два класса")
    Xtr, Xte = emb.loc[tr.host_id].to_numpy(float), emb.loc[te.host_id].to_numpy(float)
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=max_iter, random_state=seed))
    clf.fit(Xtr, tr.target.to_numpy())
    pred = clf.predict(Xte)
    y = te.target.to_numpy()
    majority = tr.target.value_counts().idxmax()
    correct = (pred == y).astype(float)
    out = {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "majority_baseline_accuracy": float((y == majority).mean()),
        "n_train": int(len(tr)), "n_test": int(len(te)), "n_classes": int(tr.target.nunique()),
        "n_test_classes_unseen_in_train": int(len(set(y) - set(tr.target))),
        "label_coverage": float(len(labels) / len(emb)),
        "split": eval_split, "labels_source": str(labels_source),
    }
    out["accuracy_ci"] = bootstrap_ci(lambda i: float(correct[i].mean()), len(y), n_boot=n_boot, seed=seed)
    return out
