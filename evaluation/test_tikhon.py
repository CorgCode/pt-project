"""Метрики Тихона: прямой прогноз по близости и перенос окрестности."""
import numpy as np
import pandas as pd

from evaluation.tikhon import tikhon_metrics


def _partners(source, target):
    rows = [(a, b, w, c) for w, pairs in (("source", source), ("target", target)) for a, b in pairs
            for a, b, c in ((a, b, 1), (b, a, 1))]
    return pd.DataFrame(rows, columns=["host_id", "partner", "window", "count"])


def test_close_vectors_that_talk_give_perfect_direct_lp():
    # пары (A,B), (C,D): векторы внутри пары совпадают, и в target общаются именно они
    emb = pd.DataFrame([[1, 0], [1, 0], [0, 1], [0, 1]], index=["A", "B", "C", "D"], dtype=float)
    pairs = [("A", "B"), ("C", "D"), ("A", "C"), ("B", "D")]
    r = tikhon_metrics(emb, _partners(pairs, [("A", "B"), ("C", "D")]), k=1)
    assert r["lp_direct"]["mrr"] == 1.0 and r["lp_inverted"]["mrr"] < 1.0
    assert r["n_candidates"] == 4 and r["target_coverage"] == 1.0


def test_transfer_uses_partners_of_the_most_similar_host():
    # A похож на B; B в train общался с E — значит, E первый кандидат для A
    emb = pd.DataFrame([[1, 0], [1, 0.01], [0, 1], [-1, 0], [0, -1]], index=list("ABCDE"), dtype=float)
    src = [("B", "E"), ("C", "D"), ("A", "C"), ("D", "E")]
    p = _partners(src, [])
    p.loc[len(p)] = ["A", "E", "target", 1]                  # проверяем только A
    r = tikhon_metrics(emb, p, k=1)
    assert np.isclose(r["transfer_direct"]["mrr"], 1.0)
    assert set(r) >= {"lp_direct", "transfer_soft", "history", "popularity", "random"}
