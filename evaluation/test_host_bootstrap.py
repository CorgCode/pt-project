"""Пары делят хостов, поэтому бутстреп по парам занижает неопределённость; пересэмплировать нужно хосты."""
import itertools

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

from evaluation.triplet_accuracy import triplet_accuracy
from evaluation.uncertainty import auc_with_ci, dyadic_bootstrap_ci, host_index


def draw(rng, n):
    u = rng.normal(0, 1.2, n)                                   # эффекты хостов, общие для многих пар
    pairs = np.array(list(itertools.combinations(range(n), 2)))
    latent = u[pairs[:, 0]] + u[pairs[:, 1]]
    y = (rng.random(len(latent)) < 1 / (1 + np.exp(-latent))).astype(int)
    return pairs, y, latent + rng.normal(0, 1.0, len(latent))


def test_host_index_maps_equal_ids_to_equal_integers():
    cols, n = host_index(["a", "b", "a"], ["b", "c", "c"])
    assert n == 3
    assert cols[0].tolist() == [0, 1, 0] and cols[1].tolist() == [1, 2, 2]     # a=0, b=1, c=2 в обеих колонках


def test_unit_weights_reproduce_the_point_estimate():
    rng = np.random.default_rng(0)
    pairs, y, s = draw(rng, 25)
    seen = []
    dyadic_bootstrap_ci(lambda w: seen.append(w.copy()) or float(np.average(y, weights=w)),
                        [pairs[:, 0], pairs[:, 1]], 25, n_boot=20, seed=1)
    assert all((w >= 0).all() for w in seen) and any((w == 0).any() for w in seen)       # часть пар выпадает
    # повтор, где каждый хост выпал ровно один раз, даёт каждой паре вес 1
    counts = np.ones(25)
    assert np.allclose(counts[pairs[:, 0]] * counts[pairs[:, 1]], 1.0)
    assert roc_auc_score(y, s, sample_weight=np.ones(len(y))) == pytest.approx(roc_auc_score(y, s))


def test_weights_are_the_product_of_endpoint_multiplicities():
    captured = []
    pairs = np.array([[0, 1], [1, 2], [0, 2], [2, 3]])
    dyadic_bootstrap_ci(lambda w: captured.append(w.copy()) or 0.0, [pairs[:, 0], pairs[:, 1]], 4, n_boot=12, seed=3)
    rng = np.random.default_rng(3)                              # повторяем выбор хостов первого повтора
    counts = np.bincount(rng.integers(0, 4, 4), minlength=4)
    expected = counts[pairs[:, 0]] * counts[pairs[:, 1]]
    assert captured[0].tolist() == expected.tolist()
    triplet_w = []
    trip = np.array([[0, 1, 2], [1, 2, 3]])
    dyadic_bootstrap_ci(lambda w: triplet_w.append(w.copy()) or 0.0, [trip[:, 0], trip[:, 1], trip[:, 2]], 4,
                        n_boot=12, seed=3)
    assert triplet_w[0].tolist() == (counts[trip[:, 0]] * counts[trip[:, 1]] * counts[trip[:, 2]]).tolist()


def test_host_bootstrap_is_wider_and_better_calibrated_than_the_pair_bootstrap():
    truth = float(np.mean([roc_auc_score(*draw(np.random.default_rng(i), 60)[1:]) for i in range(40)]))
    host_cov = item_cov = 0
    host_w = item_w = 0.0
    reps = 30
    for r in range(reps):
        pairs, y, s = draw(np.random.default_rng(1000 + r), 30)
        hosts = host_index(pairs[:, 0].astype(str), pairs[:, 1].astype(str))
        h = auc_with_ci(y, s, n_boot=100, seed=r, hosts=hosts)
        i = auc_with_ci(y, s, n_boot=100, seed=r)
        host_cov += h["roc_auc_ci"][0] <= truth <= h["roc_auc_ci"][1]
        item_cov += i["roc_auc_ci"][0] <= truth <= i["roc_auc_ci"][1]
        host_w += h["roc_auc_ci"][1] - h["roc_auc_ci"][0]
        item_w += i["roc_auc_ci"][1] - i["roc_auc_ci"][0]
    assert host_w > 1.5 * item_w                                  # примерно вдвое шире
    assert host_cov / reps >= 0.85 and item_cov / reps < host_cov / reps    # 95% только для хостов
    assert item_cov / reps <= 0.85                                # бутстреп по парам недопокрывает


def test_ci_method_is_reported():
    pairs, y, s = draw(np.random.default_rng(5), 20)
    hosts = host_index(pairs[:, 0].astype(str), pairs[:, 1].astype(str))
    assert "host-cluster" in auc_with_ci(y, s, n_boot=20, hosts=hosts)["ci_method"]
    assert "independent" in auc_with_ci(y, s, n_boot=20)["ci_method"]


def test_triplet_accuracy_uses_the_host_bootstrap():
    import pandas as pd
    rng = np.random.default_rng(2)
    emb = pd.DataFrame(rng.normal(size=(30, 4)), index=[f"h{i}" for i in range(30)])
    trip = pd.DataFrame([(f"h{a}", f"h{p}", f"h{n}", "test") for a, p, n in
                         (rng.choice(30, 3, replace=False) for _ in range(200))],
                        columns=["anchor", "positive", "negative", "split"])
    r = triplet_accuracy(emb, trip, n_boot=100, seed=1)
    assert "host-cluster" in r["ci_method"] and r["accuracy_ci"][0] <= r["accuracy"] <= r["accuracy_ci"][1]


def test_evaluator_records_the_ci_method_per_metric(reference, tmp_path):
    import pandas as pd
    from evaluation.evaluate import run
    from pipeline.data import ProcessedData
    hosts = pd.read_csv(reference / "test" / "eval_hosts.csv", dtype=str).host_id.tolist()
    p = tmp_path / "e.csv"
    pd.DataFrame(np.random.default_rng(0).normal(size=(len(hosts), 5)),
                 index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(p)
    r = run(p, reference / "test", "m", n_boot=30)
    for k in ("link_roc_auc", "pair_roc_auc", "triplet_accuracy"):
        assert "host-cluster" in r["ci_methods"][k], k
    assert "bootstrap" in r["ci_methods"]["consistency@10_mean"]
