import numpy as np
import pandas as pd
import pytest

from evaluation.clustering import (clustering_ari, clustering_nmi, cluster_embeddings,
                                   compare_with_random, random_baseline)
from evaluation.evaluate import run


def _blob_emb(rng, centers, n_per_cluster=30, scale=0.2):
    rows, ids = [], []
    for label, center in enumerate(centers):
        for i in range(n_per_cluster):
            rows.append(rng.normal(center, scale))
            ids.append(f"c{label}_h{i}")
    return pd.DataFrame(rows, index=ids).rename(columns=lambda c: f"emb_{c}")


def test_cluster_embeddings_finds_structure_and_baseline():
    rng = np.random.default_rng(0)
    emb = _blob_emb(rng, [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])

    details, assignments = cluster_embeddings(emb, min_cluster_size=10, min_samples=5)

    assert details["n_clusters"] >= 3
    assert 0.0 <= details["noise_fraction"] <= 1.0
    assert details["silhouette"] is not None and details["silhouette"] > 0.3
    assert details["davies_bouldin"] is not None and details["davies_bouldin"] < 1.5
    assert len(assignments) == len(emb)
    assert set(assignments.columns) == {"cluster"}

    baseline = random_baseline(emb, assignments, seed=0, n_draws=3)
    report = compare_with_random(details, baseline)
    assert report["delta_silhouette_vs_random"] > 0.2
    assert report["delta_davies_bouldin_vs_random"] > 0.0


def test_cosine_is_scale_invariant_and_zeros_do_not_crash():
    rng = np.random.default_rng(1)
    base = _blob_emb(rng, [[4.0, 0.0, 0.0, 0.0], [0.0, 4.0, 0.0, 0.0]], n_per_cluster=20)
    scaled = base.copy()
    scaled.iloc[:20] *= 100.0
    scaled.iloc[20:] *= 0.01

    plain, _ = cluster_embeddings(base, metric="cosine", min_cluster_size=8, min_samples=3)
    same, _ = cluster_embeddings(scaled, metric="cosine", min_cluster_size=8, min_samples=3)
    assert plain["n_clusters"] == same["n_clusters"]
    assert plain["silhouette"] == pytest.approx(same["silhouette"], abs=1e-6)

    with_zero = base.copy()
    with_zero.iloc[0] = 0.0
    details, assignments = cluster_embeddings(with_zero, metric="cosine", min_cluster_size=8, min_samples=3)
    assert len(assignments) == len(with_zero)
    assert details["n_clusters"] >= 1


def test_noise_only_or_single_cluster_has_no_internal_scores():
    emb = pd.DataFrame(np.random.default_rng(2).normal(size=(8, 4)),
                       index=[f"h{i}" for i in range(8)]).rename(columns=lambda c: f"emb_{c}")
    details, assignments = cluster_embeddings(emb, min_cluster_size=5, min_samples=5)

    assert details["n_clusters"] <= 1
    assert details["silhouette"] is None
    assert details["davies_bouldin"] is None
    assert (assignments["cluster"] == -1).any()


def test_ari_nmi_require_matching_hosts_and_labels():
    rng = np.random.default_rng(3)
    emb = _blob_emb(rng, [[3.0, 0.0], [0.0, 3.0], [3.0, 3.0]], n_per_cluster=25)
    _, assignments = cluster_embeddings(emb, min_cluster_size=8, min_samples=3)
    truth = pd.Series({idx: idx.split("_")[0] for idx in emb.index}, name="target")

    ari = clustering_ari(assignments, truth)
    nmi = clustering_nmi(assignments, truth)
    assert ari > 0.6
    assert 0.0 <= nmi <= 1.0

    wrong = truth.copy()
    wrong.index = [f"x{i}" for i in range(len(wrong))]
    with pytest.raises(ValueError, match="разные host_id"):
        clustering_ari(assignments, wrong)


def test_invalid_inputs_raise_clear_errors():
    ok = pd.DataFrame({"emb_0": [0.0, 1.0, 2.0, 3.0, 4.0], "emb_1": [0.0, 1.0, 2.0, 3.0, 4.0]},
                      index=["a", "b", "c", "d", "e"])
    with pytest.raises(ValueError, match="min_cluster_size"):
        cluster_embeddings(ok, min_cluster_size=1)
    with pytest.raises(ValueError, match="число хостов"):
        cluster_embeddings(ok.iloc[:2], min_cluster_size=5)

    bad = ok.copy()
    bad.loc["a", "emb_0"] = np.nan
    with pytest.raises(ValueError, match="конечной"):
        cluster_embeddings(bad, min_cluster_size=2)

    dup = pd.DataFrame([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], index=["h", "h", "c", "d", "e"],
                       columns=["emb_0", "emb_1"])
    with pytest.raises(ValueError, match="host_id"):
        cluster_embeddings(dup, min_cluster_size=2)


def test_evaluate_adds_clustering_metric(reference, processed, tmp_path):
    from pipeline.data import ProcessedData

    cohort = ProcessedData(processed).cohort
    emb = pd.DataFrame(np.random.default_rng(4).normal(size=(len(cohort), 8)),
                       index=pd.Index(cohort, name="host_id")).add_prefix("emb_")
    path = tmp_path / "emb.csv"
    emb.to_csv(path)

    r = run(path, reference / "test", "random", n_boot=0, seed=0)
    assert r["metric_status"]["embedding_clustering"] == "ok"
    m = r["metrics"]
    assert "clustering_n_clusters" in m
    assert "clustering_noise_fraction" in m
    assert "clustering_silhouette_random" in m
    assert "clustering_silhouette_gain" in m
    detail = r["metric_details"]["embedding_clustering"]
    assert detail["algorithm"] == "HDBSCAN"
    assert len(detail["assignments"]) == len(cohort)
    assert all(str(h) in detail["assignments"] for h in cohort)


def test_cluster_embeddings_reports_external_labels_for_model():
    rng = np.random.default_rng(5)
    emb = _blob_emb(rng, [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]], n_per_cluster=40)
    truth = pd.Series({idx: idx.split("_")[0] for idx in emb.index}, name="target")

    details, assignments = cluster_embeddings(emb, min_cluster_size=10, min_samples=5, expected=truth)
    baseline = random_baseline(emb, assignments, seed=0, n_draws=2, expected=truth)
    report = compare_with_random(details, baseline)

    assert details["ari"] is not None and details["ari"] > 0.5
    assert details["nmi"] is not None and 0.0 <= details["nmi"] <= 1.0
    assert report["delta_ari_vs_random"] is not None
    assert report["delta_nmi_vs_random"] is not None
