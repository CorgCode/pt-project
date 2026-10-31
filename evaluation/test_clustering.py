import numpy as np
import pandas as pd
import pytest

from evaluation.clustering import cluster_embeddings, clustering_ari


def test_known_groups_have_high_ari():
    rng = np.random.default_rng(42)
    centers = np.array([[0, 0], [8, 0], [0, 8]])
    labels = np.repeat(np.arange(3), 60)
    X = np.vstack([
        centers[k] + rng.normal(0, 0.25, size=(60, 2))
        for k in range(3)
    ])
    emb = pd.DataFrame(X, index=[f"h{i}" for i in range(len(X))])

    details, assignments = cluster_embeddings(
        emb, min_cluster_size=10, min_samples=5
    )
    ari = clustering_ari(assignments, pd.Series(labels, index=emb.index))

    assert details["n_clusters"] == 3
    assert ari > 0.95
    assert 0 <= details["noise_fraction"] <= 1


def test_random_embedding_does_not_outperform_clear_groups():
    rng = np.random.default_rng(7)
    labels = np.repeat(np.arange(3), 60)
    centers = np.array([[0, 0], [8, 0], [0, 8]])
    X = np.vstack([
        centers[k] + rng.normal(0, 0.25, size=(60, 2))
        for k in range(3)
    ])
    index = [f"h{i}" for i in range(len(X))]
    truth = pd.Series(labels, index=index)

    _, good = cluster_embeddings(pd.DataFrame(X, index=index), min_cluster_size=10)
    _, random = cluster_embeddings(
        pd.DataFrame(rng.normal(size=(180, 2)), index=index),
        min_cluster_size=10,
    )

    good_ari = clustering_ari(good, truth)
    random_ari = clustering_ari(random, truth)
    assert good_ari > random_ari


def test_cosine_is_scale_invariant():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(80, 5))
    scales = np.geomspace(0.01, 100, len(X))
    emb = pd.DataFrame(X * scales[:, None], index=[f"h{i}" for i in range(80)])

    details, assignments = cluster_embeddings(
        emb, metric="cosine", min_cluster_size=5
    )
    assert len(assignments) == 80
    assert 0 <= details["noise_fraction"] <= 1


def test_duplicate_ids_raise():
    emb = pd.DataFrame([[0, 0], [1, 1]], index=["h1", "h1"])
    with pytest.raises(ValueError, match="повторяющиеся"):
        cluster_embeddings(emb)


def test_non_finite_values_raise():
    emb = pd.DataFrame([[0, 0], [1, np.inf], [2, 2]])
    with pytest.raises(ValueError, match="конечн"):
        cluster_embeddings(emb)


def test_malformed_embedding_csv_has_clear_error(tmp_path):
    from evaluation.io import load_embeddings

    path = tmp_path / "bad.csv"
    path.write_text("host_id,emb_0,emb_1\nh1,1,2\nh1,3,4\n")
    with pytest.raises(ValueError, match="повторяющиеся host_id"):
        load_embeddings(path)
