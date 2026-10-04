import numpy as np
import pandas as pd
import pytest

from benchmark.baselines.make_synthetic_reference import make
from benchmark.baselines.methods import METHODS
from benchmark.baselines.run_all import evaluate, generate


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("reference")
    make(d, n_hosts=120)
    return d


@pytest.fixture(scope="module")
def generated(reference_dir, tmp_path_factory):
    out = tmp_path_factory.mktemp("submissions")
    return generate(reference_dir, out, dim=8, seed=0), out


def test_format_and_host_ids(reference_dir, generated):
    paths, _ = generated
    ref_ids = pd.read_csv(reference_dir / "features.csv")["host_id"]
    assert set(paths) == set(METHODS)
    for name, path in paths.items():
        df = pd.read_csv(path)
        assert list(df.columns)[0] == "host_id"
        assert all(c.startswith("emb_") for c in df.columns[1:])
        assert df["host_id"].tolist() == ref_ids.tolist()
        assert not df.isna().any().any()


def test_dimensions(generated):
    paths, _ = generated
    for name in ("random", "pca", "svd"):
        assert pd.read_csv(paths[name]).shape[1] - 1 == 8
    assert pd.read_csv(paths["reference_identity"]).shape[1] - 1 > 8


def test_deterministic_given_seed(reference_dir, generated, tmp_path):
    paths, _ = generated
    again = generate(reference_dir, tmp_path, dim=8, seed=0)
    for name in paths:
        pd.testing.assert_frame_equal(pd.read_csv(paths[name]), pd.read_csv(again[name]))
    other = generate(reference_dir, tmp_path / "s1", dim=8, seed=1, methods=["random"])
    assert not pd.read_csv(other["random"]).equals(pd.read_csv(paths["random"]))


def test_dim_is_capped_not_crashing(reference_dir, tmp_path):
    with pytest.warns(UserWarning):
        paths = generate(reference_dir, tmp_path, dim=10_000, methods=["pca", "svd"])
    assert pd.read_csv(paths["pca"]).shape[1] > 1


def test_sanity_ordering_and_unavailable_metrics(reference_dir, generated, tmp_path):
    paths, _ = generated
    evaluate(paths, reference_dir, tmp_path)
    import json
    res = {n: json.loads((tmp_path / f"baseline_{n}.json").read_text()) for n in paths}
    c10 = {n: r["metrics"]["consistency@10_mean"] for n, r in res.items()}
    assert c10["reference_identity"] >= c10["pca"] > c10["random"]
    assert c10["svd"] > c10["random"]
    # metrics whose reference artifacts are absent stay unavailable, never invented
    r = res["random"]
    assert any(m.startswith("triplet_accuracy") for m in r["unavailable_metrics"])
    assert "triplet_accuracy" not in "".join(r["metrics"])
