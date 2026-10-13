import pandas as pd
import pytest
from evaluation.temporal_pairs import build_pairs, common_cohort


def test_build_pairs_deterministic_and_disjoint_splits():
    hosts = [f"h{i:02d}" for i in range(30)]
    a = build_pairs(hosts, "A", "B", seed=41, negatives_per_positive=2)
    b = build_pairs(hosts, "A", "B", seed=41, negatives_per_positive=2)
    pd.testing.assert_frame_equal(a, b)
    assert set(a.label) == {0, 1}
    assert (a.loc[a.label == 1, "host_a"] == a.loc[a.label == 1, "host_b"]).all()
    assert (a.loc[a.label == 0, "host_a"] != a.loc[a.label == 0, "host_b"]).all()
    cohorts = {s: set(a.loc[a.split == s, "host_a"]) for s in ("train", "validation", "test")}
    assert not cohorts["train"] & cohorts["validation"]
    assert not cohorts["train"] & cohorts["test"]
    assert not cohorts["validation"] & cohorts["test"]


def test_build_pairs_rejects_small_cohort():
    with pytest.raises(ValueError, match="at least 10"):
        build_pairs(["a", "b", "c"], "A", "B")


def test_common_cohort_intersects_flow_threshold_and_all_methods(tmp_path):
    fa, fb = tmp_path / "fa.csv", tmp_path / "fb.csv"
    pd.DataFrame({"src": ["h1", "h1", "h2", "h3", "h3"],"dst": ["h2", "h2", "h1", "h4", "h1"]}).to_csv(fa, index=False)
    pd.DataFrame({"src": ["h1", "h2", "h2", "h3"], "dst": ["h2", "h1", "h3", "h4"]}).to_csv(fb, index=False)
    def emb(name, ids):
        p = tmp_path / name
        pd.DataFrame({"host_id": ids, "emb_0": [1.] * len(ids), "emb_1": [.5] * len(ids)}).to_csv(p, index=False)
        return p
    ea = {"m1": emb("m1a.csv", ["h1", "h2", "h3", "h4"]), "m2": emb("m2a.csv", ["h1", "h2", "h3"])}
    eb = {"m1": emb("m1b.csv", ["h1", "h2", "h3", "h4"]), "m2": emb("m2b.csv", ["h1", "h2", "h3"])}
    assert common_cohort(fa, fb, ea, eb, 2) == ["h1", "h2", "h3"]
