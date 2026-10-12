import numpy as np
import pandas as pd
import pytest
from evaluation.temporal_auc import evaluate_method, load_temporal_pairs


def fixture_data():
    hosts = [f"h{i:02d}" for i in range(40)]
    rng = np.random.default_rng(9)
    a = pd.DataFrame(rng.normal(size=(40, 4)), index=hosts, columns=[f"emb_{i}" for i in range(4)])
    b = a + rng.normal(scale=.03, size=a.shape)
    rows = []
    for split, cohort in (("train", hosts[:20]), ("validation", hosts[20:30]), ("test", hosts[30:])):
        for h in cohort:
            rows.append(("A", h, "B", h, 1, split))
            for other in cohort:
                if other != h: rows.append(("A", h, "B", other, 0, split))
    p = pd.DataFrame(rows, columns=["window_a", "host_a", "window_b", "host_b", "label", "split"])
    p["label"] = p.label.astype(str)
    p["split"] = p.split.astype(str)
    return a, b, p


def test_reports_auc_metrics_for_validation_and_test():
    a, b, p = fixture_data()
    rows = evaluate_method(a, b, p, method="toy", n_boot=0)
    assert {r["metric"] for r in rows} == {"roc_auc", "pr_auc", "nearest_candidate_accuracy"}
    assert {r["split"] for r in rows} == {"validation", "test"}
    assert all(0 <= r["score_value"] <= 1 for r in rows)
    assert all(r["score_name"] == "cosine_procrustes" for r in rows)
    retrieval = [r for r in rows if r["metric"] == "nearest_candidate_accuracy"]
    assert len(retrieval) == 2
    assert all(r["n_examples"] == 10 for r in retrieval)
    assert all(r["score_value"] >= 0.9 for r in retrieval)


def test_loader_rejects_wrong_identity_labels(tmp_path):
    p = tmp_path / "pairs.csv"
    pd.DataFrame([{"window_a":"A", "host_a":"h1", "window_b":"B", "host_b":"h1", "label":0, "split":"test"}]).to_csv(p, index=False)
    with pytest.raises(ValueError, match="label must be 1"):
        load_temporal_pairs(p)
