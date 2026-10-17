import numpy as np
import pandas as pd
import pytest

from evaluation.identity import identity_retrieval
from evaluation.io import ReferenceError, cosine_pairs, load_embeddings
from evaluation.linear_probe import linear_probe, load_labels
from evaluation.retrieval import load_pairs, pair_auc, precision_recall_at_k
from evaluation.triplet_accuracy import load_triplets, triplet_accuracy
from evaluation.uncertainty import auc_with_ci, bootstrap_ci, reweighted_average_precision


def emb(rows: dict) -> pd.DataFrame:
    return pd.DataFrame(rows).T.astype(float).rename(columns=lambda c: f"emb_{c}")


E = emb({"a": [1, 0], "p": [0.9, 0.1], "n": [0, 1], "q": [0, 1.1]})


def trip(rows, split="test"):
    return pd.DataFrame([(a, p, n, split) for a, p, n in rows], columns=["anchor", "positive", "negative", "split"])


# ---------------------------------------------------------------- точность по тройкам
def test_triplet_perfect_inverted_and_chance_tie():
    assert triplet_accuracy(E, trip([("a", "p", "n")]))["accuracy"] == 1.0
    assert triplet_accuracy(E, trip([("a", "n", "p")]))["accuracy"] == 0.0
    tie = emb({"a": [1, 0], "p": [0, 1], "n": [0, 1]})
    r = triplet_accuracy(tie, trip([("a", "p", "n")]))
    assert r["accuracy"] == 0.5 and r["n_ties"] == 1


def test_triplet_split_filter_and_unknown_split():
    t = pd.concat([trip([("a", "p", "n")], "test"), trip([("a", "n", "p")], "validation")])
    assert triplet_accuracy(E, t, "test")["accuracy"] == 1.0
    assert triplet_accuracy(E, t, "validation")["accuracy"] == 0.0
    with pytest.raises(ReferenceError, match="no rows"):
        triplet_accuracy(E, t, "train")


def test_triplet_missing_host_is_error_unless_allowed_and_counted():
    t = trip([("a", "p", "n"), ("a", "ghost", "n")])
    with pytest.raises(ReferenceError, match="absent"):
        triplet_accuracy(E, t)
    r = triplet_accuracy(E, t, allow_missing=True)
    assert r["n_triplets"] == 1 and r["n_excluded_missing_hosts"] == 1 and r["n_in_file_for_split"] == 2


def test_triplet_row_order_independent():
    t = trip([("a", "p", "n"), ("a", "n", "q"), ("q", "n", "a")])
    a = triplet_accuracy(E, t)["accuracy"]
    assert a == triplet_accuracy(E.sample(frac=1, random_state=1), t.sample(frac=1, random_state=2))["accuracy"]


def test_triplet_loader_validation(tmp_path):
    p = tmp_path / "t.csv"
    for text, msg in [("anchor,positive,negative,split\na,a,n,test\n", "different hosts"),
                      ("anchor,positive,negative,split\na,p,n,bogus\n", "unknown split"),
                      ("anchor,positive,negative,split\na,,n,test\n", "blank"),
                      ("anchor,positive,split\na,p,test\n", "missing columns"),
                      ("anchor,positive,negative,split\n", "empty")]:
        p.write_text(text)
        with pytest.raises(ReferenceError, match=msg):
            load_triplets(p)


def test_triplet_zero_vector_rejected():
    z = emb({"a": [0, 0], "p": [1, 0], "n": [0, 1]})
    with pytest.raises(ReferenceError, match="zero-norm"):
        triplet_accuracy(z, trip([("a", "p", "n")]))


# ---------------------------------------------------------------- AUC по парам и P@k / R@k
def pairs(rows, split="test"):
    return pd.DataFrame([(a, b, str(l), split) for a, b, l in rows], columns=["host_a", "host_b", "label", "split"])


def test_pair_auc_perfect_and_one_class_error():
    P = pairs([("a", "p", 1), ("a", "n", 0), ("n", "q", 1), ("p", "q", 0)])
    r = pair_auc(E, P)
    assert r["roc_auc"] == 1.0 and r["pr_auc"] == 1.0 and r["positive_rate"] == 0.5
    with pytest.raises(ReferenceError, match="both"):
        pair_auc(E, pairs([("a", "p", 1), ("n", "q", 1)]))


def test_pair_loader_rejects_bad_files(tmp_path):
    p = tmp_path / "p.csv"
    for text, msg in [("host_a,host_b,label,split\na,b,2,test\n", "label"),
                      ("host_a,host_b,label,split\na,a,1,test\n", "itself"),
                      ("host_a,host_b,label,split\na,b,1,test\nb,a,0,test\n", "duplicate")]:
        p.write_text(text)
        with pytest.raises(ReferenceError, match=msg):
            load_pairs(p)


def retr_emb():
    # хост x; кандидаты по близости: r1 (похож) > i1 (нет) > r2 (похож) > i2
    return emb({"x": [1, 0], "r1": [1, 0.05], "i1": [1, 0.5], "r2": [1, 1.0], "i2": [0, 1]})


def retr_pairs():
    return pairs([("x", "r1", 1), ("x", "i1", 0), ("x", "r2", 1), ("x", "i2", 0)])


def test_precision_recall_hand_computed():
    r = precision_recall_at_k(retr_emb(), retr_pairs(), ks=(1, 2, 4))["by_k"]
    # только у x |C| >= k; у остальных по 1 кандидату (при k>1 исключаются)
    assert r["2"]["eligible_anchors"] == 1
    assert r["1"]["precision"] == pytest.approx(1.0)       # участвуют хосты с хотя бы одним похожим: x, r1, r2
    assert r["2"]["precision"] == pytest.approx(0.5) and r["2"]["recall"] == pytest.approx(0.5)
    assert r["4"]["precision"] == pytest.approx(0.5) and r["4"]["recall"] == pytest.approx(1.0)
    assert r["2"]["oracle_precision"] == 1.0 and r["2"]["chance_precision"] == 0.5


def test_unjudged_pairs_are_not_negatives():
    base = precision_recall_at_k(retr_emb(), retr_pairs(), ks=(2,))["by_k"]["2"]
    # ближайший по эмбеддингу хост БЕЗ разметки не должен менять результат
    e = pd.concat([retr_emb(), emb({"u": [1, 0.0]})])
    again = precision_recall_at_k(e, retr_pairs(), ks=(2,))["by_k"]["2"]
    assert again["precision"] == base["precision"] and again["recall"] == base["recall"]


def test_ineligible_anchors_excluded_and_coverage_reported():
    P = pairs([("x", "r1", 1), ("x", "i1", 0), ("y", "i1", 0), ("y", "i2", 0)])   # у y нет похожих
    e = precision_recall_at_k(pd.concat([retr_emb(), emb({"y": [0.5, 0.5]})]), P, ks=(2,))["by_k"]["2"]
    assert e["eligible_anchors"] == 1 and 0 < e["coverage"] < 1
    none = precision_recall_at_k(retr_emb(), pairs([("x", "i1", 0), ("x", "i2", 0)]), ks=(1,))
    assert none["by_k"]["1"]["eligible_anchors"] == 0


def test_retrieval_row_order_and_ties_deterministic():
    tie = emb({"x": [1, 0], "b": [0, 1], "a": [0, 1], "c": [1, 0]})
    P = pairs([("x", "a", 1), ("x", "b", 0), ("x", "c", 1)])
    r1 = precision_recall_at_k(tie, P, ks=(2,))["by_k"]["2"]
    r2 = precision_recall_at_k(tie.sample(frac=1, random_state=0), P.sample(frac=1, random_state=0), ks=(2,))["by_k"]["2"]
    assert r1["precision"] == r2["precision"]


def test_retrieval_split_isolation():
    P = pd.concat([retr_pairs(), pairs([("x", "i1", 1), ("x", "r1", 0)], "validation")])
    t = precision_recall_at_k(retr_emb(), P, ks=(1,), split="test")["by_k"]["1"]["precision"]
    v = precision_recall_at_k(retr_emb(), P, ks=(1,), split="validation")["by_k"]["1"]["precision"]
    assert t != v


# ---------------------------------------------------------------- доля положительных и неопределённость
def test_reweighted_ap_matches_ap_at_sample_prevalence_and_drops_when_rarer():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(200), np.zeros(200)].astype(int)
    s = np.r_[rng.normal(1, 1, 200), rng.normal(0, 1, 200)]
    from sklearn.metrics import average_precision_score
    assert reweighted_average_precision(y, s, 0.5) == pytest.approx(average_precision_score(y, s))
    assert reweighted_average_precision(y, s, 0.05) < reweighted_average_precision(y, s, 0.5)
    for bad in (0.0, 1.0, 1.5):
        with pytest.raises(ValueError):
            reweighted_average_precision(y, s, bad)


def test_bootstrap_ci_contains_estimate_and_is_reproducible():
    v = np.random.default_rng(1).normal(0.6, 0.2, 400)
    lo, hi = bootstrap_ci(lambda i: float(v[i].mean()), len(v), n_boot=300, seed=4)
    assert lo < v.mean() < hi and (lo, hi) == bootstrap_ci(lambda i: float(v[i].mean()), len(v), n_boot=300, seed=4)
    assert bootstrap_ci(lambda i: 0.0, 100, n_boot=0) is None


def test_auc_with_ci_chance_interval_includes_half():
    rng = np.random.default_rng(2)
    r = auc_with_ci(rng.integers(0, 2, 500), rng.normal(size=500), n_boot=200)
    lo, hi = r["roc_auc_ci"]
    assert lo < r["roc_auc"] < hi and hi - lo < 0.2 and lo > 0.35 and hi < 0.65


# ---------------------------------------------------------------- линейный зонд
def probe_data(n=60, seed=0, sep=3.0):
    rng = np.random.default_rng(seed)
    y = np.array(["web", "db", "dns"])[rng.integers(0, 3, n)]
    centers = {"web": [sep, 0], "db": [0, sep], "dns": [-sep, -sep]}
    X = np.array([centers[t] for t in y]) + rng.normal(size=(n, 2))
    hosts = [f"h{i}" for i in range(n)]
    split = np.where(np.arange(n) < n * 0.6, "train", "test")
    return (pd.DataFrame(X, index=hosts, columns=["emb_0", "emb_1"]),
            pd.DataFrame({"host_id": hosts, "target": y, "split": split}))


def test_probe_separable_classes_high_and_beats_majority():
    e, lab = probe_data()
    r = linear_probe(e, lab, labels_source="unit-test ground truth")
    assert r["accuracy"] > 0.9 and r["accuracy"] > r["majority_baseline_accuracy"] + 0.3
    assert r["n_train"] + r["n_test"] == len(lab) and r["split"] == "test"


def test_probe_random_embedding_near_majority():
    e, lab = probe_data(n=300, seed=1)
    e = pd.DataFrame(np.random.default_rng(5).normal(size=e.shape), index=e.index, columns=e.columns)
    r = linear_probe(e, lab, labels_source="x")
    assert r["accuracy"] < r["majority_baseline_accuracy"] + 0.15


def test_probe_requires_provenance_and_valid_splits():
    e, lab = probe_data()
    for src in (None, "", "   "):
        with pytest.raises(ReferenceError, match="provenance"):
            linear_probe(e, lab, labels_source=src)
    with pytest.raises(ValueError):
        linear_probe(e, lab, labels_source="x", eval_split="train")
    with pytest.raises(ReferenceError, match="absent"):
        linear_probe(e.iloc[:10], lab, labels_source="x")
    only_one = lab.assign(target="web")
    with pytest.raises(ReferenceError, match="two classes"):
        linear_probe(e, only_one, labels_source="x")
    with pytest.raises(ReferenceError, match="need labelled"):
        linear_probe(e, lab[lab.split == "train"], labels_source="x")


def test_probe_never_trains_on_evaluation_rows():
    e, lab = probe_data(seed=2)
    r1 = linear_probe(e, lab, labels_source="x")
    flipped = lab.copy()                                   # перемешиваем только метки TEST
    flipped.loc[flipped.split == "test", "target"] = np.roll(flipped.loc[flipped.split == "test", "target"].to_numpy(), 7)
    # строки train те же => модель та же => предсказания на test те же
    r2 = linear_probe(e, flipped, labels_source="x")
    assert r1["n_train"] == r2["n_train"] and r1["accuracy"] != r2["accuracy"]


def test_probe_reports_unseen_test_class_and_rejects_duplicate_hosts(tmp_path):
    e, lab = probe_data(seed=3)
    lab.loc[(lab.split == "test") & (lab.index == lab[lab.split == "test"].index[0]), "target"] = "novel"
    assert linear_probe(e, lab, labels_source="x")["n_test_classes_unseen_in_train"] == 1
    p = tmp_path / "l.csv"
    pd.concat([lab, lab.iloc[:1]]).to_csv(p, index=False)
    with pytest.raises(ReferenceError, match="duplicate"):
        load_labels(p)


# ---------------------------------------------------------------- узнаваемость во времени
def rand_orth(d, seed):
    q, _ = np.linalg.qr(np.random.default_rng(seed).normal(size=(d, d)))
    return q


def windows(n=80, d=8, noise=0.05, seed=0):
    rng = np.random.default_rng(seed)
    hosts = [f"h{i}" for i in range(n)]
    A = rng.normal(size=(n, d))
    B = A + rng.normal(scale=noise, size=A.shape)
    return pd.DataFrame(A, index=hosts), pd.DataFrame(B, index=hosts), hosts


def test_identity_shared_coordinates_recovers_hosts():
    A, B, _ = windows()
    r = identity_retrieval(A, B, alignment="none")
    assert r["top1"] > 0.95 and r["mrr"] > 0.95 and r["chance_top1"] == pytest.approx(1 / r["n_eval_hosts"])


def test_identity_rotated_windows_need_alignment():
    A, B, _ = windows()
    Brot = pd.DataFrame(B.to_numpy() @ rand_orth(8, 3), index=B.index)
    raw = identity_retrieval(A, Brot, alignment="none")
    aligned = identity_retrieval(A, Brot, alignment="procrustes")
    assert raw["top1"] < 0.3 and aligned["top1"] > 0.95
    assert aligned["n_alignment_hosts"] + aligned["n_eval_hosts"] == len(A)


def test_identity_random_embeddings_at_chance():
    A, _, hosts = windows(n=200, seed=1)
    B = pd.DataFrame(np.random.default_rng(9).normal(size=A.shape), index=hosts)
    r = identity_retrieval(A, B, alignment="none")
    assert r["top1"] < 0.1 and abs(r["mrr"] - r["chance_mrr"]) < 0.05


def test_identity_constant_embedding_scores_exactly_chance():
    hosts = [f"h{i}" for i in range(60)]
    C = pd.DataFrame(np.ones((60, 4)), index=hosts)
    r = identity_retrieval(C, C, alignment="none")
    assert r["top1"] == pytest.approx(r["chance_top1"]) and r["mrr"] == pytest.approx(r["chance_mrr"])


def test_identity_input_validation():
    A, B, hosts = windows(n=40)
    with pytest.raises(ValueError, match="разные хосты"):
        identity_retrieval(A, B.iloc[:-1])
    with pytest.raises(ValueError, match="размерности"):
        identity_retrieval(A, B.iloc[:, :4], alignment="procrustes")
    with pytest.raises(ValueError, match="alignment"):
        identity_retrieval(A, B, alignment="magic")
    with pytest.raises(ValueError, match="20 хостов"):
        identity_retrieval(A.iloc[:10], B.iloc[:10])


def test_identity_row_order_independent_and_seeded():
    A, B, _ = windows(seed=4)
    a = identity_retrieval(A, B, alignment="procrustes", seed=3)
    b = identity_retrieval(A.sample(frac=1, random_state=1), B.sample(frac=1, random_state=2), alignment="procrustes", seed=3)
    assert a["top1"] == b["top1"] and a["mrr"] == b["mrr"]


# ---------------------------------------------------------------- загрузчики
def test_load_embeddings_rejects_bad_files(tmp_path):
    p = tmp_path / "e.csv"
    for text, msg in [("host_id,emb_0\nA,1\nA,2\n", "duplicate"), ("host_id,emb_0\nA,nan\n", "NaN"),
                      ("host_id,emb_0\nA,inf\nB,1\n", "finite"), ("host_id,emb_0\n,1\n", "blank"),
                      ("id,emb_0\nA,1\n", "missing required"), ("host_id,x\nA,1\n", "no columns"),
                      ("host_id,emb_0\nA,abc\n", "numeric")]:
        p.write_text(text)
        with pytest.raises(ValueError, match=msg):
            load_embeddings(p)


def test_host_ids_are_strings_not_numbers(tmp_path):
    p = tmp_path / "e.csv"
    p.write_text("host_id,emb_0\n001,1\n1,2\n")
    assert list(load_embeddings(p).index) == ["001", "1"]


def test_cosine_pairs_matches_by_id_not_position():
    e = emb({"a": [1, 0], "b": [0, 1], "c": [1, 1]})
    s = cosine_pairs(e, ["a", "a"], ["b", "c"])
    assert s[0] == pytest.approx(0) and s[1] == pytest.approx(1 / np.sqrt(2))
    assert cosine_pairs(e.iloc[::-1], ["a", "a"], ["b", "c"]).tolist() == s.tolist()
