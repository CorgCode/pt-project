import json

import numpy as np
import pytest

from methods import graph as methods
from methods.graph.conftest_helpers import TrainOnlyGuard
from methods.graph.deepwalk import DeepWalk
from methods.graph.graph_input import from_edges
from methods.graph.node2vec import Node2Vec
from methods.graph.sgns import SkipGram
from methods.graph.walks import walk_corpus
from experiments.run_experiment import run_experiment
from experiments.tune_methods import tune

FAST = dict(num_walks=3, walk_length=15, window=3)


@pytest.fixture(scope="module")
def data(processed):
    return TrainOnlyGuard(processed)


def test_registered_with_defaults_and_validation():
    assert "node2vec" in methods.available()
    m = Node2Vec(dim=4)
    assert (m.params["p"], m.params["q"]) == (1.0, 1.0) and "Grover" in m.reference
    for bad in ({"p": 0}, {"q": -1.0}):
        with pytest.raises(ValueError, match="> 0"):
            Node2Vec(dim=4, **bad)
    with pytest.raises(ValueError, match="неизвестные параметры"):
        Node2Vec(dim=4, objective="hs")                 # node2vec — только negative sampling
    assert m.info()["params"]["q"] == 1.0 and m.info()["reference"]


def test_p_and_q_equal_to_one_reproduce_deepwalk_with_negative_sampling(data):
    kw = dict(dim=6, seed=3)
    n2v = Node2Vec(**kw, **FAST).fit(data).embed(data, "train")
    dw = DeepWalk(**kw, **FAST, objective="neg").fit(data).embed(data, "train")
    assert np.array_equal(n2v.to_numpy(), dw.to_numpy())             # одинаковые блуждания, цель и seed


def test_bias_changes_the_embedding_and_is_recorded(data):
    a = Node2Vec(dim=6, seed=3, **FAST).fit(data).embed(data, "train")
    b = Node2Vec(dim=6, seed=3, p=0.25, q=4.0, **FAST).fit(data)
    assert not np.array_equal(a.to_numpy(), b.embed(data, "train").to_numpy())
    assert b.info()["params"]["p"] == 0.25 and b.info()["params"]["q"] == 4.0


def planted(seed, h=30, pin=0.25, pout=0.02):
    rng = np.random.default_rng(seed)
    n = 2 * h
    s, d = [], []
    for i in range(n):
        for j in range(i + 1, n):
            if rng.random() < (pin if (i < h) == (j < h) else pout):
                s.append(i); d.append(j)
    return from_edges(np.array(s), np.array(d), np.ones(len(s)), np.array([f"h{i}" for i in range(n)], dtype=object)), h


def community_gap(q, seed):
    g, h = planted(seed)
    walks = walk_corpus(g, 10, 30, q=q, seed=seed)
    E = SkipGram(g.n, 16, objective="neg", window=5, epochs=2, seed=seed, node_freq=g.strength).fit(walks).embeddings
    E = E / np.linalg.norm(E, axis=1, keepdims=True)
    S = E @ E.T
    return (S[:h, :h].sum() - h + S[h:, h:].sum() - h) / (2 * h * (h - 1)) - S[:h, h:].mean()


def test_low_q_separates_communities_better_than_high_q():
    """Утверждение статьи про гомофилию (блуждания вглубь, q < 1): устойчивый монотонный эффект на разных графах."""
    gaps = {q: np.mean([community_gap(q, s) for s in range(4)]) for q in (0.25, 1.0, 4.0)}
    assert gaps[0.25] > gaps[1.0] > gaps[4.0], gaps
    assert gaps[0.25] - gaps[4.0] > 0.05


def test_output_contract_and_determinism(data):
    mk = lambda s: Node2Vec(dim=5, seed=s, p=0.5, q=2.0, **FAST).fit(data).embed(data, "train")
    e = mk(1)
    assert list(e.index) == data.cohort and np.isfinite(e.to_numpy()).all() and (np.linalg.norm(e.to_numpy(), axis=1) > 0).all()
    assert np.array_equal(e.to_numpy(), mk(1).to_numpy()) and not np.array_equal(e.to_numpy(), mk(2).to_numpy())


def test_fit_reads_only_the_train_window(processed):
    d = TrainOnlyGuard(processed)
    d.fitting = True
    try:
        Node2Vec(dim=4, p=0.5, q=2.0, num_walks=2, walk_length=10, window=3).fit(d)
    finally:
        d.fitting = False


def test_runs_through_the_runner_and_the_tuner(processed, reference, tmp_path):
    m = run_experiment(processed, reference, tmp_path / "o", dim=6, n_boot=0, do_audit=False, methods=["node2vec"],
                       method_config={"node2vec": dict(p=0.5, q=2.0, **FAST)})
    assert "node2vec" in m["methods"] and "node2vec" not in m["missing_expected_methods"]
    assert m["graph_methods"]["node2vec"]["params"]["p"] == 0.5
    res = json.loads((tmp_path / "o" / "results" / "test" / "node2vec.json").read_text())
    assert res["metric_status"]["pairwise_roc_auc"] == "ok"
    out = tune(processed, reference, "node2vec", {"p": [0.5, 2.0], "q": [0.5, 2.0]}, dim=4, metric="pair_roc_auc", fast=True)
    assert out["n_configurations"] == 4 and set(out["best_config"]) == {"p", "q"} and out["test_reference_read"] is False


def test_service_graph_separates_roles_that_share_the_same_servers():
    """Роли различаются только сервисами (портами), а серверы у всех общие: граф «кто с кем» роли
    не видит, граф «хост — сервис» — видит."""
    import numpy as np
    import pandas as pd
    from methods.graph.graph_input import from_flows
    from methods.graph.node2vec import Node2Vec

    rng = np.random.default_rng(0)
    servers = [f"s{j}" for j in range(4)]
    ports = {0: ["22", "3389"], 1: ["80", "443"], 2: ["53", "123"]}
    rows, role = [], {}
    for r, ps in ports.items():
        for i in range(15):
            h = f"r{r}h{i}"
            role[h] = r
            for _ in range(30):
                rows.append((h, rng.choice(servers), "6", rng.choice(ps)))
    flows = pd.DataFrame(rows, columns=["src", "dst", "proto", "dst_port"])
    hosts = sorted(role)

    def separation(graph):
        g = from_flows(flows, graph=graph)
        m = Node2Vec(dim=8, seed=0, num_walks=10, walk_length=20, window=4, epochs=2)
        x = m.embed_graph(g)[g.index_of(hosts)]
        x = x / np.linalg.norm(x, axis=1, keepdims=True)
        sim = x @ x.T
        same = np.array([[role[a] == role[b] for b in hosts] for a in hosts])
        np.fill_diagonal(same, False)
        off = ~np.eye(len(hosts), dtype=bool)
        return sim[same].mean() - sim[off & ~same].mean()

    assert separation("services") > separation("hosts") + 0.2
