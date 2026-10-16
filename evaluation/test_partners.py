import numpy as np
import pandas as pd
import pytest

from evaluation.evaluate import run
from evaluation.io import ReferenceError
from evaluation.partners import _mid_ranks, build_partners, load_partners, partner_ranking


def _flows(pairs):
    return pd.DataFrame(pairs, columns=["src", "dst"])


def test_build_partners_is_undirected_and_drops_loops():
    p = build_partners(_flows([("a", "x"), ("x", "a"), ("a", "a")]), _flows([("b", "y")]), ["a", "b"])
    src = p[p.window == "source"]
    assert src.to_dict("records") == [{"host_id": "a", "partner": "x", "window": "source", "count": 2}]
    assert p[p.window == "target"].partner.tolist() == ["y"]


def test_mid_rank_splits_ties():
    assert _mid_ranks(np.array([1.0, 1.0, 0.0]), np.array([0, 2])).tolist() == [1.5, 3.0]


def test_role_structure_is_recovered():
    # две роли: хосты роли r общаются с серверами s_r; эмбеддинг = роль
    hosts = [f"h{r}{i}" for r in range(2) for i in range(10)]
    src, tgt = [], []
    for h in hosts:
        r = int(h[1])
        i = int(h[2:])
        src += [(h, f"s{r}{(i + j) % 5}") for j in range(2)]          # в train — 2 из 5 серверов своей роли
        tgt += [(h, f"s{r}{(i + 3) % 5}")]                            # в target — новый сервер своей роли
    partners = build_partners(_flows(src), _flows(tgt), hosts)
    good = pd.DataFrame([[1.0, 0.0] if h[1] == "0" else [0.0, 1.0] for h in hosts], index=hosts)
    bad = pd.DataFrame(np.random.default_rng(0).normal(size=(20, 2)), index=hosts)
    rg, rb = partner_ranking(good, partners, k=5), partner_ranking(bad, partners, k=5)
    assert rg["new"]["embedding"]["mrr"] > rb["new"]["embedding"]["mrr"]
    assert rg["new"]["embedding"]["mrr"] > rg["new"]["history"]["mrr"]       # история не знает новых
    assert rg["target_coverage"] == 1.0


def test_load_partners_rejects_bad_window(tmp_path):
    f = tmp_path / "p.csv"
    pd.DataFrame({"host_id": ["a"], "partner": ["b"], "window": ["later"], "count": [1]}).to_csv(f, index=False)
    with pytest.raises(ReferenceError, match="window"):
        load_partners(f)


def test_evaluator_reports_partner_metrics(reference, tmp_path):
    hosts = pd.read_csv(reference / "test" / "eval_hosts.csv", dtype=str).host_id.tolist()
    p = tmp_path / "e.csv"
    pd.DataFrame(np.random.default_rng(0).normal(size=(len(hosts), 4)),
                 index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(p)
    r = run(p, reference / "test", "m")
    assert r["metric_status"]["partner_ranking"] == "ok"
    assert {"partner_mrr", "partner_new_mrr", "partner_history_mrr"} <= set(r["metrics"])
    assert any(s["metric"] == "partner_new_mrr" for s in r["summary"])
