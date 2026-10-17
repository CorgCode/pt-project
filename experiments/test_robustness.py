import numpy as np
import pandas as pd

from experiments.robustness import ReducedData, knn_overlap, procrustes_cosine, robustness
from pipeline.data import ProcessedData


def _emb(x, hosts):
    return pd.DataFrame(x, index=hosts)


def test_overlap_and_procrustes_are_one_for_rotated_copy():
    rng = np.random.default_rng(0)
    hosts = [f"h{i}" for i in range(50)]
    a = rng.normal(size=(50, 6))
    q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
    assert knn_overlap(_emb(a, hosts), _emb(a @ q, hosts)) == 1.0
    assert procrustes_cosine(_emb(a, hosts), _emb(a @ q, hosts)) > 0.999
    assert knn_overlap(_emb(a, hosts), _emb(rng.normal(size=(50, 6)), hosts)) < 0.5
    assert procrustes_cosine(_emb(a, hosts), _emb(a[:, :4], hosts)) is None


def test_reduced_data_drops_whole_pairs_from_train_only(processed):
    base = ProcessedData(processed)
    d = ReducedData(base, 0.3, seed=0)
    full, cut = base.training_flows(), d.training_flows()
    assert len(cut) < len(full) and d.n_dropped == round(0.3 * d.n_pairs)
    key = lambda f: set(map(frozenset, zip(f.src.astype(str), f.dst.astype(str))))
    assert len(key(full) - key(cut)) == d.n_dropped                 # пары удаляются целиком
    assert len(d.flows("test")) == len(base.flows("test"))          # другие окна не тронуты


def test_robustness_report(processed, tmp_path):
    r = robustness(processed, tmp_path, ["behavioral", "random"], fractions=(0.2,))
    b = r["methods"]["behavioral"]
    assert b["noise_floor"]["overlap"] == 1.0                       # детерминированный метод
    rnd = r["methods"]["random"]
    assert rnd["noise_floor"]["overlap"] < 0.3                      # случайный эмбеддинг: другой seed — другие соседи
    assert b["by_fraction"]["0.2"]["overlap"] > rnd["noise_floor"]["overlap"]
    assert "Устойчивость" in (tmp_path / "robustness.md").read_text(encoding="utf-8")
