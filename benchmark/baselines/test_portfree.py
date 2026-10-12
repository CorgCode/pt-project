import numpy as np
import pandas as pd
import pytest

from benchmark.baselines.temporal import (BehavioralBaseline, BehavioralPCA, BehavioralPortFree, BehavioralPortFreePCA,
                                          default_baselines)
from pipeline.data import ProcessedData


class Guard(ProcessedData):
    fitting = False

    def flows(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} window")
        return super().flows(window)


class PortsShuffled(ProcessedData):
    """Same data with the destination-port column permuted in EVERY window."""

    def _shuf(self, df):
        out = df.copy()
        out["dst_port"] = np.random.default_rng(1).permutation(out["dst_port"].to_numpy())
        return out

    def flows(self, window):
        return self._shuf(super().flows(window))

    def training_flows(self):
        return self._shuf(super().training_flows())


def test_variants_are_registered_only_on_request_and_do_not_change_the_defaults():
    names = lambda **kw: [b.name for b in default_baselines(4, 0, **kw)]
    assert "behavioral_portfree" not in names() and names() == ["random", "behavioral", "behavioral_pca", "behavioral_svd", "graph_spectral"]
    assert names(port_free=True)[-2:] == ["behavioral_portfree", "behavioral_portfree_pca"]


def test_port_free_embedding_drops_exactly_the_port_columns(processed):
    d = ProcessedData(processed)
    full = BehavioralBaseline().fit(d).embed(d, "train")
    free = BehavioralPortFree().fit(d).embed(d, "train")
    n_ports = len(BehavioralBaseline().fit(d)._train_feats.filter(like="ports_").columns)
    assert n_ports > 0 and free.shape[1] == full.shape[1] - n_ports and list(free.index) == d.cohort
    assert BehavioralPortFree().info()["excluded_feature_blocks"] == ["ports"]
    assert BehavioralBaseline().info()["excluded_feature_blocks"] == []
    assert "port block removed" in BehavioralPortFree.description


def test_port_free_embedding_is_invariant_to_port_values_but_the_full_one_is_not(processed):
    d, s = ProcessedData(processed), PortsShuffled(processed)
    for cls in (BehavioralPortFree,):
        for window in ("train", "test"):
            hosts = d.evaluable("test")
            a = cls().fit(d).embed(d, window, hosts).to_numpy()
            b = cls().fit(s).embed(s, window, hosts).to_numpy()
            assert np.array_equal(a, b), window                                       # bit for bit
    assert not np.array_equal(BehavioralBaseline().fit(d).embed(d, "train").to_numpy(),
                              BehavioralBaseline().fit(s).embed(s, "train").to_numpy())


def test_fit_reads_only_the_train_window(processed):
    d = Guard(processed)
    d.fitting = True
    try:
        BehavioralPortFree().fit(d)
        BehavioralPortFreePCA(dim=4).fit(d)
    finally:
        d.fitting = False


def test_later_windows_use_train_scaling_and_pca_keeps_the_exclusion(processed):
    d = ProcessedData(processed)
    hosts = d.evaluable("test")
    m = BehavioralPortFree().fit(d)
    e = m.embed(d, "test", hosts)
    assert list(e.index) == hosts and np.isfinite(e.to_numpy()).all() and not np.allclose(e.to_numpy().mean(0), 0, atol=1e-8)
    p = BehavioralPortFreePCA(dim=5, seed=0).fit(d)
    assert p.embed(d, "train").shape[1] == 5 and p.exclude_blocks == ("ports",)
    assert np.array_equal(p.embed(d, "train").to_numpy(), BehavioralPortFreePCA(dim=5, seed=0).fit(d).embed(d, "train").to_numpy())


def test_a_custom_exclusion_can_be_passed_to_the_generic_baseline(processed):
    d = ProcessedData(processed)
    h = BehavioralBaseline(exclude_blocks=("hourly",)).fit(d).embed(d, "train")
    full = BehavioralBaseline().fit(d).embed(d, "train")
    assert h.shape[1] == full.shape[1] - 24 and BehavioralPCA(dim=3, exclude_blocks=("proto",)).fit(d).embed(d, "train").shape[1] == 3
