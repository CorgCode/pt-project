import numpy as np
import pandas as pd
import pytest

from host_profile import HostProfileEmbedder, make_synthetic
from methods.aggregated_host_profile import AggregatedHostProfile, aggregate


@pytest.mark.parametrize("weighting,direction,center,struct_weight", [
    ("raw", False, False, 0), ("raw", True, False, 0),
    ("log", True, False, 0), ("tfidf", False, False, 0),
    ("tfidf", True, False, 0), ("tfidf", True, True, 0),
    ("tfidf", True, True, .5),
])
def test_streaming_matches_original(weighting, direction, center, struct_weight):
    flows, _ = make_synthetic(t_end=5400, rate_scale=.01)
    kw = dict(dim=16, weighting=weighting, direction=direction, center=center,
              struct_weight=struct_weight, seed=0, min_flows=1)
    train, test = flows[flows.time < 3600], flows[flows.time >= 3600]
    original = HostProfileEmbedder(**kw).fit(train)
    def counts(f):
        parts = [aggregate(p.rename(columns={"time": "ts", "sport": "src_port", "dport": "dst_port"}))
                 for p in (f.iloc[idx] for idx in np.array_split(np.arange(len(f)), 3))]
        return pd.concat(parts).groupby(["host", "key", "w"], as_index=False)["n"].sum()
    model = AggregatedHostProfile(**kw).fit_profile(counts(train), original.hosts_, train[["src", "dst"]].drop_duplicates())
    np.testing.assert_allclose(original.idf_, model.idf_)
    for part in (train, test):
        expected, _ = original.transform(part)
        actual = model.transform_profile(counts(part), part[["src", "dst"]].drop_duplicates())
        np.testing.assert_allclose(actual.to_numpy(), expected, atol=1e-9)


def test_unseen_tokens_and_hosts_do_not_change_fitted_model():
    flows, _ = make_synthetic(t_end=3600, rate_scale=.01)
    original = HostProfileEmbedder(dim=16, min_flows=1).fit(flows)
    canonical = flows.rename(columns={"time": "ts", "sport": "src_port", "dport": "dst_port"})
    model = AggregatedHostProfile(dim=16).fit_profile(aggregate(canonical), original.hosts_)
    heldout = pd.DataFrame({"time": [4000., 4001.], "src": [original.hosts_[0], "new-host"],
                            "dst": ["new-server", "new-server"], "proto": [254, 254],
                            "sport": [65535, 65535], "dport": [65535, 65535]})
    before = model.idf_.copy(), model.svd_.components_.copy(), model.mean_.copy()
    counts = aggregate(heldout.rename(columns={"time": "ts", "sport": "src_port", "dport": "dst_port"}))
    actual = model.transform_profile(counts)
    expected, active = original.transform(heldout)
    assert not active.any()
    np.testing.assert_array_equal(actual.to_numpy(), expected)
    for fitted, saved in zip((model.idf_, model.svd_.components_, model.mean_), before):
        np.testing.assert_array_equal(fitted, saved)
