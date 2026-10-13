"""Port-free behavioural features: ports are removed entirely, nothing else changes (issue #43)."""
import numpy as np
import pandas as pd
import pytest

from evaluation.knn_consistency import BLOCKS, build_reference_features
from evaluation.temporal_reference import load_flows


@pytest.fixture(scope="module")
def flows(processed):
    return {w: load_flows(processed / "flows" / f"{w}.csv") for w in ("train", "test")}


@pytest.fixture(scope="module")
def hosts(processed):
    from pipeline.data import ProcessedData
    return ProcessedData(processed).evaluable("test")


def shuffled_ports(df, seed=0):
    out = df.copy()
    out["dst_port"] = np.random.default_rng(seed).permutation(out["dst_port"].to_numpy())
    return out


def test_no_port_columns_and_all_other_columns_identical(flows, hosts):
    full = build_reference_features(flows["train"], hosts=hosts)
    free = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=("ports",))
    assert any(c.startswith("ports_") for c in full.columns) and not any(c.startswith("ports_") for c in free.columns)
    pd.testing.assert_frame_equal(free, full[[c for c in full.columns if not c.startswith("ports_")]])


def test_port_free_features_do_not_depend_on_port_values_at_all(flows, hosts):
    base = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=("ports",))
    shuf = build_reference_features(shuffled_ports(flows["train"]), hosts=hosts, exclude_blocks=("ports",))
    pd.testing.assert_frame_equal(base, shuf)                                     # bit for bit
    full = build_reference_features(flows["train"], hosts=hosts)
    full_shuf = build_reference_features(shuffled_ports(flows["train"]), hosts=hosts)
    assert not np.allclose(full.filter(like="ports_").to_numpy(), full_shuf.filter(like="ports_").to_numpy())
    pd.testing.assert_frame_equal(full.drop(columns=full.filter(like="ports_").columns),
                                  full_shuf.drop(columns=full_shuf.filter(like="ports_").columns))   # only ports changed


def test_missing_or_garbage_port_column_is_irrelevant_when_excluded(flows, hosts):
    tr = flows["train"].copy()
    tr["dst_port"] = "x"                                                          # a constant, meaningless port field
    a = build_reference_features(tr, hosts=hosts, exclude_blocks=("ports",))
    b = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=("ports",))
    pd.testing.assert_frame_equal(a, b)


def test_fitted_state_is_reused_on_later_windows_and_remembers_the_exclusion(flows, hosts):
    X, fit = build_reference_features(flows["train"], hosts=hosts, return_fitted=True, exclude_blocks=("ports",))
    assert fit["excluded"] == ("ports",) and "ports" not in fit["scalers"] and fit["ports"] == []
    Xt = build_reference_features(flows["test"], hosts=hosts, fitted=fit, exclude_blocks=("ports",))
    assert list(Xt.columns) == list(X.columns) and np.isfinite(Xt.to_numpy()).all()
    assert not np.allclose(Xt.mean(axis=0), 0, atol=1e-8)                              # not re-standardised on the later window
    with pytest.raises(ValueError, match="excluded blocks"):
        build_reference_features(flows["test"], hosts=hosts, fitted=fit)           # applied without the exclusion
    _, full_fit = build_reference_features(flows["train"], hosts=hosts, return_fitted=True)
    with pytest.raises(ValueError, match="excluded blocks"):
        build_reference_features(flows["test"], hosts=hosts, fitted=full_fit, exclude_blocks=("ports",))


def test_validation_and_normalisation_of_the_exclusion(flows, hosts):
    with pytest.raises(ValueError, match="unknown block"):
        build_reference_features(flows["train"], hosts=hosts, exclude_blocks=("port",))
    with pytest.raises(ValueError, match="every block"):
        build_reference_features(flows["train"], hosts=hosts, exclude_blocks=BLOCKS)
    a = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=("hourly", "ports", "ports"))
    b = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=("ports", "hourly"))
    pd.testing.assert_frame_equal(a, b)


@pytest.mark.parametrize("block", ["volume", "peers", "ratio", "proto", "hourly"])
def test_any_block_can_be_dropped_leaving_the_rest_identical(flows, hosts, block):
    full = build_reference_features(flows["train"], hosts=hosts)
    cut = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=(block,))
    assert not any(c.startswith(f"{block}_") for c in cut.columns)
    pd.testing.assert_frame_equal(cut, full[[c for c in full.columns if not c.startswith(f"{block}_")]])


def test_default_behaviour_is_unchanged(flows, hosts):
    a = build_reference_features(flows["train"], hosts=hosts)
    b = build_reference_features(flows["train"], hosts=hosts, exclude_blocks=())
    pd.testing.assert_frame_equal(a, b)
