import pickle
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

torch = pytest.importorskip("torch")
from torch.nn import functional as F

from experiments.dc_sage.data import Preprocessor, merge, summarize, tokens
from experiments.dc_sage.anomalies import combine
from experiments.dc_sage.predict import predict
from methods.graph._dc_sage import Config, DCSAGE, Decoder, LinkSampler, info_nce, norm_adj, prepare_graph, train_epoch


def test_token_rules_keep_anonymous_categories_and_neutral_ties():
    flows = pd.DataFrame({"src": ["a"]*4, "dst": ["b"]*4, "proto": ["6"]*4,
                          "src_port": ["Port001", "443", "53", "2048"],
                          "dst_port": ["Port999", "Port444", "53", "8080"]})
    result = tokens(flows).token.tolist()
    assert result == ["6/a:Port999/client", "6/443/server", "6/other/neutral", "6/other/neutral",
                      "6/a:Port999/server", "6/443/client", "6/other/neutral", "6/other/neutral"]


def test_graph_is_directed_weighted_normalized_and_handles_isolation():
    c = sp.csr_matrix(np.array([[0, 4, 1], [0, 0, 2], [0, 0, 0]], dtype=np.float32))
    out, incoming = prepare_graph(c, np.array([1, 1, 5]), replace(Config(), topk=1))
    assert out[0, 2] > 0 and out[0, 1] == 0
    assert incoming[1, 0] > 0 and incoming[0].nnz == 0
    np.testing.assert_allclose(norm_adj(out).to_dense().sum(1).numpy(), [1, 1, 0])
    np.testing.assert_allclose(norm_adj(out, 1., np.random.default_rng(0)).to_dense().numpy(), 0)


def test_zero_graph_initialization_equals_skip_and_gradients_start_at_gate():
    torch.manual_seed(1)
    x = torch.randn(6, 8)
    model = DCSAGE(8, replace(Config(), hidden=8, dim=4, feature_dropout=0))
    adjacency = norm_adj(sp.eye(6, format="csr", dtype=np.float32))
    z, own, added = model(x, adjacency, adjacency)
    torch.testing.assert_close(z, F.normalize(model.skip(x), dim=-1))
    assert torch.count_nonzero(added) == 0
    (z * torch.randn_like(z)).sum().backward()
    assert model.graph_map.weight.grad.abs().sum() > 0
    assert model.input[0].weight.grad.abs().sum() == 0


def test_false_negative_mask_keeps_positive_diagonal():
    z = F.normalize(torch.randn(5, 4), dim=-1)
    profiles = torch.ones(5, 4)
    value = info_nce(z, z, profiles, Config())
    assert abs(float(value)) < 1e-6
    assert float(info_nce(z, z, profiles, replace(Config(), false_negative_mask=False))) > 0


def test_negative_sampling_excludes_self_and_s2_edges():
    one = sp.csr_matrix(np.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=np.float32))
    two = one + sp.csr_matrix(([1.], ([0], [2])), shape=(4, 4))
    sampler = LinkSampler(one, two, np.arange(4), replace(Config(), link_batch=100))
    positives, negatives = sampler.sample(np.random.default_rng(3))
    assert (positives[:50].numpy() == [0, 2]).all()
    for (source, _), targets in zip(positives.numpy(), negatives.numpy()):
        assert not (targets == source).any()
        assert all(two[source, target] == 0 for target in targets)


def test_preprocessor_chunk_invariance_and_heldout_no_refit():
    rows = []
    for i in range(12):
        for p in (22, 53, 88, 443):
            rows.append((f"h{i}", f"h{(i+1)%12}", "Port123", str(p), "6", i*3+p%10, 0, i+1, 2, 3, 1))
    f = pd.DataFrame(rows, columns=["src", "dst", "src_port", "dst_port", "proto", "ts", "duration", "bytes", "packets", "resp_bytes", "resp_packets"])
    full = summarize(f, (0, 100))
    chunks = merge([summarize(f.iloc[:17], (0, 100)), summarize(f.iloc[17:], (0, 100))])
    for key in full:
        pd.testing.assert_frame_equal(full[key], chunks[key], check_dtype=False)
    prep = Preprocessor(dim=4).fit(full, [f"h{i}" for i in range(12)])
    old = prep.mean.copy(), prep.sd.copy(), prep.idf.copy(), prep.svd.components_.copy()
    late = f.copy()
    late["dst_port"] = "new-service"
    prep.transform(summarize(late, (0, 100)))
    for before, after in zip(old, (prep.mean, prep.sd, prep.idf, prep.svd.components_)):
        np.testing.assert_array_equal(before, after)


def test_full_epoch_updates_graph_without_nonfinite_values():
    torch.set_num_threads(1)
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    counts = sp.csr_matrix(np.ones((20, 20), dtype=np.float32) - np.eye(20, dtype=np.float32))
    # Для негативов оставляем достаточно неположительных пар.
    counts = sp.csr_matrix(np.eye(20, k=1, dtype=np.float32) + np.eye(20, k=-1, dtype=np.float32))
    config = replace(Config(), hidden=8, dim=4, link_batch=16, batch_size=10, feature_dropout=0, edge_dropout=0)
    model, decoder = DCSAGE(6, config), Decoder(4)
    optimizer = torch.optim.AdamW([*model.parameters(), *decoder.parameters()])
    x = torch.randn(20, 6)
    weights = prepare_graph(counts, np.ones(20), config)
    sampler = LinkSampler(counts, counts, np.arange(20), config)
    result = train_epoch(model, decoder, optimizer, x, x, weights, weights, np.arange(20), torch.randn(20, 4), sampler, rng)
    assert np.isfinite(list(result.values())).all()
    assert model.graph_map.weight.abs().sum() > 0


def test_encoder_is_equivariant_to_host_indices():
    torch.manual_seed(11)
    model = DCSAGE(8, replace(Config(), hidden=8, dim=4, feature_dropout=0))
    torch.nn.init.normal_(model.graph_map.weight, std=.1)
    model.eval()
    x = torch.randn(10, 8)
    a = sp.csr_matrix(np.eye(10, k=1, dtype=np.float32))
    b = a.T.tocsr()
    perm = np.random.default_rng(4).permutation(10)
    z = model(x, norm_adj(a), norm_adj(b))[0]
    reordered = model(x[perm], norm_adj(a[perm][:, perm]), norm_adj(b[perm][:, perm]))[0]
    torch.testing.assert_close(reordered, z[perm])


def small_flows():
    rows = []
    for i in range(12):
        for j in range(i+1):
            rows.append((f"h{i}", f"h{(i+j+1)%12}", "Port123", str([22, 53, 88, 443][j%4]), "6", i*3+j, 0, i+j+1, 2, 3, 1))
    return pd.DataFrame(rows, columns=["src", "dst", "src_port", "dst_port", "proto", "ts", "duration", "bytes", "packets", "resp_bytes", "resp_packets"])


def test_injection_subtraction_matches_rebuilding_flows():
    f = small_flows()
    removed = f[f.src == "h1"]
    added = removed.copy()
    added["dst_port"] = "443"
    original = summarize(f, (0, 100))
    actual = combine(original, [(summarize(removed, (0, 100)), -1), (summarize(added, (0, 100)), 1)])
    expected = summarize(pd.concat([f[f.src != "h1"], added]), (0, 100))
    for key in expected:
        pd.testing.assert_frame_equal(actual[key], expected[key], check_dtype=False)


def test_predict_replays_frozen_preprocessing_and_weights(tmp_path):
    from methods.graph._dc_sage import infer
    from evaluation.io import load_embeddings
    torch.manual_seed(5)
    flows = small_flows()
    stats = summarize(flows, (0, 100))
    prep = Preprocessor(dim=4).fit(stats, [f"h{i}" for i in range(12)])
    hosts, x, p, c = prep.transform(stats)
    config = replace(Config(), hidden=8, dim=4)
    model, decoder = DCSAGE(x.shape[1], config), Decoder(4)
    torch.nn.init.normal_(model.graph_map.weight, std=.1)
    expected = infer(model, x, prepare_graph(c, prep.idf_neighbors(hosts), config))[0]
    torch.save({"inputs": x.shape[1], "config": config.dictionary(), "model": model.state_dict(), "decoder": decoder.state_dict()}, tmp_path / "best.pt")
    with (tmp_path / "preprocessor.pkl").open("wb") as stream:
        pickle.dump(prep, stream)
    flows.to_csv(tmp_path / "flows.csv", index=False)
    predict(tmp_path, tmp_path / "preprocessor.pkl", tmp_path / "flows.csv", (0, 100), tmp_path / "embedding.csv")
    actual = load_embeddings(tmp_path / "embedding.csv").loc[hosts].to_numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)
    pd.DataFrame({"host_id": ["absent", *hosts]}).to_csv(tmp_path / "cohort.csv", index=False)
    predict(tmp_path, tmp_path / "preprocessor.pkl", tmp_path / "flows.csv", (0, 100), tmp_path / "cohort_embedding.csv", tmp_path / "cohort.csv")
    with_empty = load_embeddings(tmp_path / "cohort_embedding.csv")
    assert "absent" in with_empty.index
    assert np.isfinite(with_empty.to_numpy()).all()
    np.testing.assert_allclose(with_empty.loc[hosts].to_numpy(), expected, atol=1e-6)
