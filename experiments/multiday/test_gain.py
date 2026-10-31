from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from experiments.multiday.gain import check_chronology, combine_views


def day(start, end):
    return SimpleNamespace(manifest={"windows": {"train": {"start": start, "end": end}}})


def test_evaluation_exports_required_host_header_for_models_with_unnamed_index(tmp_path,monkeypatch):
    import json
    import numpy as np
    import pandas as pd
    from experiments.multiday import gain as module
    from evaluation.io import load_embeddings
    class Model:
        def embed(self,data,window,hosts):
            return pd.DataFrame([[0.6,0.8],[0.8,0.6]],index=hosts,columns=['emb_0','emb_1'])
    def score(path,*args,**kwargs):
        frame=load_embeddings(path,'host_id','emb_')
        assert frame.index.tolist()==['a','b']
        np.testing.assert_allclose(frame.to_numpy(),[[0.6,0.8],[0.8,0.6]])
        return {'method':'test','summary':[{'metric':'m','title':'m','group':'g','chance':0}], 'metrics':{'m':0.8}}
    monkeypatch.setattr(module,'run',score)
    monkeypatch.setattr(module.submit,'add_identity',lambda *args,**kwargs:None)
    ref=tmp_path/'ref';ref.mkdir();(ref/'manifest.json').write_text(json.dumps({}))
    data=SimpleNamespace(cohort=['a','b'],evaluable=lambda window:['a','b'])
    result,_=module.evaluate(Model(),data,'validation',ref,tmp_path/'out','test')
    assert result==pytest.approx(80)
    assert load_embeddings(tmp_path/'out/later.csv').index.tolist()==['a','b']


def test_rejects_training_after_validation_or_overlapping_days():
    with pytest.raises(ValueError, match="предшествовать"):
        check_chronology([day(0, 20), day(30, 50)], 40)
    with pytest.raises(ValueError, match="пересекаются"):
        check_chronology([day(0, 20), day(10, 30)], 40)
    with pytest.raises(ValueError, match="одинаковой"):
        check_chronology([day(0, 20), day(30, 60)], 70)
    check_chronology([day(0, 20), day(30, 50)], 50)


def test_multiday_features_preserve_counts_and_average_rates():
    idx = pd.MultiIndex.from_tuples([("a", "peer")], names=["h", "t"])
    def view(count, rate):
        return {"counts": {k: pd.Series([count], index=idx) for k in ("peers", "services", "context")},
                "activity": np.log1p([[rate]]), "flows": np.array([count])}
    combined = combine_views([view(3, 2), view(7, 8)])
    assert combined["counts"]["peers"].iloc[0] == 10
    np.testing.assert_allclose(np.expm1(combined["activity"]), [[5]])
    np.testing.assert_array_equal(combined["flows"], [10])


def test_checkpoint_keeps_local_torch_encoder_predictions():
    import cloudpickle
    import torch
    from experiments.overnight.model import T
    net = T._encoder(4).eval()
    restored = cloudpickle.loads(cloudpickle.dumps(net))
    x = torch.ones(3, 4)
    with torch.no_grad():
        torch.testing.assert_close(net(x), restored(x))


def test_directed_sage_starts_from_own_profile_and_trains_with_sparse_graph():
    import scipy.sparse as sp
    import torch
    from torch.nn import functional as F
    from experiments.multiday.dc_core import Config, DCSAGE, norm_adj, prepare_graph
    graph = sp.csr_matrix(([3., 4., 2.], ([0, 1, 2], [1, 2, 0])), shape=(3, 3))
    net = DCSAGE(4, Config(dim=8, hidden=8, feature_dropout=0))
    weights = prepare_graph(graph, np.ones(3), net.config)
    adjacency = [norm_adj(w) for w in weights]
    x = torch.randn(3, 4)
    z, own, neighbors = net(x, *adjacency)
    torch.testing.assert_close(z, F.normalize(own, dim=-1))
    torch.testing.assert_close(neighbors, torch.zeros_like(neighbors))
    z.square().sum().backward()
    assert net.graph_map.weight.grad is not None
    assert torch.isfinite(net.graph_map.weight.grad).all()


def test_directed_counts_do_not_turn_incoming_tokens_into_outgoing_edges():
    from experiments.multiday.dc import counts_from_view
    index = pd.MultiIndex.from_tuples([("a", "o:b"), ("b", "i:a"), ("a", "o:external")], names=["h", "t"])
    view = {"counts": {"peers": pd.Series([3, 3, 5], index=index)}}
    counts = counts_from_view(view, ["a", "b"], pd.Index(["o:b", "i:a", "o:external"]))
    np.testing.assert_array_equal(counts.toarray(), [[0, 3], [0, 0]])


def test_posterior_depends_on_features_and_is_equivariant_to_row_order():
    import torch
    from experiments.multiday.gain import MultiDayEncoder
    from experiments.overnight.model import T
    model = MultiDayEncoder()
    model.net = T._encoder(4)
    model.classifier = torch.randn(5, 128)
    model.class_projection = np.random.default_rng(0).standard_normal((5, 128)).astype(np.float32)
    model.posterior_mix = 1.0
    x = np.random.default_rng(1).standard_normal((7, 4)).astype(np.float32)
    order = np.array([6, 1, 0, 4, 3, 5, 2])
    np.testing.assert_allclose(model._neural(x)[order], model._neural(x[order]), atol=1e-6)


def test_prepare_new_day_keeps_old_cohort_and_excludes_straddling_flows(tmp_path):
    import json
    from experiments.multiday.prepare import prepare_day
    from pipeline.lanl_netflow import RAW_COLUMNS
    base = tmp_path / "base"
    base.mkdir()
    pd.DataFrame({"host_id": ["a", "b"], "events_train": [1000, 1000]}).to_csv(base / "cohort.csv", index=False)
    (base / "manifest.json").write_text(json.dumps({"config": {}}))
    offset = 3 * 86400
    def row(t, duration=0, src="a"):
        return [t + offset, duration, src, "b", "6", "40000", "443", 1, 1, 100, 50]
    # Время не отсортировано: ридер должен найти и более ранние строки после поздних.
    raw = pd.DataFrame([row(126000), row(129599, 1), row(129600), row(131400, src="future"),
                        row(140000), row(129599)], columns=RAW_COLUMNS)
    path = tmp_path / "netflow_day-05.bz2"
    raw.to_csv(path, header=False, index=False)
    out = tmp_path / "out"
    manifest = prepare_day(path, base, out, 5, chunksize=2, data_kind="synthetic_fixture")
    assert manifest["synthetic"]
    assert manifest["windows"]["train"]["n_flows"] == 2
    assert set(pd.read_csv(out / "cohort.csv").host_id) == {"a", "b"}
    assert len(pd.read_csv(out / "flows/test.csv")) == 1


def test_roles_recognize_server_on_both_source_and_destination_ports():
    from experiments.multiday.roles import service_roles
    flows = pd.DataFrame({"src": ["a", "b", "a"], "dst": ["b", "a", "b"],
                          "src_port": ["Port1", "443", "80"], "dst_port": ["443", "Port2", "80"],
                          "proto": ["6", "6", "6"]})
    counts = service_roles(flows, ["a", "b"])
    assert counts.loc[("a", "6/443/client")] == 2
    assert counts.loc[("b", "6/443/server")] == 2
    assert counts.loc[("a", "6/other/neutral")] == 1


def test_multiday_role_encoder_fits_only_training_and_embeds_in_shared_coordinates(tmp_path):
    from experiments.multiday.roles import RoleEncoder
    hosts = [f"h{i}" for i in range(24)]
    rng = np.random.default_rng(5)

    class TrainingDay:
        def __init__(self, name, start):
            self.root, self.cohort = tmp_path / name, list(hosts)
            self.manifest = {"windows": {"train": {"start": start, "end": start + 3600}}}
            self.frame = pd.DataFrame({"src": rng.choice(hosts, 1200), "dst": rng.choice(hosts, 1200),
                                       "proto": rng.choice(["6", "17"], 1200),
                                       "src_port": rng.choice(["Port1", "443", "53"], 1200),
                                       "dst_port": rng.choice(["443", "53", "Port2"], 1200),
                                       "ts": start + rng.integers(0, 3598, 1200), "duration": 0,
                                       "bytes": rng.integers(100, 10000, 1200), "resp_bytes": 100,
                                       "packets": 4, "resp_packets": 2})

        def training_flows(self):
            return self.frame.copy()

        def flows(self, window):
            raise AssertionError("fit читает только training_flows")

    days = [TrainingDay(f"day{i}", i * 86400) for i in range(6)]
    first = days[0]
    model = RoleEncoder().fit_days(days)
    model.train_head(first, "cca", 0)
    model.fit_pca(first)
    embeddings = model.embed(first, "train", hosts)
    assert list(embeddings.index) == hosts
    assert embeddings.shape[0] == 24
    assert np.isfinite(embeddings.to_numpy()).all()
    class EvaluationDay:
        def __init__(self, name):
            self.root, self.calls = tmp_path / name, 0
            self.manifest = {"windows": {"validation": {"start": 6 * 86400, "end": 6 * 86400 + 3600}}}

        def flows(self, window):
            self.calls += 1
            frame = first.frame.copy()
            frame.ts += 6 * 86400
            return frame

    observation = EvaluationDay("evaluation")
    result = model.embed(observation, "validation", hosts)
    assert observation.calls == 2
    model.train_head(first, "cca", 0)
    model.fit_pca(first)
    np.testing.assert_allclose(model.embed(observation, "validation", hosts), result, atol=1e-6)
    assert observation.calls == 2  # новая голова не перечитывает и не переобучает признаки validation
    another = EvaluationDay("different_data")
    model.embed(another, "validation", hosts)
    assert another.calls == 2
    import cloudpickle
    from experiments.multiday.gain import restore_training_features
    checkpoint = tmp_path / "features.pkl"
    checkpoint.write_bytes(cloudpickle.dumps(model))
    restored = restore_training_features(checkpoint, first, days)
    assert restored.pca is None and restored._parts_cache == {} and restored._evaluation_cache == {}
    assert not hasattr(restored, "proj")
    np.testing.assert_array_equal(restored._xs[0], model._xs[0])
    with pytest.raises(ValueError, match="число обучающих дней"):
        restore_training_features(checkpoint, first, days[:-1])


def test_streaming_cca_matches_concatenated_covariances():
    import itertools
    from experiments.multiday.gain import streaming_cca
    from experiments.overnight.model import T
    rng = np.random.default_rng(3)
    xs = [rng.standard_normal((30, 6)) for _ in range(4)]
    ok = rng.random((4, 30)) > 0.2
    reference = T.TemporalEncoder(encoder="cca", splits=4)
    reference._fit_cca(xs, ok)
    proj, corr, mean = streaming_cca(xs, ok)
    a = [x for i, j in itertools.combinations(range(4), 2)
         for x in [xs[i][ok[i] & ok[j]], xs[j][ok[i] & ok[j]]]]
    np.testing.assert_allclose(mean, np.vstack(a).mean(0), atol=1e-12)
    np.testing.assert_allclose(corr, reference.corr, atol=1e-12)
    np.testing.assert_allclose((proj * corr) @ (proj * corr).T,
                               (reference.proj * reference.corr) @ (reference.proj * reference.corr).T,
                               atol=1e-10)


def test_frozen_validation_winner_cannot_change_after_test_phase_starts(tmp_path):
    import json
    from experiments.multiday.finish import freeze_selection
    candidates = [tmp_path / "one", tmp_path / "two"]
    for directory, score in zip(candidates, [60, 70]):
        directory.mkdir()
        (directory / "best.json").write_text(json.dumps({"name": directory.name, "score": score}))
        (directory / "best.pkl").write_bytes(directory.name.encode())
    output = tmp_path / "frozen"
    first = freeze_selection(candidates, output)
    assert first["winner"]["name"] == "two"
    (candidates[0] / "best.json").write_text(json.dumps({"name": "one", "score": 999}))
    second = freeze_selection(candidates, output)
    assert second == first
    assert (output / "model.pkl").read_bytes() == b"two"


def test_pairing_new_validation_does_not_select_hosts_using_future_activity(tmp_path):
    import json
    from experiments.multiday.reference import pair_windows
    base, target, output = tmp_path / "base", tmp_path / "later", tmp_path / "paired"
    for directory in [base, target]:
        (directory / "flows").mkdir(parents=True)
        (directory / "graph").mkdir()
    pd.DataFrame({"host_id": ["a", "b"], "events_train": [500, 600]}).to_csv(base / "cohort.csv", index=False)
    pd.DataFrame({"host_id": ["a", "b", "future"], "events_test": [400, 0, 900],
                  "evaluable_test": [True, False, True]}).to_csv(target / "cohort.csv", index=False)
    (base / "manifest.json").write_text(json.dumps({"windows": {"train": {"start": 0, "end": 10}}, "config": {}}))
    (target / "manifest.json").write_text(json.dumps({"windows": {"test": {"start": 100, "end": 110}}}))
    (base / "flows/train.csv").write_text("source\n")
    (base / "graph/edges_train.csv").write_text("edges\n")
    (target / "flows/test.csv").write_text("target\n")
    pair_windows(base, target, output, "validation")
    cohort = pd.read_csv(output / "cohort.csv").set_index("host_id")
    assert list(cohort.index) == ["a", "b"]
    assert cohort.loc["b", "events_train"] == 600
    assert not cohort.loc["b", "evaluable_validation"]
