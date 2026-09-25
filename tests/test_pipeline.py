"""Smoke-тесты сквозного пайплайна на маленьком датасете."""
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hostembed.anomaly import detect_anomalies
from hostembed.clustering import Explainer, Similarity, cluster, select_k
from hostembed.config import PipelineConfig
from hostembed.datagen import generate_dataset
from hostembed.embeddings import FeatureEmbedder, combined_embedding, graph_embedding
from hostembed.features import extract_features
from hostembed import lanl_io, pipeline


def _prep(tmp_path):
    data = str(tmp_path / "data")
    generate_dataset(data, n_windows=4, seed=1)
    return data


def test_datagen_schema(tmp_path):
    data = _prep(tmp_path)
    windows = lanl_io.discover_windows(data)
    assert len(windows) == 4
    df = lanl_io.load_window(windows[0])
    assert {"src_device", "dst_device", "dst_port", "protocol"}.issubset(df.columns)
    assert len(df) > 100
    assert lanl_io.load_labels(data) is not None


def test_features_matrix(tmp_path):
    data = _prep(tmp_path)
    df = lanl_io.load_window(lanl_io.discover_windows(data)[0])
    feats = extract_features(df)
    assert feats.shape[0] > 50
    assert "in_peers" in feats.columns and "out_peers" in feats.columns
    assert not feats.isna().any().any()


def test_embeddings_and_clustering(tmp_path):
    data = _prep(tmp_path)
    flows = lanl_io.load_window(lanl_io.discover_windows(data)[-1])
    feats = extract_features(flows)
    fe = FeatureEmbedder(n_components=8, seed=1)
    femb = fe.fit_transform(feats)
    assert femb.shape[0] == feats.shape[0]
    gemb = graph_embedding(flows, list(feats.index), dim=8, seed=1)
    assert gemb.shape[0] == feats.shape[0]
    comb = combined_embedding(femb, gemb)
    assert comb.shape[0] == feats.shape[0]

    k, scores = select_k(femb.values, 3, 10, seed=1)
    labels, _ = cluster(femb.values, k, seed=1)
    assert len(set(labels)) == k

    sim = Similarity(femb)
    dev = feats.index[0]
    nbrs = sim.neighbors(dev, 5)
    assert len(nbrs) == 5 and all(0 <= s <= 1.001 for _, s in nbrs)

    expl = Explainer(feats, flows)
    reasons = expl.explain(dev, nbrs[0][0])
    assert isinstance(reasons, list) and reasons


def test_anomaly_detection_finds_injected(tmp_path):
    data = _prep(tmp_path)
    windows = lanl_io.discover_windows(data)
    window_feats = [extract_features(lanl_io.load_window(w)) for w in windows]
    res = detect_anomalies(window_feats, 3, 10, z_threshold=2.5, top_n=10, seed=1)
    scores = res["scores"]
    assert not scores.empty
    truth = set(lanl_io.load_anomaly_truth(data)["device"])
    top10 = set(scores.head(10)["device"])
    # Хотя бы одна внедрённая аномалия должна попасть в топ.
    assert truth & top10


def test_full_pipeline(tmp_path):
    data = _prep(tmp_path)
    art = str(tmp_path / "artifacts")
    cfg = PipelineConfig(data_dir=data, artifacts_dir=art, pca_dim=8, embed_dim=8)
    summary = pipeline.run(cfg)
    assert summary["n_devices"] > 50
    assert summary["best_method"] in cfg.methods
    for f in ("summary.json", "hosts.json", "clusters.json", "anomalies.json"):
        assert os.path.exists(os.path.join(art, f))
