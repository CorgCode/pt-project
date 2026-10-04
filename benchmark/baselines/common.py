"""Shared IO for baseline embeddings; conventions come from benchmark/config.json."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.evaluate import _load_matrix  # same loader the evaluator uses

CONFIG_PATH = Path(__file__).resolve().parents[1] / "config.json"
REF_PREFIX = "ref_"


def load_config(path: Path = CONFIG_PATH) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_reference_features(reference_dir: Path, config: dict | None = None) -> pd.DataFrame:
    """Hosts x ref_* matrix indexed by host_id (validated like the evaluator does)."""
    config = config or load_config()
    return _load_matrix(Path(reference_dir) / "features.csv", config["host_id_column"], REF_PREFIX)


def write_embeddings(path: Path, hosts: pd.Index, X: np.ndarray, config: dict | None = None) -> Path:
    """Write the common submission format: host_id, emb_0, emb_1, ..."""
    config = config or load_config()
    if X.shape[0] != len(hosts):
        raise ValueError("embedding rows do not match hosts")
    if not np.isfinite(X).all():
        raise ValueError("embedding contains NaN/Inf")
    cols = [f"{config['embedding_prefix']}{i}" for i in range(X.shape[1])]
    df = pd.DataFrame(X, columns=cols)
    df.insert(0, config["host_id_column"], list(hosts))
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
