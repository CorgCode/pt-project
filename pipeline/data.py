"""Access to a processed dataset with the training/held-out boundary made explicit."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

FLOW_DTYPES = {"src": "string", "dst": "string", "dst_port": "string", "proto": "string",
               "src_port": "string"}


class ProcessedData:
    def __init__(self, root: Path):
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"{mpath}: not a processed dataset (run `python -m pipeline.prepare run`)")
        self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        self.cohort = pd.read_csv(self.root / "cohort.csv", dtype={"host_id": "string"})["host_id"].astype(str).tolist()

    @property
    def synthetic(self) -> bool:
        return bool(self.manifest.get("synthetic"))

    @property
    def windows(self) -> list[str]:
        return list(self.manifest["windows"])

    def flows(self, window: str) -> pd.DataFrame:
        """Canonical flows of one window (any window; prefer training_flows for model fitting)."""
        if window not in self.manifest["windows"]:
            raise KeyError(f"unknown window {window!r}; have {self.windows}")
        return pd.read_csv(self.root / "flows" / f"{window}.csv", dtype=FLOW_DTYPES)

    def graph(self, window: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "graph" / f"edges_{window}.csv",
                           dtype={"host_a": "string", "host_b": "string"})

    # --- model inputs: ONLY the training window -------------------------------------------
    def training_flows(self) -> pd.DataFrame:
        return self.flows("train")

    def training_graph(self) -> pd.DataFrame:
        return self.graph("train")
