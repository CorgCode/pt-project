"""Доступ к подготовленному датасету с явной границей «обучение / отложенные окна»."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.windows import ORDER

FLOW_DTYPES = {"src": "string", "dst": "string", "dst_port": "string", "proto": "string",
               "src_port": "string"}


class ProcessedData:
    def __init__(self, root: Path):
        self.root = Path(root)
        mpath = self.root / "manifest.json"
        if not mpath.exists():
            raise FileNotFoundError(f"{mpath}: это не подготовленный датасет (запустите python -m pipeline.prepare run)")
        self.manifest = json.loads(mpath.read_text(encoding="utf-8"))
        table = pd.read_csv(self.root / "cohort.csv", dtype={"host_id": "string"})
        self.cohort = table["host_id"].astype(str).tolist()
        self._evaluable = {c[len("evaluable_"):]: table.loc[table[c].astype(str).str.lower().eq("true"), "host_id"]
                           .astype(str).tolist() for c in table.columns if c.startswith("evaluable_")}

    @property
    def synthetic(self) -> bool:
        return bool(self.manifest.get("synthetic"))

    @property
    def kind(self) -> str:
        """synthetic_fixture | unverified | lanl_declared (см. pipeline.provenance)."""
        return self.manifest.get("data_kind", "unverified")

    @property
    def cohort_policy(self) -> str:
        return self.manifest["cohort"].get("policy", "all_windows")

    def evaluable(self, window: str) -> list[str]:
        """Хосты когорты, активные в окне (для train — вся когорта)."""
        if window == "train":
            return list(self.cohort)
        if window not in self._evaluable:
            raise KeyError(f"неизвестное окно {window!r}")
        return list(self._evaluable[window])

    @property
    def windows(self) -> list[str]:
        """Окна по времени (train, validation, test)."""
        return [w for w in ORDER if w in self.manifest["windows"]]

    def flows(self, window: str) -> pd.DataFrame:
        """Потоки одного окна (для обучения используйте training_flows)."""
        if window not in self.manifest["windows"]:
            raise KeyError(f"неизвестное окно {window!r}; есть {self.windows}")
        return pd.read_csv(self.root / "flows" / f"{window}.csv", dtype=FLOW_DTYPES)

    def graph(self, window: str) -> pd.DataFrame:
        return pd.read_csv(self.root / "graph" / f"edges_{window}.csv",
                           dtype={"host_a": "string", "host_b": "string"})

    # --- вход моделей: ТОЛЬКО train-окно -------------------------------------------------
    def training_flows(self) -> pd.DataFrame:
        return self.flows("train")

    def training_graph(self) -> pd.DataFrame:
        return self.graph("train")
