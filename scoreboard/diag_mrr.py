"""Проверка для анализа качества по метрике MRR на validation: какой блок (и комбинация) дает метрику.

    python scoreboard/diag_mrr.py --gnn <gnn submission> --params <params.json>

Считает evaluation.partners.partner_ranking, вид «new» на эмбеддингах окна validation
для каждого блока отдельно и для комбинаций.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from gnn_combine_v3 import Data, ORDER, RUS  # noqa: E402
from evaluation.partners import load_partners, partner_ranking  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=str(ROOT / "runs" / "blocks"))
    ap.add_argument("--gnn", required=True)
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--params", required=True)
    ap.add_argument("--window", default="validation")
    a = ap.parse_args()
    t0 = time.time()
    D = Data(Path(a.blocks), Path(a.gnn), Path(a.processed))
    p = json.loads(Path(a.params).read_text(encoding="utf-8"))
    p = {k: p[k] for k in ("w_peers", "w_services", "w_gnn", "k_peers", "k_services")}
    w = a.window
    parts = D.parts(w, p)                                    # блоки с итоговыми весами
    hosts = D.hosts[D.ev[w]]
    partners = load_partners(ROOT / "scoreboard" / "data" / "reference" / w / "partners.csv")
    variants = [("только поведение", ["behav"]), ("только собеседники", ["peers"]), ("только сервисы", ["services"]),
                ("только GNN", ["gnn"]), ("собеседники + сервисы", ["peers", "services"]),
                ("поведение + GNN", ["behav", "gnn"]), ("собеседники + сервисы + GNN", ["peers", "services", "gnn"]),
                ("все блоки (итог)", ORDER)]
    print(f"\nMRR новых собеседников на {w} (чем больше, тем лучше), хостов {len(hosts):,}")
    print(f"{'набор блоков':32s}{'MRR':>8s}{'Recall@10':>11s}")
    base = None
    for name, keys in variants:
        z = np.hstack([parts[k] for k in keys if k in parts])
        r = partner_ranking(pd.DataFrame(z, index=hosts), partners)["new"]
        e = r["embedding"]
        print(f"{name:32s}{e['mrr']:8.3f}{e['recall_at_n']:11.3f}", flush=True)
        base = r
    print(f"Без эмбеддинга: по истории собеседников {base['history']['mrr']:.3f}, по популярности {base['popularity']['mrr']:.3f}")
    print(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
