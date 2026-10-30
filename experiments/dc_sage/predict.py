"""Применить замороженную модель к новому CSV NetFlow в канонической схеме проекта."""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.dc_sage.data import merge, summarize, universe
from experiments.dc_sage.run import frozen_model, write_embeddings
from methods.graph._dc_sage import infer, prepare_graph
from pipeline.windows import window_mask


def predict(model_dir, preprocessor, flows, bounds, output, cohort=None):
    with Path(preprocessor).open("rb") as stream:
        prep = pickle.load(stream)
    model, _ = frozen_model(Path(model_dir))
    pieces = []
    for chunk in pd.read_csv(flows, dtype={c: str for c in ("src", "dst", "src_port", "dst_port", "proto")}, chunksize=400_000):
        for col, default in (("src_port", "unknown"), ("duration", 0), ("bytes", 0), ("packets", 0), ("resp_bytes", 0), ("resp_packets", 0)):
            if col not in chunk:
                chunk[col] = default
        chunk = chunk.loc[window_mask(chunk.ts.to_numpy(), chunk.duration.to_numpy(), bounds, True)]
        if len(chunk):
            pieces.append(summarize(chunk, bounds))
        if len(pieces) > 8:
            pieces = [merge(pieces)]
    if not pieces:
        raise ValueError("В заданном окне нет потоков")
    stats = merge(pieces)
    requested = None if cohort is None else pd.Index(pd.read_csv(cohort, dtype=str).host_id)
    hosts, x, _, counts = prep.transform(stats, universe(stats, () if requested is None else requested))
    z, own, graph = infer(model, x, prepare_graph(counts, prep.idf_neighbors(hosts), model.config))
    requested = hosts if requested is None else requested
    indices = hosts.get_indexer(requested)
    write_embeddings(Path(output), z[indices], requested)
    np.savez_compressed(Path(output).with_suffix(".parts.npz"), own=own[indices], graph=graph[indices],
                        hosts=requested.to_numpy(str), empty=x[indices, -1])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-dir", type=Path, required=True)
    ap.add_argument("--preprocessor", type=Path, required=True)
    ap.add_argument("--flows", type=Path, required=True)
    ap.add_argument("--window", required=True, help="start:end в секундах Time")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--cohort", type=Path)
    args = ap.parse_args()
    torch.set_num_threads(4)
    predict(args.model_dir, args.preprocessor, args.flows, tuple(map(float, args.window.split(":"))), args.output, args.cohort)


if __name__ == "__main__":
    main()
