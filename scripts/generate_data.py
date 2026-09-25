#!/usr/bin/env python3
"""Обёртка для генерации синтетических данных."""
import argparse
import json

from hostembed.config import RANDOM_SEED
from hostembed.datagen import generate_dataset

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Генерация netflow-данных в формате LANL.")
    p.add_argument("--out", default="data")
    p.add_argument("--windows", type=int, default=5)
    p.add_argument("--seed", type=int, default=RANDOM_SEED)
    p.add_argument("--no-anomalies", action="store_true")
    args = p.parse_args()
    meta = generate_dataset(args.out, args.windows, args.seed, not args.no_anomalies)
    print(f"Сгенерировано окон: {meta['n_windows']}, хостов: {meta['n_internal_hosts']}")
    print(json.dumps(meta["injected_anomalies"], ensure_ascii=False, indent=2))
