#!/usr/bin/env python3
"""Обёртка для запуска пайплайна."""
import argparse
import json

from hostembed.config import PipelineConfig
from hostembed.pipeline import run

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Пайплайн host2vec: признаки -> embeddings -> кластеры -> аномалии.")
    p.add_argument("--data", default="data")
    p.add_argument("--artifacts", default="artifacts")
    args = p.parse_args()
    summary = run(PipelineConfig(data_dir=args.data, artifacts_dir=args.artifacts))
    print(json.dumps(summary, ensure_ascii=False, indent=2))
