"""Применение сохранённого гибрида к двум CSV векторов нового окна."""
import argparse
from pathlib import Path

import joblib

from evaluation.io import load_embeddings
from evaluation.validate import validate_submission


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--leader", type=Path, required=True)
    ap.add_argument("--graph", type=Path, help="обязателен только при положительном весе")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    model = joblib.load(args.model)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    graph = None if args.graph is None else load_embeddings(args.graph)
    model.transform(load_embeddings(args.leader), graph).to_csv(args.output)
    validate_submission(str(args.output))


if __name__ == "__main__":
    main()
