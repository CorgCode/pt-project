"""Generate baseline embeddings from reference/features.csv and (optionally) evaluate them.

    python -m benchmark.baselines.run_all --reference-dir reference --dim 16 --seed 0 --evaluate
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmark.baselines.common import load_config, load_reference_features, write_embeddings
from benchmark.baselines.methods import METHODS


def generate(reference_dir: Path, out_dir: Path, dim: int = 16, seed: int = 0,
             methods: list[str] | None = None) -> dict[str, Path]:
    config = load_config()
    ref = load_reference_features(reference_dir, config)
    paths = {}
    for name in methods or list(METHODS):
        X = METHODS[name](ref, dim, seed)
        paths[name] = write_embeddings(Path(out_dir) / name / "embeddings.csv", ref.index, X, config)
        print(f"{name}: {X.shape[0]} hosts x {X.shape[1]} dims -> {paths[name]}")
    return paths


def evaluate(paths: dict[str, Path], reference_dir: Path, results_dir: Path) -> None:
    """Run the existing common evaluator (no metric logic lives here)."""
    from evaluation.evaluate import run

    results_dir.mkdir(parents=True, exist_ok=True)
    for name, path in paths.items():
        result = run(path, reference_dir, f"baseline_{name}")
        (results_dir / f"baseline_{name}.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--reference-dir", type=Path, default=Path("reference"))
    p.add_argument("--out-dir", type=Path, default=Path("submissions/baselines"))
    p.add_argument("--dim", type=int, default=16, help="embedding dimension (default 16)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--methods", nargs="+", choices=list(METHODS), default=None)
    p.add_argument("--evaluate", action="store_true", help="also run the common evaluator")
    p.add_argument("--results-dir", type=Path, default=Path("benchmark/results"))
    args = p.parse_args()

    paths = generate(args.reference_dir, args.out_dir, args.dim, args.seed, args.methods)
    if args.evaluate:
        evaluate(paths, args.reference_dir, args.results_dir)


if __name__ == "__main__":
    main()
