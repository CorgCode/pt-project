"""One command: raw netflow -> windows -> frozen references -> all methods -> report.

    # real data (files from the LANL form):
    python -m experiments.end_to_end --netflow data/raw/netflow_day-0*.bz2 \
        --config my_config.json --workdir runs/lanl

    # pipeline smoke test on a SYNTHETIC LANL-format fixture (no download needed):
    python -m experiments.end_to_end --fixture --workdir runs/synthetic_demo

Stages and outputs under --workdir: processed/ (pipeline.prepare), reference/
(pipeline.reference), run/ (experiments.run_experiment: report.md, figures/, results/, audit.json).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from experiments.run_experiment import run_experiment
from pipeline import fixtures
from pipeline.prepare import prepare
from pipeline.reference import build_reference
from pipeline.windows import PipelineConfig

FIXTURE_CONFIG = PipelineConfig(windows={"train": (0, 21600), "validation": (21600, 32400),
                                         "test": (32400, 43200)})


def end_to_end(workdir: Path, *, netflow: list[Path] | None = None, config: PipelineConfig | None = None,
               fixture: bool = False, n_hosts: int = 300, dim: int = 16, seed: int = 0, n_boot: int = 200,
               submissions_dir: Path | None = None, pairs_per_anchor: int = 10, labels: Path | None = None,
               labels_source: str | None = None, overwrite: bool = False) -> dict:
    workdir = Path(workdir)
    if fixture:
        raw = workdir / "fixture"
        fixtures.generate(raw, n_hosts=n_hosts, seed=seed)
        netflow, config = sorted(raw.glob("netflow_day-*.bz2")), config or FIXTURE_CONFIG
    if not netflow or config is None:
        raise ValueError("provide --netflow files and --config, or use --fixture")
    prepare(netflow, config, workdir / "processed", overwrite=overwrite)
    build_reference(workdir / "processed", workdir / "reference", seed=seed,
                    pairs_per_anchor=pairs_per_anchor, labels=labels, labels_source=labels_source)
    return run_experiment(workdir / "processed", workdir / "reference", workdir / "run", dim=dim, seed=seed,
                          n_boot=n_boot, submissions_dir=submissions_dir)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", type=Path, required=True)
    ap.add_argument("--netflow", nargs="+", type=Path)
    ap.add_argument("--config", type=Path)
    ap.add_argument("--fixture", action="store_true", help="use a synthetic LANL-format fixture")
    ap.add_argument("--n-hosts", type=int, default=300, help="fixture size")
    ap.add_argument("--dim", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--bootstrap", type=int, default=200)
    ap.add_argument("--submissions-dir", type=Path)
    ap.add_argument("--labels", type=Path)
    ap.add_argument("--labels-source")
    ap.add_argument("--overwrite", action="store_true")
    a = ap.parse_args()
    m = end_to_end(a.workdir, netflow=a.netflow, config=PipelineConfig.from_json(a.config) if a.config else None,
                   fixture=a.fixture, n_hosts=a.n_hosts, dim=a.dim, seed=a.seed, n_boot=a.bootstrap,
                   submissions_dir=a.submissions_dir, labels=a.labels, labels_source=a.labels_source,
                   overwrite=a.overwrite)
    print(json.dumps({k: m[k] for k in ("data_kind", "n_hosts", "methods", "missing_expected_methods")}, indent=2))
    print(f"report: {a.workdir / 'run' / 'report.md'}")


if __name__ == "__main__":
    main()
