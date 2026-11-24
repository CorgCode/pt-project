"""Fit reproducible methods on the frozen source; never open late test data."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from .download import atomic_json
from .models import AVAILABLE, ROOT, SOURCE, WORK


def run(base=SOURCE, work=WORK):
    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)
    status_path = work / "model_status.json"
    statuses = json.loads(status_path.read_text()) if status_path.exists() else {}
    for result in sorted((ROOT / "scoreboard/results").glob("*.json")):
        if result.stem not in AVAILABLE:
            statuses[result.stem] = {"status": "unavailable", "reason":
                "Original Plus head and inference assembly are unavailable; the base is evaluated separately."
                if result.stem == "multiday_codebook_plus" else
                "No complete reproducible inference adapter/checkpoint for a new window; historical test vectors are insufficient."}
    atomic_json(status_path, statuses)
    for name in AVAILABLE:
        log = work / f"train_{name}.log"
        statuses[name] = {"status": "training_on_original_source", "log": str(log.relative_to(ROOT))}
        atomic_json(status_path, statuses)
        command = [sys.executable, "-u", "-m", "experiments.late_benchmark.models", "train",
                   "--method", name, "--base", str(base), "--work", str(work)]
        print("source-only preparation:", name, flush=True)
        with log.open("a") as stream:
            completed = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PT_COMPACT": "1"})
        if completed.returncode:
            statuses[name] = {"status": "failed", "stage": "source_only_training", "log": str(log.relative_to(ROOT))}
        else:
            statuses[name] = {"status": "trained", "training": json.loads((work / "models" / name / "training.json").read_text())}
        atomic_json(status_path, statuses)
    return statuses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=SOURCE)
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    print(json.dumps(run(args.base, args.work), indent=2), flush=True)


if __name__ == "__main__":
    main()
