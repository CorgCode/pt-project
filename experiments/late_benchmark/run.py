"""Resume the complete local migration once the authorised archive downloads finish.

No network publishing, no push, and no change to the active scoreboard before the
full evaluation passes its publication gate. Progress lives in runs/late_benchmark.
"""
from __future__ import annotations

import argparse
import fcntl
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

from .download import atomic_json
from .models import ROOT, WORK
from .prepare import load_protocol


def now():
    return datetime.now(timezone.utc).isoformat()


def run(data_root=ROOT / "data/late_benchmark", work=WORK, *, wait=False, publish=True):
    data_root, work = Path(data_root), Path(work)
    work.mkdir(parents=True, exist_ok=True)
    protocol = load_protocol()
    state_path = work / "state.json"
    def state(stage, **details):
        value = {"at": now(), "benchmark_id": protocol["benchmark_id"], "stage": stage,
                 "publication": "local_only_no_push", **details}
        atomic_json(state_path, value)
        return value
    def command(stage, module, *arguments):
        state(stage)
        with (work / f"{stage}.log").open("a") as log:
            result = subprocess.run([sys.executable, "-u", "-m", module, *map(str, arguments)], cwd=ROOT,
                stdout=log, stderr=subprocess.STDOUT,
                env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2", "PT_COMPACT": "1",
                     "PT_PUBLICATION_LOCAL_ONLY": "1"})
        if result.returncode:
            state("failed", failed_stage=stage, log=str(work / f"{stage}.log"), exit_code=result.returncode)
            raise RuntimeError(f"stage {stage} failed; historical scoreboard preserved until publication")
    expected = [data_root / "raw" / f"netflow_day-{day:02d}.bz2" for day in protocol["test_days"].values()]
    while not all(path.is_file() and path.with_suffix(path.suffix + ".json").exists() for path in expected):
        missing = [path.name for path in expected if not path.exists()]
        downloaded = sum(path.stat().st_size for path in (data_root / "raw").glob("*.parts/*.part"))
        downloaded += sum(path.stat().st_size for path in expected if path.exists())
        total = sum(json.loads(path.read_text())["bytes"] for path in (data_root / "raw").glob("*.parts/remote.json"))
        state("waiting_for_download", missing=missing, downloaded_bytes=downloaded, total_bytes=total)
        if not wait:
            return json.loads(state_path.read_text())
        # Waits are short and interruptible; no test references or scores are fabricated.
        time.sleep(15)
    training_path = work / "model_status.json"
    while True:
        statuses = json.loads(training_path.read_text()) if training_path.exists() else {}
        if statuses and not any(item["status"] == "training_on_original_source" for item in statuses.values()):
            break
        state("waiting_for_source_only_training")
        if not wait:
            return json.loads(state_path.read_text())
        time.sleep(15)
    failures = [name for name, item in statuses.items() if item["status"] == "failed"]
    if failures:
        state("failed", failed_stage="source_only_training", methods=failures)
        raise RuntimeError("source-only training failed")
    atomic_json(work / "environment.json", {"python": sys.version, "platform": platform.platform(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "packages": subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines(),
        "protocol": protocol, "at": now()})
    command("prepare", "experiments.late_benchmark.prepare", "--raw", data_root / "raw", "--output", data_root / "processed")
    command("reference", "experiments.late_benchmark.reference", "--processed", data_root / "processed", "--output", data_root / "reference",
            "--behavior", work / "models/behavioral/embeddings.csv")
    ready = [name for name, item in statuses.items() if item["status"] == "trained"]
    for name in ready:
        command("infer_" + name, "experiments.late_benchmark.models", "infer", "--method", name,
                "--processed", data_root / "processed", "--work", work)
        command("evaluate_" + name, "experiments.late_benchmark.evaluate", "--method", name, "--data", data_root, "--work", work)
    if publish:
        command("publish", "experiments.late_benchmark.publish", "--data", data_root, "--work", work)
        return state("complete", methods=ready, table=str(ROOT / "scoreboard/README.md"))
    return state("evaluation_complete_not_promoted", methods=ready)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/late_benchmark")
    parser.add_argument("--work", type=Path, default=WORK)
    parser.add_argument("--wait", action="store_true")
    parser.add_argument("--no-publish", action="store_true")
    args = parser.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("The late-benchmark pipeline is already running.") from None
        print(json.dumps(run(args.data, args.work, wait=args.wait, publish=not args.no_publish), indent=2), flush=True)


if __name__ == "__main__":
    main()
