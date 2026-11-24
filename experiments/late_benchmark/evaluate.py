"""Use the official evaluator on all predeclared late windows, without retuning."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np

from evaluation import scoreboard, submit
from evaluation.io import load_embeddings
from pipeline.prepare import _sha256

from .models import ROOT, WORK
from .prepare import load_protocol

REQUIRED = ("consistency@10_mean", "link_learned_roc_auc", "partner_new_mrr", "pair_roc_auc", "identity_top1")


def score_method(name, data_root=ROOT / "data/late_benchmark", work=WORK):
    data_root, work = Path(data_root), Path(work)
    protocol = load_protocol()
    reference = data_root / "reference"
    if not (reference / "complete.json").exists():
        raise ValueError("common late references/calibration have not been frozen")
    folder = work / "models" / name
    training = json.loads((folder / "training.json").read_text())
    if _sha256(folder / "embeddings.csv") != training["source_sha256"]:
        raise ValueError("source vectors changed after source-only preparation")
    inputs = {"source_sha256": training["source_sha256"], "references": {}, "later": {}}
    for window in protocol["test_days"]:
        manifest = json.loads((reference / window / "manifest.json").read_text())
        if manifest.get("benchmark_id") != protocol["benchmark_id"]:
            raise ValueError("mixed old/new benchmark references")
        for window_name, bounds in protocol["windows"].items():
            actual = manifest["windows"].get(window_name, {})
            if [actual.get("start"), actual.get("end")] != bounds:
                raise ValueError("reference time windows do not match the declared late protocol")
        for filename, expected in manifest["files_sha256"].items():
            if expected is not None and _sha256(reference / window / filename) != expected:
                raise ValueError("a frozen test reference was modified")
        receipt = json.loads((folder / f"{window}.json").read_text())
        if (receipt["inputs"]["benchmark_id"] != protocol["benchmark_id"]
                or receipt["inputs"]["processed_manifest"] != _sha256(data_root / "processed/manifest.json")
                or receipt["inputs"]["training"] != training["source_sha256"]
                or receipt["sha256"] != _sha256(folder / f"{window}.csv")):
            raise ValueError("late vectors do not match this benchmark")
        inputs["references"][window] = _sha256(reference / window / "manifest.json")
        inputs["later"][window] = receipt["sha256"]
    result_file = work / "results" / f"{name}.json"
    if result_file.exists():
        cached = json.loads(result_file.read_text())
        if cached.get("late_inputs") == inputs:
            return cached
    for filename in ("README.md", "AUTHOR"):
        source = ROOT / "scoreboard/submissions" / name / filename
        if source.exists():
            shutil.copy2(source, folder / filename)
    if name != "random":
        (folder / "SHARED_COORDINATES").touch()
    submit.DATA, submit.RESULTS, submit.N_BOOT = data_root, work / "results", protocol["bootstrap"]
    result = submit.score(folder)
    for window, row in result["by_window"].items():
        if any(row["metrics"].get(key) is None or not np.isfinite(row["metrics"][key]) for key in REQUIRED):
            raise ValueError(f"{name}/{window}: incomplete mandatory evaluation")
    result.update(late_inputs=inputs, training_provenance=training)
    result_file.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    columns = {item["metric"]: item["title"] for item in result["summary"]}
    score = float(scoreboard.overall_score([result], columns)[name])
    if not np.isfinite(score):
        raise ValueError(f"{name}: normalized score is undefined")
    print(json.dumps({"method": name, "score": score, "metrics": {key: result["metrics"][key] for key in REQUIRED}}), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", required=True)
    parser.add_argument("--data", type=Path, default=ROOT / "data/late_benchmark")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    score_method(args.method, args.data, args.work)


if __name__ == "__main__":
    main()
