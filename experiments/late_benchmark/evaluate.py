"""Use the official evaluator on all predeclared late windows, without retuning."""
import argparse
import ast
import hashlib
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
PUBLISHED_METRICS = (*REQUIRED, *scoreboard.EXTRA)
ADAPTER_FUNCTIONS = {"_score_window", "score", "combine_windows", "_train_file", "_mean", "test_windows", "window_file", "_inputs"}


def adapter_sha256(source=None):
    """Hash scoring glue, independently of Markdown rendering and comments."""
    source = source if source is not None else Path(submit.__file__).read_text()
    tree = ast.parse(source)
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in ADAPTER_FUNCTIONS]
    if {node.name for node in nodes} != ADAPTER_FUNCTIONS:
        raise ValueError("scoring adapter functions are missing")
    return hashlib.sha256(ast.dump(ast.Module(body=nodes, type_ignores=[]), include_attributes=False).encode()).hexdigest()


def cache_matches(cached, inputs, training, evaluator_sha256, seed, n_bootstrap, validation_files, adapter):
    """A vector/reference match alone is insufficient after evaluator or fit changes."""
    return (cached.get("late_inputs") == inputs
            and cached.get("training_provenance") == training
            and cached.get("code_sha256") == evaluator_sha256
            and cached.get("late_adapter_sha256") == adapter
            and cached.get("seed") == seed and cached.get("n_bootstrap") == n_bootstrap
            and bool(cached.get("by_window"))
            and all(row.get("reference_info", {}).get("fit_reference_files_sha256") == validation_files
                    and all(isinstance(row.get("metrics", {}).get(k), (int, float))
                            and np.isfinite(row["metrics"][k]) for k in PUBLISHED_METRICS)
                    for row in cached["by_window"].values()))


def score_method(name, data_root=ROOT / "data/late_benchmark", work=WORK):
    data_root, work = Path(data_root), Path(work)
    protocol = load_protocol()
    reference = data_root / "reference"
    if not (reference / "complete.json").exists():
        raise ValueError("common late references/calibration have not been frozen")
    folder = work / "models" / name
    validation = json.loads((reference / "validation/manifest.json").read_text())
    for filename, digest in validation["files_sha256"].items():
        if digest is not None and _sha256(reference / "validation" / filename) != digest:
            raise ValueError("a frozen validation reference was modified")
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
    submit.DATA, submit.RESULTS, submit.N_BOOT, submit.SEED = data_root, work / "results", protocol["bootstrap"], protocol["seed"]
    evaluator_sha256 = submit._code_sha256()
    adapter = adapter_sha256()
    if result_file.exists():
        cached = json.loads(result_file.read_text())
        if (cached.get("method") == name and set(cached.get("by_window", {})) == set(protocol["test_days"])
                and cache_matches(cached, inputs, training, evaluator_sha256, protocol["seed"], protocol["bootstrap"], validation["files_sha256"], adapter)):
            return cached
    for filename in ("README.md", "AUTHOR"):
        source = ROOT / "scoreboard/submissions" / name / filename
        if source.exists():
            shutil.copy2(source, folder / filename)
    if name != "random":
        (folder / "SHARED_COORDINATES").touch()
    result = submit.score(folder)
    for window, row in result["by_window"].items():
        if any(row["metrics"].get(key) is None or not np.isfinite(row["metrics"][key]) for key in PUBLISHED_METRICS):
            raise ValueError(f"{name}/{window}: incomplete mandatory evaluation")
    result.update(late_inputs=inputs, training_provenance=training, late_adapter_sha256=adapter)
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
