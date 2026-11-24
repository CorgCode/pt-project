"""Build one common reference per late window and freeze explicit score calibration."""
import argparse
import json
from pathlib import Path
import shutil

from evaluation.frozen_neighbors import evaluate_frozen_neighbors
from evaluation.io import load_embeddings
from pipeline.prepare import _sha256
from pipeline.reference import build_reference

from .models import ROOT, WORK
from .prepare import load_protocol


def copy_validation(source, target):
    """Preserve metric CSVs; legacy AppleDouble hashes are not evaluator inputs."""
    source, target = Path(source), Path(target)
    if not target.exists():
        shutil.copytree(source, target)
    metadata = json.loads((source / "manifest.json").read_text())
    original_files = metadata["files_sha256"]
    metric_files = {name: digest for name, digest in original_files.items() if not name.startswith(".")}
    for name, digest in metric_files.items():
        if digest is not None and _sha256(target / name) != digest:
            raise ValueError("the original validation metric CSV changed")
    metadata["files_sha256"] = metric_files
    metadata["source_validation_manifest_sha256"] = _sha256(source / "manifest.json")
    metadata["excluded_legacy_nonmetric_files"] = {name: digest for name, digest in original_files.items() if name.startswith(".")}
    return metadata


def build(processed, output, source_reference=None, behavior=None):
    processed, output = Path(processed), Path(output)
    protocol = load_protocol()
    manifest = json.loads((processed / "manifest.json").read_text())
    if manifest.get("benchmark_id") != protocol["benchmark_id"]:
        raise ValueError("processed data belongs to another benchmark")
    for window in protocol["test_days"]:
        if _sha256(processed / "flows" / f"{window}.csv") != manifest["windows"][window]["flows_sha256"]:
            raise ValueError("a prepared test window changed")
    source_reference = Path(source_reference or ROOT / "PT_PROJECT_ML/late_test/reference/validation")
    behavior = Path(behavior or WORK / "models/behavioral/embeddings.csv")
    output.mkdir(parents=True, exist_ok=True)
    validation = output / "validation"
    source = load_embeddings(behavior)
    validation_meta = copy_validation(source_reference, validation)
    import pandas as pd
    val_hosts = pd.read_csv(validation / "eval_hosts.csv", dtype=str).host_id.tolist()
    val_agreement = evaluate_frozen_neighbors(source.loc[val_hosts], validation / "neighbors.csv", ks=(10,), split="validation")
    validation_meta.update(benchmark_id=protocol["benchmark_id"], windows=manifest["windows"],
        pipeline_manifest_sha256=_sha256(processed / "manifest.json"), cohort_sha256=manifest["cohort"]["sha256"],
        score_calibration={"neighbors_ideal": float(val_agreement.loc[(10, "consistency"), "mean"]),
                           "method": "behavioral_source_persistence", "source_embeddings_sha256": _sha256(behavior),
                           "n_evaluable": len(val_hosts)})
    (validation / "manifest.json").write_text(json.dumps(validation_meta, ensure_ascii=False, indent=2) + "\n")
    ready = output / "complete.json"
    inputs = {"processed_manifest": _sha256(processed / "manifest.json"), "behavior_source": _sha256(behavior),
              "validation_manifest": _sha256(validation / "manifest.json")}
    if ready.exists() and json.loads(ready.read_text())["inputs"] == inputs:
        return json.loads(ready.read_text())
    build_reference(processed, output, seed=protocol["seed"], pairs_per_anchor=protocol["pairs_per_anchor"],
                    targets=tuple(protocol["test_days"]))
    calibrations = {}
    for window in protocol["test_days"]:
        folder = output / window
        meta = json.loads((folder / "manifest.json").read_text())
        import pandas as pd
        hosts = pd.read_csv(folder / "eval_hosts.csv", dtype=str).host_id.tolist()
        agreement = evaluate_frozen_neighbors(source.loc[hosts], folder / "neighbors.csv", ks=(10,), split="test")
        ceiling = float(agreement.loc[(10, "consistency"), "mean"])
        if not ceiling > 10 / (len(hosts) - 1):
            raise ValueError(f"{window}: behavioral persistence is at/below chance; an overall score is undefined")
        calibration = {"neighbors_ideal": ceiling, "method": "behavioral_source_persistence",
                       "source_embeddings_sha256": _sha256(behavior), "n_evaluable": len(hosts)}
        meta.update(benchmark_id=protocol["benchmark_id"], score_calibration=calibration,
                    protocol_sha256=manifest["protocol_sha256"])
        (folder / "manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n")
        calibrations[window] = calibration
    result = {"inputs": inputs, "benchmark_id": protocol["benchmark_id"], "calibrations": calibrations,
              "reference_manifests_sha256": {w: _sha256(output / w / "manifest.json") for w in protocol["test_days"]}}
    ready.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed", type=Path, default=ROOT / "data/late_benchmark/processed")
    parser.add_argument("--output", type=Path, default=ROOT / "data/late_benchmark/reference")
    parser.add_argument("--behavior", type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.processed, args.output, behavior=args.behavior), indent=2), flush=True)


if __name__ == "__main__":
    main()
