"""Frozen model settings; source-only refits where no serialised state exists."""
from __future__ import annotations

import argparse
import importlib
import json
import os
from pathlib import Path
import pickle
import subprocess
import sys
import tarfile

import numpy as np

from evaluation.io import load_embeddings
from pipeline.data import ProcessedData
from pipeline.prepare import _sha256

ROOT = Path(__file__).resolve().parents[2]
WORK = ROOT / "runs/late_benchmark"
SOURCE = ROOT / "PT_PROJECT_ML/late_test/processed"
FROZEN_CODE_COMMIT = "86d709d"
AVAILABLE = ("behavioral", "random", "temporal_stable_hybrid", "multiday_codebook", "hybrid_tfidf",
             "temporal_cca_tuned", "temporal_encoder", "temporal_encoder_tuned", "temporal_dae",
             "future_profile_predictor", "relation_graph_hybrid", "behavioral_dae", "behavioral_dae_consistency")


class LegacyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("experiments.fingerprint_"):
            module = "experiments.multiday." + module.removeprefix("experiments.fingerprint_")
        if module.startswith("experiments.multiday_gain"):
            module = module.replace("experiments.multiday_gain", "experiments.multiday.gain")
        return super().find_class(module, name)


def archived_source(work=WORK):
    root = Path(work) / "model_code"
    if not (root / "experiments/overnight/model.py").exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = Path(work) / "model_source.tar"
        with archive.open("wb") as stream:
            subprocess.run(["git", "archive", FROZEN_CODE_COMMIT, "experiments/multiday", "experiments/overnight"],
                           cwd=ROOT, stdout=stream, check=True)
        with tarfile.open(archive) as stream:
            stream.extractall(root, filter="data")
    import experiments
    path = str(root / "experiments")
    if path not in experiments.__path__:
        experiments.__path__.insert(0, path)
    sys.path.insert(0, str(ROOT / "scoreboard"))
    os.environ["PT_ML_ROOT"] = str(ROOT / "PT_PROJECT_ML")
    return root


def clear_window_cache(model, work=WORK):
    """Original caches sometimes omit dataset identity; never reuse old 'test' features."""
    for attribute in ("_parts_cache", "_evaluation_cache", "_behavior_cache", "_encoded", "_fan"):
        if hasattr(model, attribute):
            setattr(model, attribute, {})
    if isinstance(getattr(model, "cache", None), str):
        model.cache = str(Path(work) / "fingerprint_cache")
    for name in ("base",):
        nested = getattr(model, name, None)
        if nested is not None:
            clear_window_cache(nested, work)


def parameters(name):
    path = ROOT / "scoreboard/submissions" / name / "params.json"
    return json.loads(path.read_text()) if path.exists() else {}


def instantiate(name):
    sys.path.insert(0, str(ROOT / "scoreboard"))
    params = parameters(name)
    if name in ("behavioral_dae", "behavioral_dae_consistency"):
        from methods.baselines.autoencoder import BehavioralDAE, BehavioralDAEConsistency
        cls = BehavioralDAE if name == "behavioral_dae" else BehavioralDAEConsistency
        return cls(dim=16, hidden_dim=64, epochs=30, batch_size=256, seed=17)
    if name in ("behavioral", "random"):
        from methods.baselines.temporal import BehavioralBaseline, RandomBaseline
        return BehavioralBaseline() if name == "behavioral" else RandomBaseline(dim=16, seed=0)
    if name == "hybrid_tfidf":
        return importlib.import_module("hybrid_tfidf").HybridTfidf(dim=params["dim"], weights=params["weights"])
    if name in ("temporal_encoder", "temporal_encoder_tuned", "temporal_cca_tuned", "temporal_dae"):
        cls = (importlib.import_module("temporal_dae").TemporalDAE if name == "temporal_dae" else
               importlib.import_module("temporal_encoder").TemporalEncoder)
        allowed = ("recipe", "epochs", "pca_dim", "splits", "encoder", "reconstruction_weight")
        return cls(**{key: params[key] for key in allowed if key in params})
    if name == "future_profile_predictor":
        return importlib.import_module(name).FutureProfilePredictor(
            epochs=params["epochs"], pca_dim=params["pca_dim"], splits=params["splits"], weight=params["recipe"]["neural"])
    if name == "relation_graph_hybrid":
        return importlib.import_module(name).RelationGraphHybrid(pca_dim=params["pca_dim"], weight=params["recipe"]["relation"])
    raise ValueError(f"no source-only model adapter for {name}")


def load_model(name, work=WORK):
    sys.path.insert(0, str(ROOT / "scoreboard"))
    if name == "multiday_codebook":
        archived_source(work)
        checkpoint = ROOT / "PT_PROJECT_ML/fingerprint_codebook_recent/final/model.pkl"
        if _sha256(checkpoint) != parameters(name)["checkpoint_sha256"]:
            raise ValueError("the codebook checkpoint differs from the published SHA-256")
        with checkpoint.open("rb") as stream:
            model = LegacyUnpickler(stream).load()
        restore_native_classes(model)
        clear_window_cache(model, work)
        return model
    if name == "temporal_stable_hybrid":
        sys.path.insert(0, str(ROOT / "scoreboard"))
        checkpoint = ROOT / "runs/temporal_stable_checkpoint/temporal_stable_hybrid_day02.npz"
        record = json.loads(checkpoint.with_name("manifest.json").read_text())
        if _sha256(checkpoint) != record["sha256"]:
            raise ValueError("stable-hybrid checkpoint hash mismatch")
        model = importlib.import_module(name).TemporalStableHybrid.load(checkpoint)
        for key in ("dim", "stable_dim", "splits", "regularization", "min_flows", "behavioral_weight", "stable_weight"):
            if getattr(model, key) != parameters(name)[key]:
                raise ValueError(f"checkpoint settings do not match the published method: {key}")
        return model
    import cloudpickle
    with (Path(work) / "models" / name / "model.pkl").open("rb") as stream:
        model = cloudpickle.load(stream)
    clear_window_cache(model, work)
    return model


def restore_native_classes(model):
    """Keep checkpoint state but compile by-value legacy classes on this Python.

    Cloudpickle can deserialize older CPython bytecode without rejecting it;
    invoking those functions on Python 3.14 can segfault. Use the original
    repository implementations and retain all learned tensors and statistics.
    """
    bag = importlib.import_module("experiments.multiday.bag")
    expanded = importlib.import_module("experiments.multiday.expanded_data")
    temporal = importlib.import_module("temporal_encoder")
    native = {"BagHead": bag.BagHead, "BagNetwork": bag.BagNetwork,
              "BagConfig": bag.BagConfig, "ExpandedBank": expanded.ExpandedBank}
    seen = set()
    def visit(value):
        if id(value) in seen or not hasattr(value, "__dict__"):
            return
        seen.add(id(value))
        cls = type(value)
        if cls.__module__ == "__main__" and cls.__name__ in native:
            value.__class__ = native[cls.__name__]
        elif cls.__module__ == "temporal_encoder" and cls.__name__ == "Encoder":
            value.__class__ = type(temporal._encoder(value.shortcut.in_features))
        for key in ("base", "head", "bank", "net", "config"):
            if hasattr(value, key):
                visit(getattr(value, key))
    visit(model)


def train(name, base=SOURCE, work=WORK):
    if name not in AVAILABLE:
        raise ValueError(f"no reproducible inference implementation for {name}")
    base, work = Path(base), Path(work)
    folder = work / "models" / name
    folder.mkdir(parents=True, exist_ok=True)
    record_path = folder / "training.json"
    inputs = {"method": name, "source_manifest_sha256": _sha256(base / "manifest.json"),
              "source_cohort_sha256": _sha256(base / "cohort.csv"), "parameters": parameters(name)}
    if record_path.exists():
        previous = json.loads(record_path.read_text())
        if previous["inputs"] != inputs:
            raise ValueError("saved model belongs to different training data/settings")
        return previous
    data = ProcessedData(base)
    # Only the frozen source dataset is passed to fit; no late test CSV is reachable here.
    frozen = name in ("multiday_codebook", "temporal_stable_hybrid")
    model = load_model(name, work) if frozen else instantiate(name).fit(data)
    source = model.embed(data, "train", data.cohort)
    if list(source.index) != data.cohort or not np.isfinite(source.to_numpy()).all():
        raise ValueError("source output does not match the frozen cohort")
    source.to_csv(folder / "embeddings.csv", index_label="host_id", float_format="%.12g")
    old = ROOT / "PT_PROJECT_ML/late_test/vectors" / name / "embeddings.csv"
    comparison = None
    if old.exists():
        expected = load_embeddings(old)
        if source.shape == expected.shape and set(source.index) == set(expected.index):
            comparison = float(np.max(np.abs(source.loc[expected.index].to_numpy() - expected.to_numpy())))
    if name == "multiday_codebook" and (comparison is None or comparison > 1e-7):
        raise ValueError("codebook source differs from the authenticated original vectors")
    if not frozen:
        import cloudpickle
        clear_window_cache(model, work)
        if hasattr(model, "train") and not callable(model.train):
            model.train = None  # full raw train frame is not part of inference state
        with (folder / "model.pkl").open("wb") as stream:
            cloudpickle.dump(model, stream)
    result = {"inputs": inputs, "mode": "original_checkpoint" if frozen else "source_only_refit_fixed_published_parameters",
              "n_hosts": len(source), "dim": source.shape[1], "source_sha256": _sha256(folder / "embeddings.csv"),
              "published_source_max_abs_difference": comparison, "new_test_used_in_fit_or_selection": False}
    record_path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"trained": name, "dim": source.shape[1], "mode": result["mode"]}), flush=True)
    return result


def infer(name, processed, work=WORK):
    data, work = ProcessedData(Path(processed)), Path(work)
    folder = work / "models" / name
    training = json.loads((folder / "training.json").read_text())
    if (training["inputs"]["source_manifest_sha256"] != data.manifest["base_manifest_sha256"]
            or training["inputs"]["source_cohort_sha256"] != data.manifest["cohort"]["source_cohort_sha256"]):
        raise ValueError("model and late dataset have different frozen training sources")
    # A changed evaluation cohort may only remove inactive queries; source identities remain fixed.
    source = load_embeddings(folder / "embeddings.csv")
    if list(source.index) != data.cohort:
        raise ValueError("late dataset changed the model's source cohort")
    model = load_model(name, work)
    for window in data.test_windows:
        target = folder / f"{window}.csv"
        receipt = target.with_suffix(".json")
        fingerprint = {"benchmark_id": data.manifest["benchmark_id"], "training": training["source_sha256"],
                       "processed_manifest": _sha256(data.root / "manifest.json"), "window": window}
        if receipt.exists() and target.exists() and json.loads(receipt.read_text())["inputs"] == fingerprint:
            continue
        clear_window_cache(model, work)
        hosts = data.evaluable(window)
        embedding = model.embed(data, window, hosts)
        if list(embedding.index) != hosts or embedding.shape[1] != source.shape[1] or not np.isfinite(embedding.to_numpy()).all():
            raise ValueError("invalid late-window model output")
        embedding.to_csv(target, index_label="host_id", float_format="%.12g")
        receipt.write_text(json.dumps({"inputs": fingerprint, "sha256": _sha256(target)}, indent=2) + "\n")
        print(f"inferred {name}: {window}, {len(hosts)} hosts", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("train", "infer"))
    parser.add_argument("--method", choices=AVAILABLE, required=True)
    parser.add_argument("--base", type=Path, default=SOURCE)
    parser.add_argument("--processed", type=Path, default=ROOT / "data/late_benchmark/processed")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    train(args.method, args.base, args.work) if args.stage == "train" else infer(args.method, args.processed, args.work)


if __name__ == "__main__":
    main()
