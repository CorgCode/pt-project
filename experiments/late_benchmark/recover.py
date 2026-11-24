"""Recover the published PR 150/151/166 methods using their original builders.

Fixed published recipes are refitted; late windows are only transformed/scored.
This does not claim bit-identical recovery of an absent historical checkpoint.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pandas as pd

from evaluation.io import load_embeddings
from pipeline.data import ProcessedData
from pipeline.prepare import _sha256
from .download import atomic_json
from .evaluate import score_method
from .models import ROOT, SOURCE, WORK, parameters
from .prepare import load_protocol

RECOVERED = ("gnn_v1p1_tuned", "gnn_big", "gnn_new", "gnn_v2", "gnn_tfidf_ref",
             "gnn_tfidf_new", "ref_g", "ref_g_fff", "ref_x", "ref_x_f", "ref_xm", "multi_x", "multi_xd")

LIMITATIONS = {
    "multiday_codebook_plus": {
        "status": "unavailable", "code_available": False, "reviewed_pr": 153,
        "reason_ru": "PR #153 добавляет карточку, параметры и результаты, но не код обучения/сборки Plus. "
                     "Оригинальная partner-head с опубликованным SHA отсутствует; описание позволяет отдельную реконструкцию, "
                     "но не подтверждение оригинальных весов."},
    "multi_xd2": {
        "status": "unavailable", "code_available": True, "reviewed_pr": 166,
        "reason_ru": "Код cross_day.py найден в PR #166. Для multi_xd и multi_xd2 сохранены одинаковые веса склейки, "
                     "но разные исходные векторы; точная конфигурация второго encoder (окна/исключения/параметры) не зафиксирована. "
                     "Нельзя выдать повтор первого варианта за восстановление второго."},
}


def write_audit(work):
    work = Path(work)
    statuses = json.loads((work / "model_status.json").read_text())
    statuses.update(LIMITATIONS)
    for name in RECOVERED:
        if statuses.get(name, {}).get("status") != "trained":
            statuses[name] = {"status": "pending_recovery", "reason": "Original implementation found in merged PR 151/166."}
    atomic_json(work / "model_status.json", statuses)
    atomic_json(work / "recovery/audit.json", {
        "reviewed_prs": [{"number": n, "url": f"https://github.com/CorgCode/pt-project/pull/{n}"} for n in (150, 151, 153, 166)],
        "previous_blanket_unavailable_classification_was_incorrect": True,
        "recoverable_from_original_code": ["behavioral_dae", "behavioral_dae_consistency", *RECOVERED],
        "recovery_policy": "Refit original implementations with published recipes; no late-test training or retuning. "
                           "This does not claim bit-identical missing historical checkpoints.",
        "remaining_details": LIMITATIONS})


def gnn_settings(name):
    if name == "gnn_v2":
        return {}
    return dict(dim=32 if name == "gnn_v1p1_tuned" else 128,
                hidden=64 if name == "gnn_v1p1_tuned" else 256,
                concat_k=32 if name == "gnn_v1p1_tuned" else 64, beta=2.04, lam=0.229)


def recipe(name):
    # PR 166 distinguishes its 66-feature profile from the reference 57-feature
    # behavioral block. These must not be silently interchanged.
    return {
        "gnn_tfidf_ref": ("gnn_big", True, True),
        "gnn_tfidf_new": ("gnn_new", False, True),
        "ref_g": ("gnn_new", True, True),
        "ref_g_fff": ("gnn_new", True, True),
        "ref_x": ("xb_s", True, True),
        "ref_x_f": ("xb_s", True, True),
        "ref_xm": ("xb_m", True, True),
        "multi_x": ("xb_s", True, True),
        "multi_xd": ("xb_s", True, True),
    }[name]


def run(data_root=ROOT / "data/late_benchmark", work=WORK):
    data_root, work = Path(data_root), Path(work)
    spec = load_protocol()
    data = ProcessedData(data_root / "processed")
    base = ProcessedData(SOURCE)
    if data.cohort != base.cohort:
        raise ValueError("source cohort changed")
    root = work / "recovery"
    root.mkdir(parents=True, exist_ok=True)
    write_audit(work)
    windows = ["train", "validation", *spec["test_days"]]
    filenames = ["embeddings.csv" if w == "train" else f"{w}.csv" for w in windows]

    def command(label, script, arguments, outputs, dependencies=()):
        receipt = root / f"{label}.json"
        fingerprint = {"script": _sha256(ROOT / "scoreboard" / script), "arguments": list(map(str, arguments)),
                       "protocol": spec["benchmark_id"], "processed": _sha256(data.root / "manifest.json"),
                       "dependencies": {str(p): _sha256(p) for p in dependencies}}
        if receipt.exists():
            old = json.loads(receipt.read_text())
            if old["inputs"] == fingerprint and all(p.exists() and old["outputs"].get(str(p)) == _sha256(p) for p in outputs):
                return
        print("recover stage:", label, flush=True)
        with (root / f"{label}.log").open("w") as log:
            subprocess.run([sys.executable, "-u", str(ROOT / "scoreboard" / script), *map(str, arguments)],
                           cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "OPENBLAS_NUM_THREADS": "2", "OMP_NUM_THREADS": "2"})
        atomic_json(receipt, {"inputs": fingerprint, "outputs": {str(p): _sha256(p) for p in outputs}})

    features = root / "features"
    feature_outputs = [features / w / f for w in windows for f in ("nodes.csv", "edges.csv", "services.csv")]
    command("features", "build_features.py", ["--processed", data.root, "--out", features], feature_outputs)

    def finish(name, details, dependencies):
        folder = work / "models" / name
        source = load_embeddings(folder / "embeddings.csv")
        if list(source.index) != data.cohort:
            raise ValueError(f"{name}: wrong source cohort")
        for window in spec["test_days"]:
            frame = load_embeddings(folder / f"{window}.csv")
            if list(frame.index) != data.evaluable(window) or frame.shape[1] != source.shape[1] or not np.isfinite(frame.to_numpy()).all():
                raise ValueError(f"{name}/{window}: invalid vectors")
        record = {"inputs": {"method": name, "source_manifest_sha256": _sha256(SOURCE / "manifest.json"),
                  "source_cohort_sha256": _sha256(SOURCE / "cohort.csv"), "parameters": parameters(name)},
                  "mode": "refit_original_pr_implementation_fixed_published_recipe", "n_hosts": len(source),
                  "dim": source.shape[1], "source_sha256": _sha256(folder / "embeddings.csv"),
                  "new_test_used_in_fit_or_selection": False, "recovery": details,
                  "implementation_sha256": {str(p.relative_to(ROOT)): _sha256(p) for p in dependencies}}
        atomic_json(folder / "training.json", record)
        for window in spec["test_days"]:
            fingerprint = {"benchmark_id": spec["benchmark_id"], "training": record["source_sha256"],
                           "processed_manifest": _sha256(data.root / "manifest.json"), "window": window}
            atomic_json(folder / f"{window}.json", {"inputs": fingerprint, "sha256": _sha256(folder / f"{window}.csv")})
        for filename in ("README.md", "AUTHOR", "SHARED_COORDINATES"):
            card = ROOT / "scoreboard/submissions" / name / filename
            if card.exists():
                shutil.copy2(card, folder / filename)
        score_method(name, data_root, work)
        statuses = json.loads((work / "model_status.json").read_text())
        statuses[name] = {"status": "trained", "training": record, "recovered_from_pr": details["pr"]}
        atomic_json(work / "model_status.json", statuses)

    for name in ("gnn_v1p1_tuned", "gnn_big", "gnn_new", "gnn_v2"):
        settings = gnn_settings(name)
        folder = work / "models" / name
        args = ["--features", features, "--processed", data.root, "--out", work / "models", "--name", name]
        for key, value in settings.items():
            args += ["--" + key.replace("_", "-"), value]
        script = "gnn_v2.py" if name == "gnn_v2" else "gnn_v1.py"
        command(name, script, args, [folder / f for f in filenames], feature_outputs)
        finish(name, {"pr": 151 if name != "gnn_new" else 166, "settings": settings,
                     "parameter_precision": "published submission card; absent original tuned parameter file",
                     "fit_windows": ["day02_train"], "exact_historical_weights": False},
               [ROOT / "scoreboard" / script, ROOT / "scoreboard/build_features.py"])

    blocks = root / "blocks"
    command("blocks", "build_blocks.py", ["--features", features, "--processed", data.root, "--out", blocks],
            [blocks / (w + ".npz") for w in windows], feature_outputs)
    bref = root / "behavioral"
    command("behavioral", "build_behavioral.py", ["--data", data.root, "--out", bref],
            [bref / (w + ".npz") for w in windows], [ROOT / "methods/baselines/temporal.py"])
    for name, dims in (("xb_s", (128, 64, 32)), ("xb_m", (256, 128, 64))):
        args = ["--features", features, "--processed", data.root, "--out", root, "--name", name]
        for key, value in zip(("m-peers", "m-services", "m-diff"), dims):
            args += ["--" + key, value]
        command(name, "build_xblock.py", args, [root / name / f for f in filenames], feature_outputs)

    old_root = ROOT / "PT_PROJECT_ML/multiday/processed"
    old_data = ProcessedData(old_root)
    if old_data.cohort != data.cohort or any(old_data.manifest["windows"][w]["end"] > 4 * 86400 for w in old_data.windows):
        raise ValueError("multiday fit inputs must use the original source cohort and days 02-04 only")
    old_features = root / "historical_features"
    old_inputs = [old_root / "manifest.json", old_root / "cohort.csv",
                  *[old_root / "flows" / f"{w}.csv" for w in old_data.windows]]
    old_outputs = [old_features / w / f for w in ("train", "test", "test_day03", "test_day04") for f in ("nodes.csv", "edges.csv", "services.csv")]
    command("historical_features", "build_features.py", ["--processed", old_root, "--out", old_features,
            "--windows", "train", "test", "test_day03", "test_day04"], old_outputs, old_inputs)
    multiblocks = root / "multiblocks"
    command("multiblocks", "build_blocks.py", ["--features", features, "--processed", data.root, "--out", multiblocks,
            "--extra-fit", old_features / "test_day03", old_features / "test_day04"],
            [multiblocks / (w + ".npz") for w in windows], feature_outputs + [p for p in old_outputs if p.parent.name != "test"])

    sys.path.insert(0, str(ROOT / "scoreboard"))
    import gnn_combine_v3 as combine
    for name in RECOVERED[4:-1]:
        fourth, reference_behavior, center = recipe(name)
        block_dir = multiblocks if name == "multi_x" else blocks
        fourth_dir = (work / "models" if fourth.startswith("gnn") else root) / fourth
        model = combine.Data(block_dir, fourth_dir, data.root, center=center, behav_dir=bref if reference_behavior else None)
        weights = parameters(name)
        folder = work / "models" / name
        model.write(folder, weights, windows)
        del model
        finish(name, {"pr": 151 if name == "gnn_tfidf_ref" else 166, "fourth_block": fourth,
                     "reference_behavior": reference_behavior, "center_by_train": center,
                     "fit_windows": ["day02_train", "day03", "day04"] if name == "multi_x" else ["day02_train"],
                     "exact_historical_weights": False},
               [ROOT / "scoreboard" / f for f in ("gnn_combine_v3.py", "build_blocks.py", "build_xblock.py", "build_behavioral.py")])

    # PR 166's cross-day encoder uses the same source-only SVD axes for both
    # historical and late windows. Never fit these axes on a late window.
    all_features = root / "all_features"
    all_features.mkdir(exist_ok=True)
    for window in windows:
        target = all_features / window
        if not target.exists():
            target.symlink_to((features / window).resolve(), target_is_directory=True)
    for name, window in (("d02", "test"), ("d03", "test_day03"), ("d04", "test_day04")):
        target = all_features / ("old_" + name)
        if not target.exists():
            target.symlink_to((old_features / window).resolve(), target_is_directory=True)
    all_blocks = root / "all_blocks"
    command("all_blocks", "build_blocks.py", ["--features", all_features, "--processed", data.root, "--out", all_blocks],
            [all_blocks / (w + ".npz") for w in [*windows, "old_d02", "old_d03", "old_d04"]], feature_outputs + old_outputs)
    old_bref = root / "historical_behavioral"
    command("historical_behavioral", "build_behavioral.py", ["--data", old_root, "--out", old_bref],
            [old_bref / (w + ".npz") for w in old_data.windows], old_inputs)
    command("cross_day", "cross_day.py", ["--blocks", all_blocks, "--behav-dir", bref, "--old-behav-dir", old_bref,
            "--processed", data.root, "--out", root, "--name", "xtenc", "--holdout", "none"],
            [root / "xtenc" / f for f in filenames],
            [all_blocks / (w + ".npz") for w in [*windows, "old_d02", "old_d03", "old_d04"]])
    name = "multi_xd"
    if "extra" not in combine.ORDER:
        combine.ORDER.append("extra")
    try:
        model = combine.Data(all_blocks, root / "xb_s", data.root, center=True, behav_dir=bref, extra=root / "xtenc")
        model.write(work / "models" / name, parameters(name), windows)
    finally:
        combine.ORDER.remove("extra")
    finish(name, {"pr": 166, "fit_windows": ["day02_train", "day02_12:30", "day03", "day04"],
                 "cross_day_parameters": "cross_day.py published defaults, holdout=none, no retuning",
                 "exact_historical_weights": False},
           [ROOT / "scoreboard" / f for f in ("cross_day.py", "gnn_combine_v3.py", "build_blocks.py")])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/late_benchmark")
    parser.add_argument("--work", type=Path, default=WORK)
    args = parser.parse_args()
    run(args.data, args.work)
