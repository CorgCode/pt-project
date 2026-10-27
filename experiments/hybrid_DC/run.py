"""python -m experiments.hybrid_DC.run {prepare,leader,tune,evaluate,all}"""
from __future__ import annotations

import argparse
import copy
import gc
import json
import pickle
import shutil
import time
from pathlib import Path

import joblib
import numpy as np

from evaluation import scoreboard, submit
from evaluation.evaluate import run as evaluate
from evaluation.io import load_embeddings
from evaluation.validate import validate_submission
from experiments.hybrid_DC.data import ROOT, BOARD, LocalData, dump, prepare, sha
from experiments.hybrid_DC.method import LeaderGraphHybrid, LinearNeighborhood

WEIGHTS = (0., .1, .25, .5, 1., 2.)
PARAMS = ROOT / "scoreboard/submissions/temporal_encoder_tuned/params.json"


def log(message):
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def make_leader(work):
    import torch
    from scoreboard.temporal_encoder import TemporalEncoder
    folder = work / "leader"
    folder.mkdir(exist_ok=True)
    if (folder / "manifest.json").exists():
        saved = json.loads((folder / "manifest.json").read_text())
        if saved["published_params_sha256"] != sha(PARAMS):
            raise ValueError("Параметры опубликованного лидера изменились")
        return
    params = json.loads(PARAMS.read_text())
    data = LocalData(work)
    started = time.perf_counter()
    model = TemporalEncoder(recipe=params["recipe"], epochs=params["epochs"], pca_dim=params["pca_dim"],
                            splits=params["splits"], encoder=params["encoder"]).fit(data)
    for window, hosts, filename in (("train", data.cohort, "embeddings.csv"),
                                     ("validation", data.evaluable("validation"), "validation.csv")):
        model.embed(data, window, hosts).to_csv(folder / filename)
    # У _encoder локальный класс: сохраняем состояния отдельно от Python-объекта сети.
    torch.save({"inputs": model.net.net[0].in_features, "state_dict": model.net.state_dict()}, folder / "network.pt")
    attrs = {k:v for k,v in model.__dict__.items() if k not in ("net", "_parts_cache")}
    joblib.dump(attrs, folder / "transforms.joblib", compress=3)
    dump(folder / "manifest.json", {"params": params, "published_params_sha256": sha(PARAMS),
                                    "fit_windows": ["train"], "test_read": False,
                                    "seconds": time.perf_counter()-started, "torch": torch.__version__})
    log("лидер восстановлен по опубликованным параметрам; test ещё не встраивался")


def restore_leader(work):
    import torch
    from scoreboard.temporal_encoder import TemporalEncoder, _encoder
    folder = work / "leader"
    torch.set_num_threads(2)
    model = TemporalEncoder()
    model.__dict__.update(joblib.load(folder / "transforms.joblib"))
    model._parts_cache = {}
    state = torch.load(folder / "network.pt", weights_only=True)
    model.net = _encoder(state["inputs"])
    model.net.load_state_dict(state["state_dict"])
    return model


def graph_paths(graph_dir):
    graph_dir = Path(graph_dir)
    return {"train": graph_dir / "submissions/linear/embeddings.csv",
            "validation": graph_dir / "baselines_validation/linear/validation.csv",
            "test": graph_dir / "submissions/linear/test.csv"}


def metric_score(folder, window, bootstrap=0):
    ref = BOARD / "reference" / window
    r = evaluate(folder / "embeddings.csv", ref, folder.name, n_boot=bootstrap,
                 fit_reference_dir=BOARD / "reference/validation" if window == "test" else None)
    submit.add_identity(r, folder / "embeddings.csv", folder / f"{window}.csv", ref, shared=True, n_boot=bootstrap)
    cols = {s["metric"]:s["title"] for s in r["summary"]}
    return float(scoreboard.overall_score([r], cols).iloc[0]), r


def tune(work, graph_dir, weights=WEIGHTS):
    if (work / "selection.json").exists():
        log("вес уже зафиксирован; повторный подбор отказан")
        return
    if not weights or 0. not in weights:
        raise ValueError("В заранее заданной сетке должен быть вес 0 — исходный лидер")
    paths = graph_paths(graph_dir)
    graph_meta = json.loads((Path(graph_dir) / "cache/manifest.json").read_text())
    if graph_meta["target_manifest_sha256"] != sha(BOARD / "processed/manifest.json"):
        raise ValueError("Графовый блок подготовлен для другого общего датасета")
    # Координаты из двух стадий предыдущего опыта проверяются по их train-векторам.
    graph_train = load_embeddings(paths["train"])
    val_training = load_embeddings(Path(graph_dir) / "baselines_validation/linear/embeddings.csv").loc[graph_train.index]
    np.testing.assert_allclose(graph_train.to_numpy(), val_training.to_numpy(), atol=1e-6)
    graph_val = load_embeddings(paths["validation"])
    # Проверяем, что самостоятельная реализация точно воспроизводит старый графовый блок.
    with (Path(graph_dir) / "features.pkl").open("rb") as stream:
        features = pickle.load(stream)
    linear = LinearNeighborhood().fit(features["train"]["x"], features["train"]["counts"], features["cohort"])
    for window, cached in (("train", graph_train), ("validation", graph_val)):
        values = linear.transform(features[window]["x"], features[window]["counts"])[features["cohort"]]
        cached = cached.loc[features["cohort_names"]]
        np.testing.assert_allclose(values, cached.to_numpy(), atol=1e-6)
    joblib.dump(linear, work / "linear_graph.joblib", compress=3)
    del features
    base_train = load_embeddings(work / "leader/embeddings.csv")
    base_val = load_embeddings(work / "leader/validation.csv")
    dump(work / "protocol.json", {"weights": list(weights), "output_dimension": base_train.shape[1],
                                  "selection": "official validation score", "test_used_for_selection": False,
                                  "normalization": "train RMS per block; graph centered with train mean",
                                  "projection": "train-only full PCA for positive weight; exact baseline for zero",
                                  "leader_params": json.loads(PARAMS.read_text()),
                                  "input_sha256": {str(p):sha(p) for p in [paths["train"], paths["validation"],
                                      work/"leader/embeddings.csv", work/"leader/validation.csv"]}})
    trials, best = [], None
    for i, weight in enumerate(weights):
        folder = work / "candidates" / f"candidate_{i:02d}"
        folder.mkdir(parents=True, exist_ok=True)
        model = LeaderGraphHybrid(weight).fit(base_train, graph_train)
        if not weight:
            shutil.copy2(work / "leader/embeddings.csv", folder / "embeddings.csv")
            shutil.copy2(work / "leader/validation.csv", folder / "validation.csv")
        else:
            model.transform(base_train, graph_train).to_csv(folder / "embeddings.csv")
            model.transform(base_val, graph_val).to_csv(folder / "validation.csv")
        score, r = metric_score(folder, "validation")
        dump(folder / "validation.json", r)
        joblib.dump(model, folder / "fusion.joblib", compress=3)
        record = {"candidate": folder.name, "weight": float(weight), "validation_score": score,
                  "tasks": scoreboard.task_scores(r, {s["metric"]:s["title"] for s in r["summary"]})}
        trials.append(record)
        log(f"вес {weight:g}: validation={score:.4f}")
        if best is None or score > best["validation_score"]:
            best = record
    dump(work / "selection.json", {**best, "trials": trials, "test_read": False,
                                   "graph_dir": str(Path(graph_dir).resolve())})
    shutil.copy2(work / "candidates" / best["candidate"] / "fusion.joblib", work / "fusion.joblib")
    log(f"зафиксирован вес {best['weight']:g}")


def final_evaluation(work, bootstrap=200):
    if (work / "results/hybrid.json").exists():
        log("итоговая оценка уже сохранена")
        return
    selected = json.loads((work / "selection.json").read_text())
    model = restore_leader(work)
    data = LocalData(work)
    test = work / "leader/test.csv"
    model.embed(data, "test", data.evaluable("test")).to_csv(test)
    del model, data
    gc.collect()
    graph_test = load_embeddings(graph_paths(selected["graph_dir"])["test"])
    base_test = load_embeddings(test)
    fusion = joblib.load(work / "fusion.joblib")
    folder = work / "export"
    folder.mkdir(exist_ok=True)
    chosen = work / "candidates" / selected["candidate"]
    shutil.copy2(chosen / "embeddings.csv", folder / "embeddings.csv")
    shutil.copy2(chosen / "validation.csv", folder / "validation.csv")
    if not fusion.weight:
        shutil.copy2(test, folder / "test.csv")
    else:
        fusion.transform(base_test, graph_test).to_csv(folder / "test.csv")
    shutil.copy2(work / "fusion.joblib", folder / "fusion.joblib")
    shutil.copy2(work / "linear_graph.joblib", folder / "linear_graph.joblib")
    (folder / "SHARED_COORDINATES").touch()
    dump(folder / "provenance.json", {"trained_on_windows": ["train"], "selection_window": "validation",
                                       "weight": selected["weight"], "dimensions": fusion.dim_,
                                       "available_test_windows": ["test"], "missing_test_windows": ["test_day03", "test_day04"]})
    base_score, base = metric_score(work / "leader", "test", bootstrap)
    base["method"] = "leader_reproduced"
    if not fusion.weight:
        for filename in ("embeddings.csv", "test.csv"):
            if sha(folder / filename) != sha(work / "leader" / filename):
                raise ValueError("Нулевой вес должен точно сохранять файлы лидера")
        hybrid_score, hybrid = base_score, copy.deepcopy(base)
        hybrid["identical_baseline_files_verified"] = True
    else:
        hybrid_score, hybrid = metric_score(folder, "test", bootstrap)
    hybrid["method"] = "leader_graph_hybrid"
    scoreboard.check_compatible([base, hybrid])
    dump(work / "results/hybrid.json", hybrid)
    dump(work / "results/leader.json", base)
    dump(work / "result.json", {"weight": selected["weight"], "baseline_score": base_score,
                                "hybrid_score": hybrid_score, "delta": hybrid_score-base_score,
                                "test_window": "test", "fresh_unseen_test": False})
    (work / "comparison.md").write_text(scoreboard.headline_table([base, hybrid]))
    validate_submission(str(folder / "embeddings.csv"))
    validate_submission(str(folder / "test.csv"))
    log(f"итог: лидер {base_score:.4f}, гибрид {hybrid_score:.4f}, разница {hybrid_score-base_score:+.4f}")


def report(work):
    import platform
    import sklearn
    out = ROOT / "results/leader_graph_hybrid_day02"
    out.mkdir(parents=True, exist_ok=True)
    shutil.copy2(work / "linear_graph.joblib", work / "export/linear_graph.joblib")
    shutil.copy2(Path(__file__).parent / "README.md", work / "export/README.md")
    selection = json.loads((work / "selection.json").read_text())
    result = json.loads((work / "result.json").read_text())
    for filename in ("selection.json", "protocol.json", "result.json", "comparison.md"):
        shutil.copy2(work / filename, out / filename)
    shutil.copytree(work / "results", out / "metrics", dirs_exist_ok=True)
    shutil.copy2(work / "data/manifest.json", out / "data_manifest.json")
    shutil.copy2(work / "leader/manifest.json", out / "leader_manifest.json")
    if (work / "verification.json").exists():
        shutil.copy2(work / "verification.json", out / "verification.json")
    dump(out / "environment.json", {"python": platform.python_version(), "numpy": np.__version__,
                                    "sklearn": sklearn.__version__,
                                    "code_sha256": {str(p.relative_to(ROOT)):sha(p) for p in sorted(Path(__file__).parent.glob("*.py"))}})
    lines = ["# Лидер + линейный графовый блок: day02", "",
             f"Реализован метод и проверены шесть весов добавки. На validation выбран **α={selection['weight']:g}**.", "",
             f"Итоговый test: исходный лидер **{result['baseline_score']:.4f}**, выбранный вариант **{result['hybrid_score']:.4f}**; разница **{result['delta']:+.4f}**.", "",
             "| вес α | validation score |", "|---|---|"]
    lines += [f"| {r['weight']:g} | {r['validation_score']:.4f} |" for r in selection["trials"]]
    lines += ["", "При нулевом весе исходные CSV лидера копируются без изменений, что проверяется по SHA-256. "
              "Идентичные файлы оцениваются один раз; результат выбранного варианта совпадает с baseline по построению.", "",
              "Лидер обучен заново по опубликованным параметрам temporal_encoder_tuned, seed17, 4 среза train, 20 эпох, "
              "PCA-384. Графовый блок — 64 координаты линейного агрегата, воспроизведённого по 95 признакам DC-SAGE. "
              "Положительные веса используют нормировки и PCA только по train. Все кандидаты остаются 384-мерными.", "",
              "Выбор веса и фиксация модели выполнены до оценки test. Дополнительный подбор после просмотра test не проводился. "
              "Это известный test дня 02, а не новая независимая проверка. Дни 03–04 локально отсутствуют.", "",
              "[Полные метрики](comparison.md), [протокол](protocol.json), [выбор веса](selection.json). "
              "[Код и команды](../../experiments/hybrid_DC/README.md). "
              "Готовые файлы: `runs/leader_graph_hybrid/export/`; состояния исходного лидера — `runs/leader_graph_hybrid/leader/`." ]
    if not selection["weight"]:
        lines += ["", "**Вывод:** добавка в проверенной сетке не улучшила validation, поэтому сохранён исходный лидер. "
                  "Реализация готова, но гипотеза об улучшении этим способом не подтвердилась."]
    (out / "README.md").write_text("\n".join(lines)+"\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("stage", choices=["prepare", "leader", "tune", "evaluate", "report", "all"])
    ap.add_argument("--workdir", type=Path, default=ROOT / "runs/leader_graph_hybrid")
    ap.add_argument("--graph-dir", type=Path, default=ROOT / "runs/dc_sage")
    ap.add_argument("--weights", nargs="+", type=float, default=list(WEIGHTS))
    ap.add_argument("--bootstrap", type=int, default=200)
    args = ap.parse_args()
    args.workdir.mkdir(parents=True, exist_ok=True)
    if args.stage in ("prepare", "all"):
        prepare(args.workdir)
    if args.stage in ("leader", "all"):
        make_leader(args.workdir)
    if args.stage in ("tune", "all"):
        tune(args.workdir, args.graph_dir, args.weights)
    if args.stage in ("evaluate", "all"):
        final_evaluation(args.workdir, args.bootstrap)
    if args.stage in ("evaluate", "report", "all"):
        report(args.workdir)


if __name__ == "__main__":
    main()
