"""Склейка четырёх блоков в один вектор хоста; веса блоков подбираются через Optuna.

    python scoreboard/combine_v3.py --gnn <gnn_submission> --behav-dir <behav> --trials <cnt> [--score]

Блоки (все строятся по train, одинаковые оси координат для всех окон):
  behav     профиль хоста, 66 чисел            (runs/blocks/<окно>.npz, делает build_blocks.py)
  peers     TF-IDF "с кем общается" -> SVD     (первые k_peers компонент)
  services  TF-IDF "какими сервисами" -> SVD   (первые k_services компонент)
  gnn       готовый эмбеддинг GNN               (<--gnn>/embeddings.csv и <окно>.csv, делает gnn_v1.py)
Каждый блок делится на свой "масштаб" (корень из среднего квадрата длины строки по train), поэтому у каждого блока
средний квадрат длины = 1, и вес w значит одно и то же для всех блоков. Вектор = [1·behav, w_peers·peers,
w_services·services, w_gnn·gnn]. Вес поведения зафиксирован 1 — остальные веса относительные (общий множитель на
расстояния между хостами не влияет). Доля блока в векторе = w² / сумма w² (печатается в итоге).
Подбор: w_peers, w_services, w_gnn ∈ [0, 3], k_peers ∈ {64,128,256}, k_services ∈ {32,64,128}; цель — validation_score.
После подбора печатаются "абляции" (без GNN, без TF-IDF, только поведение) — они показывают вклад каждого блока.
Результат: runs/submissions/<имя>/ (embeddings.csv, validation.csv, test*.csv, SHARED_COORDINATES, README.md, params.json).
Нужны optuna и torch (CPU), как для gnn_v1.py.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

K_PEERS, K_SERVICES = [64, 128, 256], [32, 64, 128]
ORDER = ["behav", "peers", "services", "gnn"]
RUS = {"behav": "поведение (профиль)", "peers": "собеседники (TF-IDF)", "services": "сервисы (TF-IDF)", "gnn": "GNN-эмбеддинг"}


def log(msg: str) -> None:
    print(msg, flush=True)


class Data:
    def __init__(self, blocks: Path, gnn: Path, processed: Path, center: bool = True, official_val: bool = False,
                 behav_dir: Path | None = None):
        self.center, self.official_val, self.behav_dir = center, official_val, behav_dir
        self.cohort = pd.read_csv(processed / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
        self.hosts = self.cohort.index
        files = sorted(blocks.glob("*.npz"))
        if not files:
            raise SystemExit(f"нет {blocks}/*.npz — сначала python scoreboard/build_blocks.py")
        self.windows = ["train"] + [f.stem for f in files if f.stem != "train"]
        self.X: dict[str, dict[str, np.ndarray]] = {}
        self.ev: dict[str, np.ndarray] = {}
        for w in self.windows:
            z = np.load(blocks / f"{w}.npz")
            if list(z["hosts"]) != list(self.hosts):
                raise SystemExit(f"{w}.npz: порядок хостов не совпадает с cohort.csv — пересоберите блоки")
            f = gnn / ("embeddings.csv" if w == "train" else f"{w}.csv")
            if not f.exists():
                raise SystemExit(f"нет {f} — это эмбеддинг GNN для окна {w}")
            g = pd.read_csv(f, index_col=0, dtype={0: str}).reindex(self.hosts).fillna(0.0).to_numpy(np.float32)
            self.X[w] = {"behav": self._behav(z, w), "peers": z["peers"], "services": z["services"], "gnn": g}
            self.ev[w] = np.ones(len(self.hosts), bool) if w == "train" else self.cohort[f"evaluable_{w}"].to_numpy(bool)
        self.kmax = {"peers": self.X["train"]["peers"].shape[1], "services": self.X["train"]["services"].shape[1]}
        log(f"данные: хостов {len(self.hosts):,}; окна {self.windows}; peers {self.kmax['peers']}, services {self.kmax['services']}, "
            f"gnn {self.X['train']['gnn'].shape[1]}")

    def _behav(self, z, w: str) -> np.ndarray:
        if self.behav_dir is None:
            return z["behav"]
        r = np.load(self.behav_dir / f"{w}.npz")
        if list(r["hosts"]) != list(self.hosts):
            raise SystemExit(f"{self.behav_dir}/{w}.npz: порядок хостов не совпадает с cohort.csv")
        return r["behav"]

    def parts(self, w: str, p: dict) -> dict[str, np.ndarray]:
        """Взвешенные блоки окна w для хостов, которые оцениваются в этом окне (блоки с весом 0 не включаются)."""
        k = {"peers": min(p["k_peers"], self.kmax["peers"]), "services": min(p["k_services"], self.kmax["services"])}
        wt = {"behav": 1.0, "peers": p["w_peers"], "services": p["w_services"], "gnn": p["w_gnn"]}
        out = {}
        for b in ORDER:
            if wt[b] <= 0:
                continue
            tr, x = self.X["train"][b], self.X[w][b]
            if b in k:
                tr, x = tr[:, :k[b]], x[:, :k[b]]
            if self.center:
                mu = tr.astype(np.float64).mean(0)
                tr, x = tr - mu, x - mu
            scale = np.sqrt((tr.astype(np.float64) ** 2).sum(1).mean()) + 1e-9
            out[b] = (wt[b] / scale) * x[self.ev[w]]
        return out

    def embed(self, w: str, p: dict) -> pd.DataFrame:
        parts = self.parts(w, p)
        z = np.hstack([parts[b] for b in ORDER if b in parts]).astype(np.float32)
        zero = ~np.isfinite(z).all(1) | (np.abs(z).sum(1) == 0)
        if zero.any():                                    # evaluate не принимает нулевые векторы
            z[zero] = np.random.default_rng(0).normal(size=(int(zero.sum()), z.shape[1])).astype(np.float32) * 1e-3
        return pd.DataFrame(z, index=pd.Index(self.hosts[self.ev[w]], name="host_id")).add_prefix("emb_")

    def shares(self, w: str, p: dict) -> dict[str, float]:
        sq = {b: float((x.astype(np.float64) ** 2).sum(1).mean()) for b, x in self.parts(w, p).items()}
        tot = sum(sq.values())
        return {b: v / tot for b, v in sq.items()}

    def write(self, d: Path, p: dict, windows: list[str]) -> None:
        d.mkdir(parents=True, exist_ok=True)
        for w in windows:
            self.embed(w, p).to_csv(d / ("embeddings.csv" if w == "train" else f"{w}.csv"))


VAL_NEIGHBORS_IDEAL = 0.2112      # reference_neighbor_stability consistency@10 окна validation (scoreboard/data/reference/validation/manifest.json) актуально на 01.10 - константа на случай, если кто-то в очередной раз что-то поменяет, никому не сказав, и значение не будет читаться из файла


def val_score(D: Data, p: dict, tmp: Path) -> tuple[float, dict]:
    """Балл на validation. Отличие от submit.validation_score: идеал Neighbors для validation — 0.2112
    (устойчивость поведения именно в этом окне), а не среднее по test-окнам (0.145).
    Иначе Behavior на validation упирается в 100, и подбор не видит нехватки поведения.
    Подсчет балла по старому сценарию: --official-val."""
    from evaluation import scoreboard, submit
    D.write(tmp, p, ["train", "validation"])
    emb, val = tmp / "embeddings.csv", tmp / "validation.csv"
    if D.official_val or not hasattr(scoreboard, "NEIGHBORS_IDEAL_DEFAULT"):
        return submit.validation_score(emb, val)
    ref = submit.DATA / "reference" / "validation"
    r = submit.run(emb, ref, "candidate", n_boot=0)
    submit.add_identity(r, emb, val, ref, n_boot=0)
    cols = {s["metric"]: s["title"] for s in r["summary"]}
    try:
        ideal = float(json.loads((ref / "manifest.json").read_text(encoding="utf-8"))["reference_neighbor_stability"]["consistency@10"])
    except Exception:
        ideal = VAL_NEIGHBORS_IDEAL
    old = scoreboard.NEIGHBORS_IDEAL_DEFAULT
    scoreboard.NEIGHBORS_IDEAL_DEFAULT = ideal
    try:
        s = float(scoreboard.overall_score([dict(r, method="candidate")], cols)["candidate"])
        tasks = {scoreboard.TASK_EN.get(k, k): v for k, v in scoreboard.task_scores(r, cols).items()}
    finally:
        scoreboard.NEIGHBORS_IDEAL_DEFAULT = old
    return s, tasks


def fmt_tasks(t: dict) -> str:
    return ", ".join(f"{k} {v:.0f}" for k, v in t.items())


CARD = """# {name}

## Кратко

Склейка четырёх описаний хоста: профиль поведения, "с кем общается" (TF-IDF), "какими сервисами пользуется" (TF-IDF) и эмбеддинг графовой нейросети (GraphSAGE с признаками рёбер). Веса блоков подобраны на validation.

## Архитектура

Вектор = [поведение ×1, собеседники ×{w_peers:.2f}, сервисы ×{w_services:.2f}, GNN ×{w_gnn:.2f}]. Каждый блок перед умножением приведён к среднему квадрату длины 1, поэтому доли блоков в векторе: {share_txt}. Собеседники и сервисы — логарифм числа потоков, TF-IDF по хостам, SVD ({k_peers} и {k_services} компонент). GNN — готовый эмбеддинг из `{gnn}`.

## Данные

Поля src, dst, dst_port, proto, счётчики. Словари, IDF, SVD и нейросеть обучаются на train-окне и применяются к остальным окнам с теми же осями координат.

## Обучение и подбор

Подбор весов блоков и длины SVD — Optuna ({trials} проб). Балл на validation по задачам: {abl_full}. Без GNN: {abl_nognn}. Без TF-IDF (поведение + GNN): {abl_notfidf}. Только поведение: {abl_behav}.

## Ограничения

Подбор сделан на одном validation-окне сразу после train. Блок GNN обучается отдельно и в склейке не дообучается. Направление связи в GNN не используется.
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=str(ROOT / "runs" / "blocks"))
    ap.add_argument("--gnn", required=True, help="папка отправки GNN, например runs/submissions/gnn_big")
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "submissions"))
    ap.add_argument("--name", default="gnn_tfidf")
    ap.add_argument("--trials", type=int, default=40, help="0 — не подбирать, взять значения --w-*/--k-*")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--w-peers", type=float, default=0.5)
    ap.add_argument("--w-services", type=float, default=0.5)
    ap.add_argument("--w-gnn", type=float, default=1.0)
    ap.add_argument("--k-peers", type=int, default=128)
    ap.add_argument("--k-services", type=int, default=64)
    ap.add_argument("--no-center", dest="center", action="store_false",
                    help="не вычитать среднее train из блоков (по умолчанию вычитается, как у hybrid_tfidf)")
    ap.add_argument("--official-val", dest="official_val", action="store_true",
                    help="считать балл validation стандартной submit.validation_score (идеал Neighbors как у test)")
    ap.add_argument("--behav-dir", dest="behav_dir", default=None,
                    help="блок поведения из готовых файлов (runs/blocks_ref от build_behavioral.py) вместо нашего профиля")
    ap.add_argument("--author", default="", help="логин на GitHub для файла AUTHOR (по желанию)")
    ap.add_argument("--score", action="store_true", help="после записи посчитать официальный балл проекта")
    ap.add_argument("--keep-result", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    D = Data(Path(a.blocks), Path(a.gnn), Path(a.processed), a.center, a.official_val,
             Path(a.behav_dir) if a.behav_dir else None)
    log(f"центрирование блоков (вычитание среднего train): {'да' if a.center else 'нет'}")
    if "validation" not in D.windows:
        raise SystemExit("нет окна validation в блоках — нужен runs/features/validation и runs/blocks/validation.npz")
    tmp = ROOT / "runs" / "tune_tmp_v3"
    p = dict(w_peers=a.w_peers, w_services=a.w_services, w_gnn=a.w_gnn, k_peers=a.k_peers, k_services=a.k_services)

    if a.trials > 0:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
        study = optuna.create_study(direction="maximize", study_name=a.name, load_if_exists=True,
                                    storage=f"sqlite:///{ROOT / 'runs' / 'optuna_v3.db'}",
                                    sampler=optuna.samplers.TPESampler(seed=a.seed))
        if not study.trials:
            study.enqueue_trial({"w_peers": 0.5, "w_services": 0.5, "w_gnn": 1.0, "k_peers": 128, "k_services": 64})
        done = len(study.trials)

        def objective(trial):
            q = dict(w_peers=trial.suggest_float("w_peers", 0.0, 3.0), w_services=trial.suggest_float("w_services", 0.0, 3.0),
                     w_gnn=trial.suggest_float("w_gnn", 0.0, 3.0), k_peers=trial.suggest_categorical("k_peers", K_PEERS),
                     k_services=trial.suggest_categorical("k_services", K_SERVICES))
            s, tasks = val_score(D, q, tmp)
            log(f"  проба {trial.number + 1}: балл {s:.2f} | w_peers {q['w_peers']:.2f} w_services {q['w_services']:.2f} "
                f"w_gnn {q['w_gnn']:.2f} k {q['k_peers']}/{q['k_services']} | {fmt_tasks(tasks)} | {time.time() - t0:.0f} с")
            return s

        log(f"подбор весов: {a.trials} проб (уже было {done}), каждая — запись validation + validation_score")
        study.optimize(objective, n_trials=a.trials)
        p = dict(study.best_params)
        log(f"лучшая проба: {study.best_value:.2f} {p}")
        trials = len(study.trials)
    else:
        trials = 0

    log("абляции на validation (вклад блоков):")
    variants = {"full": p, "nognn": {**p, "w_gnn": 0.0}, "notfidf": {**p, "w_peers": 0.0, "w_services": 0.0},
                "behav": {**p, "w_peers": 0.0, "w_services": 0.0, "w_gnn": 0.0}}
    abl, abl_txt = {}, {}
    for k, q in variants.items():
        s, tasks = val_score(D, q, tmp)
        abl[k] = s
        abl_txt[k] = f"{s:.1f} ({fmt_tasks(tasks)})"
        log(f"  {k:8s} {abl_txt[k]}")

    sub = Path(a.out) / a.name
    log(f"запись отправки {sub}")
    D.write(sub, p, D.windows)
    (sub / "SHARED_COORDINATES").touch()
    sh = D.shares("train", p)
    share_txt = ", ".join(f"{RUS[b]} {100 * v:.0f}%" for b, v in sh.items())
    (sub / "README.md").write_text(CARD.format(name=a.name, gnn=a.gnn, trials=trials, share_txt=share_txt, abl_full=abl_txt["full"],
                                              abl_nognn=abl_txt["nognn"], abl_notfidf=abl_txt["notfidf"], abl_behav=abl_txt["behav"], **p),
                                   encoding="utf-8")
    (sub / "params.json").write_text(json.dumps({**p, "w_behav": 1.0, "ablation_validation": abl}, indent=1), encoding="utf-8")
    if a.author:
        (sub / "AUTHOR").write_text(a.author + "\n", encoding="utf-8")

    bar = "=" * 14
    print(f"\n{bar} ВЕСА БЛОКОВ В ИТОГОВОМ ВЕКТОРЕ {bar}")
    dims = {"behav": D.X["train"]["behav"].shape[1], "peers": min(p["k_peers"], D.kmax["peers"]),
            "services": min(p["k_services"], D.kmax["services"]), "gnn": D.X["train"]["gnn"].shape[1]}
    wt = {"behav": 1.0, "peers": p["w_peers"], "services": p["w_services"], "gnn": p["w_gnn"]}
    wins = [w for w in ["train", "validation", "test", "test_day03", "test_day04"] if w in D.windows]
    shs = {w: D.shares(w, p) for w in wins}
    print(f"{'блок':24s}{'чисел':>6s}{'вес w':>8s}" + "".join(f"{w:>12s}" for w in wins) + "   (доля блока в квадрате длины вектора)")
    for b in ORDER:
        print(f"{RUS[b]:24s}{dims[b]:>6d}{wt[b]:>8.2f}" + "".join(f"{100 * shs[w].get(b, 0):>11.0f}%" for w in wins))
    print(f"Всего чисел в векторе: {sum(dims[b] for b in ORDER if wt[b] > 0)}")
    print("Балл на validation (по задачам) для разных наборов блоков — вклад каждого блока:")
    for k, name in [("full", "все блоки"), ("nognn", "без GNN"), ("notfidf", "поведение + GNN (без TF-IDF)"), ("behav", "только поведение")]:
        print(f"  {name:32s} {abl_txt[k]}")
    print(f"Подбор: {trials} проб; параметры в {sub / 'params.json'}; всего {time.time() - t0:.0f} с")
    print("=" * 83)
    if a.score:
        import gnn_v1
        log("оценка официальным кодом проекта (несколько минут) ...")
        gnn_v1.official_score(sub, a.keep_result)


if __name__ == "__main__":
    main()
