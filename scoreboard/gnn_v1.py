"""GNN V1: граф «хост — хост», edge-aware GraphSAGE на профиле хоста и признаках рёбер из потоков.

    python scoreboard/gnn_v1.py --score            # обучить, сохранить отправку и посчитать ИТОГОВЫЙ БАЛЛ проекта
    python scoreboard/gnn_v1.py --epochs 5         # быстрая проверка, что всё запускается

Вход: runs/features/<окно>/{nodes.csv, edges.csv} (их делает build_features.py), runs/processed/cohort.csv.

Модель (одни и те же веса применяются к графу любого окна, поэтому координаты у окон общие):
  слой: сообщение от соседа j к хосту i = ReLU(W [вектор j, признаки ребра ij]); сообщения усредняются с весами
        log(1+потоков) / (степень соседа+1)^hub_pow  (hub_pow>0 ослабляет хабы); новый вектор = ReLU(W_s h_i + W_a agg_i)
  два слоя -> линейная голова -> вектор длины dim, нормированный до единичной длины.
Обучение без меток, две задачи сразу:
  1) предсказание связей: каждую эпоху hold% рёбер скрываются от модели и должны быть «узнаны» среди случайных пар;
  2) восстановление признаков хоста из его вектора (вес lam) — чтобы вектор нёс профиль, а не только положение в графе.
Результат: runs/submissions/<имя>/ (embeddings.csv, validation.csv, test*.csv, SHARED_COORDINATES, README.md).
С флагом --score считается официальный балл (evaluation.submit.score); файлы scoreboard/README.md и scoreboard.csv
НЕ перезаписываются, служебный файл результата удаляется (оставить: --keep-result).
Нужен torch (CPU).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


QUIET = False        # True — не печатать ход обучения (нужно для Optuna, чтобы не засорять вывод)


def log(msg: str) -> None:
    if not QUIET:
        print(msg, flush=True)


# ------------------------------------------------------------------ данные

def load_window(feat_dir: Path, w: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    nodes = pd.read_csv(feat_dir / w / "nodes.csv", dtype={"host_id": str}).set_index("host_id")
    edges = pd.read_csv(feat_dir / w / "edges.csv", dtype={"a": str, "b": str})
    return nodes, edges


def edge_matrix(e: pd.DataFrame) -> np.ndarray:
    """Симметричные признаки ребра (направление не используем: инициатор в LANL определяется неточно)."""
    fl = e["flows"].to_numpy(float)
    return np.column_stack([np.log1p(fl), np.log1p(e["bytes"].to_numpy(float)), np.log1p(e["packets"].to_numpy(float)),
                            e["flows_tcp"].to_numpy(float) / fl, e["flows_udp"].to_numpy(float) / fl,
                            e["main_port_share"].to_numpy(float), np.log1p(e["n_services"].to_numpy(float))])


def make_norm(nodes: pd.DataFrame, edges: pd.DataFrame, cols: list[str]) -> dict:
    x, e = nodes[cols].to_numpy(float), edge_matrix(edges)
    return {"xm": x.mean(0), "xs": x.std(0) + 1e-6, "em": e.mean(0), "es": e.std(0) + 1e-6}


class Graph:
    def __init__(self, nodes: pd.DataFrame, edges: pd.DataFrame, norm: dict, cols: list[str]):
        self.hosts = nodes.index
        self.n = len(self.hosts)
        x = np.nan_to_num((nodes[cols].to_numpy(float) - norm["xm"]) / norm["xs"])
        self.x = torch.tensor(np.clip(x, -5, 5), dtype=torch.float32)
        self.cohort = torch.tensor(nodes["in_cohort"].to_numpy(bool))
        a, b = self.hosts.get_indexer(edges["a"]), self.hosts.get_indexer(edges["b"])
        if (a < 0).any() or (b < 0).any():
            raise ValueError("в edges.csv есть хосты, которых нет в nodes.csv")
        self.E = len(a)
        self.pairs = (torch.tensor(a), torch.tensor(b))
        ef = np.clip(np.nan_to_num((edge_matrix(edges) - norm["em"]) / norm["es"]), -5, 5)
        self.src = torch.tensor(np.concatenate([a, b]))            # первые E рёбер a->b, следующие E: b->a
        self.dst = torch.tensor(np.concatenate([b, a]))
        self.ef = torch.tensor(np.concatenate([ef, ef]), dtype=torch.float32)
        fl = edges["flows"].to_numpy(float)
        self.flows = torch.tensor(np.concatenate([fl, fl]), dtype=torch.float32)
        self.deg = torch.tensor(np.bincount(np.concatenate([a, b]), minlength=self.n), dtype=torch.float32)

    def mp(self, hub_pow: float, keep: torch.Tensor | None = None):
        src, dst, ef, fl = self.src, self.dst, self.ef, self.flows
        if keep is not None:
            src, dst, ef, fl = src[keep], dst[keep], ef[keep], fl[keep]
        w = torch.log1p(fl) / (self.deg[src] + 1.0) ** hub_pow
        s = torch.zeros(self.n).index_add_(0, dst, w)
        return src, dst, ef, w / s[dst].clamp_min(1e-9)


# ------------------------------------------------------------------ модель

class Layer(nn.Module):
    def __init__(self, d_in: int, d_edge: int, d_out: int):
        super().__init__()
        self.msg = nn.Linear(d_in + d_edge, d_out)
        self.own = nn.Linear(d_in, d_out)
        self.agg = nn.Linear(d_out, d_out, bias=False)

    def forward(self, h, src, dst, ef, alpha):
        m = F.relu(self.msg(torch.cat([h[src], ef], 1))) * alpha.unsqueeze(1)
        agg = torch.zeros(h.size(0), m.size(1)).index_add_(0, dst, m)
        return F.relu(self.own(h) + self.agg(agg))


class Model(nn.Module):
    def __init__(self, d_node: int, d_edge: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.l1, self.l2 = Layer(d_node, d_edge, hidden), Layer(hidden, d_edge, hidden)
        self.head = nn.Linear(hidden, dim)
        self.dec = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, d_node))
        self.dropout = dropout

    def forward(self, x, mp):
        src, dst, ef, alpha = mp
        h = F.dropout(self.l1(x, src, dst, ef, alpha), self.dropout, self.training)
        return F.normalize(self.head(self.l2(h, src, dst, ef, alpha)), dim=1)


def train_model(g: Graph, a: argparse.Namespace) -> Model:
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = Model(g.x.size(1), g.ef.size(1), a.hidden, a.dim, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    p = g.deg ** 0.75
    p = p / p.sum()
    pa, pb = g.pairs
    log(f"обучение: {g.n:,} хостов, {g.E:,} связей, признаков хоста {g.x.size(1)}, признаков ребра {g.ef.size(1)}, {a.epochs} эпох")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        held = torch.from_numpy(rng.random(g.E) < a.hold)
        z = model(g.x, g.mp(a.hub_pow, ~torch.cat([held, held])))
        pu, pv = pa[held], pb[held]
        pos = (z[pu] * z[pv]).sum(1) / a.tau
        neg_idx = torch.multinomial(p, len(pu) * a.kneg, replacement=True)
        neg = (z[pu].repeat_interleave(a.kneg, 0) * z[neg_idx]).sum(1) / a.tau
        l_link = F.softplus(-pos).mean() + F.softplus(neg).mean()
        l_rec = ((model.dec(z) - g.x) ** 2)[g.cohort].mean()
        loss = a.mu * l_link + a.lam * l_rec
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == 1 or ep % 10 == 0 or ep == a.epochs:
            log(f"  эпоха {ep:>4}/{a.epochs} | связи {l_link.item():.3f} | признаки {l_rec.item():.3f} | {time.time() - t0:.0f} с")
    return model


@torch.no_grad()
def embed(model: Model, g: Graph, hub_pow: float) -> np.ndarray:
    model.eval()
    return model(g.x, g.mp(hub_pow)).numpy()


def fit_profile_pca(g: Graph, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PCA профиля хоста по train-хостам когорты: среднее и матрица проекции (признаков x k)."""
    x = g.x[g.cohort].numpy().astype(np.float64)
    mu_x = x.mean(0)
    _, _, vt = np.linalg.svd(x - mu_x, full_matrices=False)
    return mu_x, vt[:k].T


def with_profile(z: np.ndarray, x: np.ndarray, a: argparse.Namespace, pca) -> np.ndarray:
    """normalize(concat([z, beta * normalize(PCA профиля)])). Без PCA (concat_k = 0) возвращает z как есть."""
    if pca is None:
        return z
    mu_x, proj = pca
    pn = (x.astype(np.float64) - mu_x) @ proj
    pn /= np.maximum(np.linalg.norm(pn, axis=1, keepdims=True), 1e-9)
    out = np.concatenate([z, a.beta * pn], axis=1)
    return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-9)


# ------------------------------------------------------------------ отправка и оценка

CARD = """# {name}

## Кратко

Граф-нейросеть (GraphSAGE с учётом признаков рёбер) над графом «хост — хост». Хост описывается профилем из потоков (порты, протоколы, объём, активность по отрезкам окна), а вектор строится из профиля хоста и профилей его собеседников.

## Архитектура

Два слоя: сообщение от соседа строится из пары (вектор соседа, признаки ребра: log потоков, байт, пакетов, доли TCP и UDP, доля главного порта, число сервисов), сообщения усредняются с весами log(1+потоков), ослабленными для хабов (степень {hub_pow}). Выход — вектор длины {dim}, нормированный до единичной длины. Направление связи не используется: инициатор в LANL определяется неточно.{concat_note}

## Данные

Только NetFlow LANL, окна как в проекте. Признаки хоста (66 штук) и рёбер считаются по потокам окна одной и той же функцией (scoreboard/build_features.py), порты берутся dst_port, 20 самых частых портов — по train. Модель обучается только на train-окне; для остальных окон те же веса применяются к графу окна (общие координаты).

## Обучение и подбор

Без меток: предсказание скрытых связей ({hold:.0%} рёбер за эпоху, {kneg} случайных пар на связь, температура {tau}) плюс восстановление признаков хоста из вектора (вес {lam}). Эпох {epochs}, lr {lr}, hidden {hidden}, dropout {dropout}, вес связей mu {mu}, зерно {seed}. {tuning_note}

## Ограничения

Не использует направление связи и точное время внутри окна (только шесть отрезков). Хабы остаются проблемой. Число собеседников у хостов вне когорты неполное. Результат зависит от зерна обучения.{limits_note}
"""


def build_submission(a: argparse.Namespace) -> Path:
    feat, proc = Path(a.features), Path(a.processed)
    cohort = pd.read_csv(proc / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    windows = sorted(d.name for d in feat.iterdir() if d.is_dir() and (d / "nodes.csv").exists())
    if "train" not in windows:
        raise SystemExit(f"нет {feat}/train — сначала запустите build_features.py")

    if a.only_windows:
        windows = [w for w in windows if w == "train" or w in a.only_windows]
    log("[1/3] читаю признаки train-окна")
    nodes, edges = load_window(feat, "train")
    cols = [c for c in nodes.columns if c != "in_cohort"]
    norm = make_norm(nodes, edges, cols)
    g = Graph(nodes, edges, norm, cols)

    log("[2/3] обучение")
    model = train_model(g, a)

    pca = fit_profile_pca(g, a.concat_k) if a.concat_k > 0 else None
    log("[3/3] эмбеддинги окон")
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)

    def save(w: str, graph: Graph, hosts: pd.Index, fname: str) -> None:
        z = with_profile(embed(model, graph, a.hub_pow), graph.x.numpy(), a, pca)
        pd.DataFrame(z[graph.hosts.get_indexer(hosts)], index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(out / fname)
        log(f"      {w}: {len(hosts):,} хостов -> {fname}")

    save("train", g, cohort.index, "embeddings.csv")
    for w in windows:
        if w == "train":
            continue
        nw, ew = load_window(feat, w)
        gw = Graph(nw.reindex(columns=["in_cohort"] + cols), ew, norm, cols)
        hosts = cohort.index[cohort[f"evaluable_{w}"].astype(bool)]
        save(w, gw, hosts, f"{w}.csv")
    (out / "SHARED_COORDINATES").touch()
    if a.concat_k > 0:
        concat_note = (f" К вектору GNN приписана проекция профиля хоста на {a.concat_k} главных компонент (PCA по train-хостам "
                       f"когорты, вес {a.beta:.2f}); итоговая длина вектора {a.dim + a.concat_k}.")
        limits_note = (" Склейка с профилем сближает метод с поведенческим профилем; метрика «Соседи@10» построена из похожих "
                       "признаков, поэтому рост по ней частично объясняется этим. Вектор длиннее базовых 32.")
    else:
        concat_note = limits_note = ""
    tuning_note = (f"Параметры подобраны Optuna только по validation-эталону (файл {Path(a.params_json).name}), test в подборе "
                   "не участвовал." if a.params_json else "Параметры не подбирались (значения по умолчанию).")
    (out / "README.md").write_text(CARD.format(**vars(a), concat_note=concat_note, limits_note=limits_note,
                                               tuning_note=tuning_note), encoding="utf-8")
    log(f"готово: {out}")
    return out


def official_score(sub: Path, keep: bool) -> None:
    from evaluation import scoreboard, submit

    r = submit.score(sub)
    res_file = Path(submit.RESULTS) / f"{sub.name}.json"
    cols = {s["metric"]: s["title"] for s in r["summary"]}
    everyone = [r]
    try:
        everyone += [x for x in scoreboard.load_results(Path(submit.RESULTS)) if x["method"] != sub.name]
    except Exception as exc:                                    # таблица сравнения — вторично, не падаем
        log(f"  (не удалось загрузить результаты других методов: {exc})")
    overall = scoreboard.overall_score(everyone, cols).dropna().sort_values(ascending=False)
    tasks = scoreboard.task_scores(r, cols)
    if not keep:
        res_file.unlink(missing_ok=True)

    bar = "=" * 14
    print(f"\n{bar} КРАТКИЙ ИТОГ {bar}")
    mine = overall.get(sub.name)
    if mine is None:
        print("ИТОГОВЫЙ БАЛЛ: не посчитан (не хватает метрик, см. ниже)")
    else:
        print(f"ИТОГОВЫЙ БАЛЛ: {mine:.1f}   место {int((overall > mine).sum()) + 1} из {len(overall)} (по методам с полным баллом)")
    print("Баллы по задачам (0 — случайность, 100 — идеал): " + ", ".join(f"{k} {v:.0f}" for k, v in tasks.items()))
    print("Метрики (среднее по test-окнам):")
    for s in r["summary"]:
        v, c = s["value"], s.get("chance")
        print(f"  {s['title']:26s} {v:.3f}" + (f"   (случайно {c:.3f})" if c is not None else ""))
    print("Для сравнения, общий балл (на тех же эталонах):")
    for name, v in overall.head(6).items():
        print(f"  {'-> ' if name == sub.name else '   '}{name:36s} {v:5.1f}")
    for name in ("behavioral", "node2vec_services", "graphsage"):
        if name in overall.index and name not in overall.head(6).index:
            print(f"     {name:36s} {overall[name]:5.1f}")
    print("=" * 83)


def get_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default=str(ROOT / "runs" / "features"))
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "submissions"))
    ap.add_argument("--name", default="gnn_v1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=0.005)
    ap.add_argument("--wd", type=float, default=1e-5)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--tau", type=float, default=0.2)
    ap.add_argument("--hold", type=float, default=0.15)
    ap.add_argument("--kneg", type=int, default=5)
    ap.add_argument("--hub-pow", dest="hub_pow", type=float, default=0.5, help="ослабление хабов: 0 — нет, 1 — сильное")
    ap.add_argument("--lam", type=float, default=1.0, help="вес восстановления признаков")
    ap.add_argument("--mu", type=float, default=1.0, help="вес потери по связям (loss = mu * связи + lam * признаки)")
    ap.add_argument("--concat-k", dest="concat_k", type=int, default=0,
                    help="сколько главных компонент профиля приписать к вектору (0 — выключено, это чистый V1)")
    ap.add_argument("--beta", type=float, default=1.0, help="вес приписанного профиля (0 — как чистый V1)")
    ap.add_argument("--only-windows", dest="only_windows", nargs="+", default=None,
                    help="считать эмбеддинги только этих окон кроме train (для подбора: validation)")
    ap.add_argument("--params-json", dest="params_json", default=None,
                    help="файл с параметрами (например runs/best_params.json); перезаписывает одноимённые аргументы")
    ap.add_argument("--quiet", action="store_true", help="не печатать ход обучения")
    ap.add_argument("--score", action="store_true", help="после обучения посчитать официальный балл проекта")
    ap.add_argument("--keep-result", action="store_true", help="не удалять scoreboard/results/<имя>.json")
    return ap


def apply_params(a: argparse.Namespace) -> argparse.Namespace:
    if a.params_json:
        for k, v in json.loads(Path(a.params_json).read_text(encoding="utf-8")).items():
            if not hasattr(a, k):
                raise SystemExit(f"в {a.params_json} неизвестный параметр: {k}")
            setattr(a, k, v)
    return a


def main() -> None:
    global QUIET
    a = apply_params(get_parser().parse_args())
    QUIET = a.quiet
    t0 = time.time()
    sub = build_submission(a)
    if a.score:
        log("оценка официальным кодом проекта (это самый долгий шаг, несколько минут) ...")
        official_score(sub, a.keep_result)
    log(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
