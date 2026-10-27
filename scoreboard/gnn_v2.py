"""GNN V2: граф «хост — сервис» (сервис = протокол + порт), по NEXT_STEPS.md §4 и Xiao et al., 3.1.

    python scoreboard/gnn_v2.py --epochs 5                              # быстрая проверка, что всё запускается
    python scoreboard/gnn_v2.py --score                                 # обучить и посчитать ИТОГОВЫЙ БАЛЛ проекта
    python scoreboard/gnn_v2.py --concat-k 16 --beta 1 --score          # то же со склейкой с профилем (как у V1.1)

Вход: runs/features/<окно>/{nodes.csv, services.csv} (их уже сделал build_features.py, менять ничего не нужно).
Аргументы общие с gnn_v1.py (get_parser оттуда): mu, lam, hold, tau, kneg, concat-k, beta, params-json, only-windows...
Не используется: --hub-pow (его роль у V2 играет idf).

Модель (хост-хост связей нет, один слой):
  у каждого сервиса обучаемый вектор E_s; сообщение хост<-сервис = ReLU(W [E_s, признаки связи]),
  признаки связи = log1p(потоки, байты отправлено/получено, пакеты отправлено/получено) + роль (0 — хост пользуется, 1 — предоставляет);
  для хоста отдельно усредняются сообщения по роли 0 и по роли 1 с весами log1p(потоки) * idf_s;
  h = ReLU(W_own x_хоста + W_a [agg_use, agg_serve]); вектор = normalize(head(h)).
Словарь сервисов строится ОДИН раз по train: порт PortN -> узел «anon» своего протокола; сервисы, которыми пользуются
меньше --min-hosts хостов -> узел «rare» своего протокола; сервис, которого не было в train, -> узел «unknown».
Обучение без меток: каждую эпоху скрывается hold% связей хост—сервис, по вектору хоста нужно угадать скрытый сервис
среди случайных (по популярности^0.75), плюс восстановление профиля хоста (вес lam); loss = mu * связи + lam * признаки.
ВАЖНО: порты в профиле хоста (use_p*, serve_p*) по умолчанию НЕ подаются на вход, иначе скрытая связь «подсказана» профилем
(профиль считан по всем потокам, включая скрытые). Они остаются целью восстановления. Вернуть: --keep-ports.
"""
from __future__ import annotations

import sys
import time
import types
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gnn_v1 as G                          # noqa: E402
from gnn_v1 import log                      # noqa: E402

SVC_COLS = ["flows", "bytes_sent", "bytes_recv", "pkts_sent", "pkts_recv"]


# ------------------------------------------------------------------ данные

def load_nodes(feat: Path, w: str) -> pd.DataFrame:
    return pd.read_csv(feat / w / "nodes.csv", dtype={"host_id": str}).set_index("host_id")


def load_services(feat: Path, w: str) -> pd.DataFrame:
    return pd.read_csv(feat / w / "services.csv", dtype={"host": str})


class Vocab:
    """Словарь сервисов по train: (proto, port) -> номер. 0 — неизвестный сервис."""

    def __init__(self, svc: pd.DataFrame, min_hosts: int):
        nh = svc.groupby(["proto", "port"]).host.nunique()
        self.ids: dict[tuple[int, int], int] = {}
        self.rare: dict[int, int] = {}
        nxt = 1
        for (p, q), n in nh.items():
            if int(q) == -1 or n >= min_hosts:                  # анонимный порт — всегда свой узел «anon_<proto>»
                self.ids[(int(p), int(q))] = nxt
                nxt += 1
        for p in sorted({int(p) for (p, q), n in nh.items() if int(q) != -1 and n < min_hosts}):
            self.rare[p] = nxt                                  # «rare_<proto>»
            nxt += 1
        self.size = nxt

    def map(self, svc: pd.DataFrame) -> np.ndarray:
        out = np.zeros(len(svc), dtype=np.int64)
        for i, (p, q) in enumerate(zip(svc["proto"].to_numpy(), svc["port"].to_numpy())):
            s = self.ids.get((int(p), int(q)))
            out[i] = s if s is not None else self.rare.get(int(p), 0)
        return out


def prepare(svc: pd.DataFrame, vocab: Vocab, hosts: pd.Index) -> pd.DataFrame:
    """Таблица (h, role, sid) + суммы по SVC_COLS: сервисы приведены к словарю, одинаковые склеены."""
    h = hosts.get_indexer(svc["host"])
    svc = svc[h >= 0].reset_index(drop=True)
    d = svc[SVC_COLS].copy()
    d["h"], d["role"], d["sid"] = h[h >= 0], svc["role"].to_numpy(), vocab.map(svc)
    return d.groupby(["h", "role", "sid"], sort=False)[SVC_COLS].sum().reset_index()


def make_stats(nodes: pd.DataFrame, cols_all: list[str], agg: pd.DataFrame, size: int, use_idf: bool) -> dict:
    x = nodes[cols_all].to_numpy(float)
    f = np.log1p(agg[SVC_COLS].to_numpy(float))
    cnt = agg.groupby("sid").h.nunique().reindex(range(size)).fillna(0).to_numpy()
    idf = np.clip(np.log(agg.h.nunique() / (1.0 + cnt)), 0.05, None)
    idf[cnt == 0] = idf[cnt > 0].mean()
    if not use_idf:
        idf[:] = 1.0
    pop = np.bincount(agg["sid"].to_numpy(), minlength=size).astype(float) ** 0.75
    return {"xm": x.mean(0), "xs": x.std(0) + 1e-6, "fm": f.mean(0), "fs": f.std(0) + 1e-6,
            "idf": torch.tensor(idf, dtype=torch.float32), "pop": torch.tensor(pop, dtype=torch.float32)}


class SG:
    """Граф «хост — сервис» одного окна."""

    def __init__(self, nodes: pd.DataFrame, agg: pd.DataFrame, st: dict, cols_all: list[str], in_idx: list[int]):
        self.hosts, self.n = nodes.index, len(nodes)
        x = np.clip(np.nan_to_num((nodes[cols_all].to_numpy(float) - st["xm"]) / st["xs"]), -5, 5)
        self.x_all = torch.tensor(x, dtype=torch.float32)
        self.x_in = self.x_all[:, in_idx]
        self.cohort = torch.tensor(nodes["in_cohort"].to_numpy(bool))
        self.h = torch.tensor(agg["h"].to_numpy(), dtype=torch.long)
        self.sid = torch.tensor(agg["sid"].to_numpy(), dtype=torch.long)
        self.R = len(agg)
        f = np.clip((np.log1p(agg[SVC_COLS].to_numpy(float)) - st["fm"]) / st["fs"], -5, 5)
        self.ef = torch.tensor(np.column_stack([f, agg["role"].to_numpy()]), dtype=torch.float32)
        self.w = torch.log1p(torch.tensor(agg["flows"].to_numpy(), dtype=torch.float32)) * st["idf"][self.sid]
        self.grp = self.h * 2 + torch.tensor(agg["role"].to_numpy(), dtype=torch.long)

    def mp(self, keep: torch.Tensor | None = None):
        sid, ef, w, grp = self.sid, self.ef, self.w, self.grp
        if keep is not None:
            sid, ef, w, grp = sid[keep], ef[keep], w[keep], grp[keep]
        s = torch.zeros(self.n * 2).index_add_(0, grp, w)
        return sid, ef, w / s[grp].clamp_min(1e-9), grp


# ------------------------------------------------------------------ модель

class Model(nn.Module):
    def __init__(self, d_in: int, d_full: int, n_svc: int, d_service: int, hidden: int, dim: int, dropout: float):
        super().__init__()
        self.emb = nn.Embedding(n_svc, d_service)               # E_s: вход сообщений
        self.emb_out = nn.Embedding(n_svc, dim)                 # E'_s: выход, для предсказания скрытых связей
        self.msg = nn.Linear(d_service + 6, hidden)
        self.own = nn.Linear(d_in, hidden)
        self.agg = nn.Linear(2 * hidden, hidden, bias=False)
        self.head = nn.Linear(hidden, dim)
        self.dec = nn.Sequential(nn.Linear(dim, hidden), nn.ReLU(), nn.Linear(hidden, d_full))
        self.dropout = dropout

    def forward(self, g: SG, keep: torch.Tensor | None = None):
        sid, ef, alpha, grp = g.mp(keep)
        m = F.relu(self.msg(torch.cat([self.emb(sid), ef], 1))) * alpha.unsqueeze(1)
        agg = torch.zeros(g.n * 2, m.size(1)).index_add_(0, grp, m).view(g.n, -1)     # [хост: use | serve]
        h = F.dropout(F.relu(self.own(g.x_in) + self.agg(agg)), self.dropout, self.training)
        return F.normalize(self.head(h), dim=1)


def train_model(g: SG, st: dict, size: int, d_in: int, a) -> Model:
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    model = Model(d_in, g.x_all.size(1), size, a.d_service, a.hidden, a.dim, a.dropout)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    p = st["pop"] / st["pop"].sum()
    log(f"обучение: {g.n:,} хостов, {g.R:,} связей хост—сервис, {size:,} сервисов, вход {d_in} признаков, {a.epochs} эпох")
    t0 = time.time()
    for ep in range(1, a.epochs + 1):
        model.train()
        held = torch.from_numpy(rng.random(g.R) < a.hold)
        z = model(g, ~held)
        pu, ps = g.h[held], g.sid[held]
        eo = F.normalize(model.emb_out.weight, dim=1)
        pos = (z[pu] * eo[ps]).sum(1) / a.tau
        neg_idx = torch.multinomial(p, len(pu) * a.kneg, replacement=True)
        neg = (z[pu].repeat_interleave(a.kneg, 0) * eo[neg_idx]).sum(1) / a.tau
        l_link = F.softplus(-pos).mean() + F.softplus(neg).mean()
        l_rec = ((model.dec(z) - g.x_all) ** 2)[g.cohort].mean()
        loss = a.mu * l_link + a.lam * l_rec
        opt.zero_grad()
        loss.backward()
        opt.step()
        if ep == 1 or ep % 10 == 0 or ep == a.epochs:
            log(f"  эпоха {ep:>4}/{a.epochs} | связи {l_link.item():.3f} | признаки {l_rec.item():.3f} | {time.time() - t0:.0f} с")
    return model


@torch.no_grad()
def embed(model: Model, g: SG) -> np.ndarray:
    model.eval()
    return model(g).numpy()


# ------------------------------------------------------------------ отправка и оценка

CARD = """# {name}

## Кратко

Граф-нейросеть над двудольным графом «хост — сервис» (сервис = протокол + порт назначения), по идее Xiao et al. (граф IP—порт). Вектор хоста собирается из обучаемых векторов сервисов, которыми хост пользуется и которые он предоставляет; хост–хост связей в модели нет.

## Архитектура

Один слой: сообщение от сервиса к хосту = ReLU(W [вектор сервиса ({d_service}), признаки связи: log потоков, байт и пакетов в обе стороны, роль]). Для хоста отдельно усредняются сообщения по роли «пользуется» и «предоставляет» (роль — по направлению записи: src пользуется, dst предоставляет) с весами log(1+потоков){idf_text}. Вектор = normalize(линейная голова от ReLU(W_own·профиль хоста + W_a·[агрегат use, агрегат serve])), длина {dim}.{concat_note}

## Данные

Только NetFlow LANL, окна как в проекте. Профиль хоста (66 признаков, порты в нём не подаются на вход, чтобы не подсказывать скрытые связи при обучении{keep_ports_note}) и таблица «хост — роль — сервис» считаются по потокам окна одной функцией (scoreboard/build_features.py). Словарь сервисов строится по train: анонимные порты (PortN) — один узел на протокол, сервисы реже чем у {min_hosts} хостов — узел «rare» протокола, незнакомые в другом окне — узел «unknown». Модель обучается только на train-окне; в других окнах те же веса применяются к их графу (общие координаты).

## Обучение и подбор

Без меток: каждую эпоху скрыто {hold:.0%} связей хост—сервис, по вектору хоста нужно угадать скрытый сервис среди {kneg} случайных (по популярности^0.75, температура {tau}), плюс восстановление профиля хоста (вес {lam}); вес связей mu {mu}. Эпох {epochs}, lr {lr}, hidden {hidden}, dropout {dropout}, зерно {seed}. {tuning_note}

## Ограничения

Роль сервиса определяется по направлению записи, а инициатор в LANL определяется неточно; в трети потоков порт анонимен, и сервис в них неразличим. Точное время внутри окна не используется (только шесть отрезков в профиле). Хост–хост структура не учитывается.{limits_note} Результат зависит от зерна обучения.
"""


def build_submission(a) -> Path:
    feat, proc = Path(a.features), Path(a.processed)
    cohort = pd.read_csv(proc / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    windows = sorted(d.name for d in feat.iterdir() if d.is_dir() and (d / "services.csv").exists())
    if "train" not in windows:
        raise SystemExit(f"нет {feat}/train — сначала запустите build_features.py")
    if a.only_windows:
        windows = [w for w in windows if w == "train" or w in a.only_windows]

    log("[1/3] читаю признаки train-окна")
    nodes, svc = load_nodes(feat, "train"), load_services(feat, "train")
    cols_all = [c for c in nodes.columns if c != "in_cohort"]
    cols_in = cols_all if a.keep_ports else [c for c in cols_all if not c.startswith(("use_p", "serve_p"))]
    in_idx = [cols_all.index(c) for c in cols_in]
    vocab = Vocab(svc, a.min_hosts)
    agg = prepare(svc, vocab, nodes.index)
    st = make_stats(nodes, cols_all, agg, vocab.size, bool(a.idf))
    g = SG(nodes, agg, st, cols_all, in_idx)

    log("[2/3] обучение")
    model = train_model(g, st, vocab.size, len(cols_in), a)
    pca = G.fit_profile_pca(types.SimpleNamespace(x=g.x_all, cohort=g.cohort), a.concat_k) if a.concat_k > 0 else None

    log("[3/3] эмбеддинги окон")
    out = Path(a.out) / a.name
    out.mkdir(parents=True, exist_ok=True)

    def save(w: str, graph: SG, hosts: pd.Index, fname: str) -> None:
        z = G.with_profile(embed(model, graph), graph.x_all.numpy(), a, pca)
        pd.DataFrame(z[graph.hosts.get_indexer(hosts)], index=pd.Index(hosts, name="host_id")).add_prefix("emb_").to_csv(out / fname)
        log(f"      {w}: {len(hosts):,} хостов -> {fname}")

    save("train", g, cohort.index, "embeddings.csv")
    for w in windows:
        if w == "train":
            continue
        nw = load_nodes(feat, w).reindex(columns=["in_cohort"] + cols_all)
        gw = SG(nw, prepare(load_services(feat, w), vocab, nw.index), st, cols_all, in_idx)
        save(w, gw, cohort.index[cohort[f"evaluable_{w}"].astype(bool)], f"{w}.csv")
    (out / "SHARED_COORDINATES").touch()

    if a.concat_k > 0:
        concat_note = (f" К вектору приписана проекция профиля хоста на {a.concat_k} главных компонент (PCA по train-хостам когорты, "
                       f"вес {a.beta:.2f}); итоговая длина вектора {a.dim + a.concat_k}.")
        limits_note = (" Склейка с профилем сближает метод с поведенческим профилем; метрика «Соседи@10» построена из похожих "
                       "признаков, поэтому рост по ней частично объясняется этим. Вектор длиннее базовых 32.")
    else:
        concat_note = limits_note = ""
    tuning_note = (f"Параметры подобраны Optuna только по validation-эталону (файл {Path(a.params_json).name}), test в подборе "
                   "не участвовал." if a.params_json else "Параметры не подбирались (значения по умолчанию).")
    (out / "README.md").write_text(CARD.format(
        **vars(a), concat_note=concat_note, limits_note=limits_note, tuning_note=tuning_note,
        idf_text=" и редкостью сервиса (idf: общеизвестные порты весят меньше)" if a.idf else "",
        keep_ports_note=", порты на входе включены" if a.keep_ports else ""), encoding="utf-8")
    log(f"готово: {out}")
    return out


def get_parser():
    ap = G.get_parser()
    ap.set_defaults(name="gnn_v2")
    ap.add_argument("--d-service", dest="d_service", type=int, default=32, help="длина вектора сервиса")
    ap.add_argument("--min-hosts", dest="min_hosts", type=int, default=5, help="сервисы реже этого числа хостов -> «rare»")
    ap.add_argument("--idf", type=int, default=1, choices=[0, 1], help="1 — веса с idf сервиса, 0 — без")
    ap.add_argument("--keep-ports", dest="keep_ports", action="store_true", help="подавать порты профиля на вход (утечка при обучении)")
    return ap


def main() -> None:
    a = G.apply_params(get_parser().parse_args())
    G.QUIET = a.quiet
    t0 = time.time()
    sub = build_submission(a)
    if a.score:
        log("оценка официальным кодом проекта (это самый долгий шаг, несколько минут) ...")
        G.official_score(sub, a.keep_result)
    log(f"всего {time.time() - t0:.0f} с")


if __name__ == "__main__":
    main()
