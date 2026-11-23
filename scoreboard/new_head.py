"""Голова «новые собеседники»: небольшой блок эмбеддинга, обученный напрямую под метрику «Новые собеседники, MRR».

    # 1) только подготовка и базовая оценка, без обучения (torch не нужен): MRR по входному вектору окон дня 04
    python -u scoreboard/new_head.py --holdout d04 --steps 0
    # 2) проверка идеи: обучение на всех окнах, кроме d04, и честный MRR на d04
    python -u scoreboard/new_head.py --holdout d04 --exclude d02 --steps 3000
    # 3) итог: обучение на всех (кроме d02), запись блока и склейка с готовым кросс-дневным блоком
    python -u scoreboard/new_head.py --holdout none --exclude d02 --steps 3000 --name xh_1 \
        --merge runs/submissions/xt__2rand38_s5 --head-weight 1.0

Как оценивается метрика (evaluation/partners.py): для хоста A берутся 10 ближайших по эмбеддингу хостов (вес = косинус),
собираются их собеседники в train-окне (день 02, 11:00-12:00), и кандидаты ранжируются по сумме «вес соседа x число контактов».
Смотрят, где в этом списке оказались НОВЫЕ собеседники A в целевом окне (те, кого у A не было в train).
Значит, всё зависит от того, КТО попал в соседи. Голова учится выбирать таких соседей.

Обучение. Для окна W (любой старый день, он же только для обучения весов) вход - тот же вектор хоста, что в cross_day.py
(по одному часу окна). MLP даёт 64 числа. Мягкие соседи: w_ij = softmax_j(cos(z_i, z_j) / tau) среди хостов, активных в окне W.
Доля массы соседских собеседников, попавшая на новых собеседников i:
    num_i = sum_j w_ij * a_ij,   a_ij = sum_{p in New_i} M_jp
    den_i = sum_j w_ij * (r_j - o_ij),  r_j = sum_p M_jp,  o_ij = sum_{p in Old_i} M_jp  (старых собеседников i оценка выкидывает)
    loss  = -mean_i log(num_i / den_i)
M - матрица «хост x собеседник» по train-окну. Матрицы a и (r - o) считаются один раз на окно.
Вектор хоста всё равно строится по одному часу окна; train (день 02) - только источник собеседников, дни 14-16 не используются.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scoreboard"))
from cross_day import load_x, log, norm, top1                                    # noqa: E402

# окно "test" каждого набора processed_multi* -> метка в runs/blocks_all (old_<метка>); совпадает с night.py
TEST_TAG = {"processed_multi2": "d02h14", "processed_multi3": "d05h12", "processed_multi4": "d07h12",
            "processed_multi5": "d08h12", "processed_multi6": "d09h12"}


def discover(root: Path) -> dict[str, Path]:
    """метка окна (как в old_<метка>) -> csv потоков."""
    found: dict[str, Path] = {}
    for d in sorted((root / "runs").glob("processed_multi*")):
        mp = d / "manifest.json"
        if not mp.exists():
            continue
        for w in json.loads(mp.read_text(encoding="utf-8"))["windows"]:
            if w in ("train", "validation"):
                continue
            tag = w[len("test_"):] if w.startswith("test_d0") else (TEST_TAG.get(d.name) if w == "test" else None)
            p = d / "flows" / f"{w}.csv"
            if tag and p.exists():
                found[tag] = p
    # старая нарезка окон 12:30-13:30 дней 02-04: ищем в известных местах и проверяем границы окна по манифесту (чтобы не подцепить чужое)
    expect = {"test": ("d02", 131400.0), "test_day03": ("d03", 217800.0), "test_day04": ("d04", 304200.0)}
    cands = sorted((root / "runs").glob("processed_old*")) + sorted((root / "runs").glob("old_processed*")) + sorted((root / "runs" / "old").glob("processed*"))
    for d in cands:
        mp = d / "manifest.json"
        if not mp.exists():
            continue
        wins = json.loads(mp.read_text(encoding="utf-8"))["windows"]
        for w, (tag, start) in expect.items():
            f = d / "flows" / f"{w}.csv"
            if f.exists() and w in wins and float(wins[w]["start"]) == start:
                found[tag] = f
    return found


def count_pairs(path: Path, host_idx: pd.Index, univ: pd.Index | None, chunk: int = 2_000_000):
    """Счётчики «хост когорты -> собеседник» (связь ненаправленная, петли выкинуты) по потокам окна.
    univ None -> собеседники строкой (для train, из них строится словарь кандидатов); иначе индексы в univ (чужие отбрасываются)."""
    parts = []
    for ch in pd.read_csv(path, usecols=["src", "dst"], dtype=str, chunksize=chunk):
        s, d = ch["src"].to_numpy(), ch["dst"].to_numpy()
        k = s != d
        s, d = s[k], d[k]
        h, p = np.concatenate([s, d]), np.concatenate([d, s])
        hi = host_idx.get_indexer(h)
        ok = hi >= 0
        hi, p = hi[ok], p[ok]
        if univ is None:
            parts.append(pd.DataFrame({"h": hi, "p": p}).groupby(["h", "p"]).size().rename("c").reset_index())
        else:
            pi = univ.get_indexer(p)
            ok = pi >= 0
            key = hi[ok].astype(np.int64) * len(univ) + pi[ok]
            u, c = np.unique(key, return_counts=True)
            parts.append(pd.DataFrame({"k": u, "c": c}))
    df = pd.concat(parts, ignore_index=True)
    if univ is None:
        return df.groupby(["h", "p"], as_index=False)["c"].sum()
    g = df.groupby("k", as_index=False)["c"].sum()
    return pd.DataFrame({"h": g["k"].to_numpy() // len(univ), "p": g["k"].to_numpy() % len(univ), "c": g["c"].to_numpy()})


def mval(c: np.ndarray, mode: str) -> np.ndarray:
    c = c.astype(np.float64)
    return np.log1p(c) if mode == "log" else (np.sqrt(c) if mode == "sqrt" else (c if mode == "raw" else np.ones_like(c)))


def build_window(pairs: pd.DataFrame, act: np.ndarray, M: sparse.csr_matrix, Bin: sparse.csr_matrix, r: np.ndarray,
                 U: int, key_train: np.ndarray):
    """Для окна: хосты с новыми собеседниками и две плотные матрицы (хост x сосед): a и (r - o), float16."""
    h, p = pairs["h"].to_numpy(), pairs["p"].to_numpy()
    new = ~np.isin(h.astype(np.int64) * U + p, key_train, assume_unique=False) & act[h]
    h, p = h[new], p[new]
    rows = np.unique(h)
    if len(rows) == 0:
        return None
    pos = np.searchsorted(rows, h)
    T = sparse.csr_matrix((np.ones(len(h)), (pos, p)), shape=(len(rows), U))
    A = (T @ M.T).toarray()
    O = (Bin[rows] @ M.T).toarray()
    D = np.clip(r[None, :] - O, 0.0, None)
    return rows, A.astype(np.float16), D.astype(np.float16), int(len(h))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=str(ROOT / "runs" / "blocks_all"))
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"), help="папка с train и cohort.csv")
    ap.add_argument("--cache", default=str(ROOT / "runs" / "new_head_cache"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "submissions"))
    ap.add_argument("--name", default="xh")
    ap.add_argument("--holdout", default="d04", help="окна с этим префиксом не участвуют в обучении, на них считается MRR; none - итоговый режим")
    ap.add_argument("--exclude", nargs="*", default=[], help="префиксы окон, которые не брать в обучение (например d02)")
    ap.add_argument("--k-peers", type=int, default=512)
    ap.add_argument("--k-services", type=int, default=256)
    ap.add_argument("--rp-peers", type=int, default=512)
    ap.add_argument("--rp-services", type=int, default=256)
    ap.add_argument("--w-peers", type=float, default=0.5)
    ap.add_argument("--w-services", type=float, default=0.3)
    ap.add_argument("--w-rp-peers", type=float, default=1.0)
    ap.add_argument("--w-rp-services", type=float, default=0.2)
    ap.add_argument("--out-dim", type=int, default=64)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--steps", type=int, default=3000, help="0 - только подготовка и базовая оценка (torch не нужен)")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--id-weight", type=float, default=0.0, help="вес второй цели: узнавание хоста между окнами (InfoNCE, как у cross_day.py); 0 - без неё. "
                                                                  "Нужна, чтобы голова не портила Узнаваемость и Поведение при большом весе в склейке")
    ap.add_argument("--id-tau", type=float, default=0.1)
    ap.add_argument("--id-batch", type=int, default=512)
    ap.add_argument("--m-mode", choices=["log", "sqrt", "raw", "bin"], default="log", help="как превращать число контактов соседа в вес")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--eval-windows", type=int, default=3, help="на скольких контрольных окнах считать MRR во время обучения (последний замер - на всех)")
    ap.add_argument("--merge", default="", help="папка готового блока (xt_*); голова приклеивается к его эмбеддингам")
    ap.add_argument("--remerge", default="", help="папка <имя>_only от прошлого итогового запуска: только склейка с --merge другим весом, без обучения")
    ap.add_argument("--head-weight", type=float, default=1.0, help="вес головы относительно блока --merge (у блока длина строки 1)")
    a = ap.parse_args()
    t0 = time.time()
    blocks, root = Path(a.blocks), Path(a.processed)
    if a.remerge:                                                           # только склейка готовой головы с блоком --merge
        if not a.merge:
            raise SystemExit("--remerge требует --merge")
        sub = Path(a.out) / a.name
        sub.mkdir(parents=True, exist_ok=True)
        for f in sorted(Path(a.remerge).glob("*.csv")):
            head = pd.read_csv(f, index_col="host_id", dtype={"host_id": str}) * a.head_weight
            base = pd.read_csv(Path(a.merge) / f.name, index_col="host_id", dtype={"host_id": str})
            df = base.join(head, how="inner")
            if len(df) != len(base):
                raise SystemExit(f"{f.name}: хосты блока и головы не совпали")
            df.to_csv(sub / f.name, float_format="%.5g")
            log(f"записано {sub / f.name} ({len(df):,} хостов, {df.shape[1]} чисел)")
        return

    cohort = pd.read_csv(root / "cohort.csv", dtype={"host_id": str})
    hosts = pd.Index(cohort["host_id"])
    N = len(hosts)
    cache = Path(a.cache)
    cache.mkdir(parents=True, exist_ok=True)

    # --- train-окно: источник собеседников и словарь кандидатов ---
    tpath = root / "flows" / "train.csv"
    cf = cache / "train_pairs.pkl"
    if cf.exists() and cf.stat().st_mtime > tpath.stat().st_mtime:
        trp = pd.read_pickle(cf)
    else:
        log(f"считаю собеседников train-окна из {tpath} (один раз) ...")
        trp = count_pairs(tpath, hosts, None)
        trp.to_pickle(cf)
    univ = pd.Index(sorted(trp["p"].unique()))
    U = len(univ)
    trp = pd.DataFrame({"h": trp["h"].to_numpy(), "p": univ.get_indexer(trp["p"]), "c": trp["c"].to_numpy()})
    mv = mval(trp["c"].to_numpy(), a.m_mode)
    M = sparse.csr_matrix((mv, (trp["h"], trp["p"])), shape=(N, U))
    Bin = sparse.csr_matrix((np.ones(len(trp)), (trp["h"], trp["p"])), shape=(N, U))
    r = np.asarray(M.sum(1)).ravel()
    key_train = np.sort(trp["h"].to_numpy().astype(np.int64) * U + trp["p"].to_numpy())
    log(f"train: хостов {N:,}, кандидатов-собеседников {U:,}, пар {len(trp):,} (вес контактов: {a.m_mode}) [{time.time() - t0:.0f} с]")

    # --- окна для обучения/контроля: векторы хостов и цели ---
    found = discover(ROOT)
    tags = sorted(p.stem[4:] for p in blocks.glob("old_*.npz"))
    kw = dict(kp=a.k_peers, ks=a.k_services, wp=a.w_peers, ws=a.w_services, rp=a.rp_peers, rs=a.rp_services,
              wrp=a.w_rp_peers, wrs=a.w_rp_services)
    xtr, act_tr, sc, hosts0 = load_x(blocks, None, "train", "train", sc=None, **kw)
    if list(hosts) != hosts0:
        raise SystemExit("порядок хостов в блоках не совпадает с cohort.csv")
    xs, acts, win, pdf = {}, {}, {}, {}
    for tag in tags:
        if tag not in found:
            log(f"окно {tag}: нет потоков (processed_multi*/runs/old/processed*) — пропущено")
            continue
        x, act, _, hs = load_x(blocks, None, f"old_{tag}", tag, sc=sc, **kw)
        if hs != hosts0:
            raise SystemExit(f"{tag}: порядок хостов отличается")
        cf = cache / f"pairs_{tag}.pkl"
        if cf.exists() and cf.stat().st_mtime > found[tag].stat().st_mtime:
            pr = pd.read_pickle(cf)
        else:
            t1 = time.time()
            pr = count_pairs(found[tag], hosts, univ)
            pr.to_pickle(cf)
            log(f"  окно {tag}: собеседники посчитаны из {found[tag].name} за {time.time() - t1:.0f} с")
        # хост в окне = активен по входному вектору
        w = build_window(pr, act, M, Bin, r, U, key_train)
        if w is None:
            log(f"окно {tag}: нет хостов с новыми собеседниками — пропущено")
            continue
        xs[tag], acts[tag], win[tag] = x, act, w
        pdf[tag] = pr
        log(f"окно {tag}: активных {int(act.sum()):,}; хостов с новыми собеседниками {len(w[0]):,} "
            f"(новых пар {w[3]:,}); a/D {2 * w[1].nbytes / 1e6:.0f} МБ")
    if not xs:
        raise SystemExit("нет ни одного окна с целями")

    def match(n, prefixes):
        return any(n.startswith(p) for p in prefixes)

    held_prefix = [] if a.holdout == "none" else [a.holdout]
    held = [n for n in xs if match(n, held_prefix)]
    names = [n for n in xs if n not in held and not match(n, a.exclude)]
    log(f"обучение: окна {names}; контроль: {held}")

    def partners_df(tag: str) -> pd.DataFrame:
        s = pd.DataFrame({"host_id": hosts[trp["h"].to_numpy()], "partner": univ[trp["p"].to_numpy()], "window": "source",
                          "count": trp["c"].to_numpy()})
        pr = pdf[tag]
        t = pd.DataFrame({"host_id": hosts[pr["h"].to_numpy()], "partner": univ[pr["p"].to_numpy()], "window": "target",
                          "count": pr["c"].to_numpy()})
        return pd.concat([s, t], ignore_index=True)

    from evaluation.partners import partner_ranking                        # официальный код метрики

    pcache: dict[str, pd.DataFrame] = {}

    def mrr_new(zfun, ts: list[str]) -> tuple[float, float]:
        new, allv = [], []
        for t in ts:
            if t not in pcache:
                pcache[t] = partners_df(t)
            idx = np.flatnonzero(acts[t])
            emb = pd.DataFrame(zfun(xs[t])[idx], index=pd.Index(hosts[idx], name="host_id"))
            res = partner_ranking(emb, pcache[t])
            new.append(res["new"]["embedding"]["mrr"] or 0.0)
            allv.append(res["all"]["embedding"]["mrr"] or 0.0)
        return float(np.mean(new)), float(np.mean(allv))

    if held:
        m, al = mrr_new(lambda x: x, held)
        log(f"БАЗА: MRR новых собеседников по входному вектору (косинус, без обучения) на {held}: {m:.4f}  (все собеседники: {al:.4f})")
    if a.steps == 0:
        log(f"только подготовка, {time.time() - t0:.0f} с")
        return

    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    d = next(iter(xs.values())).shape[1]
    model = nn.Sequential(nn.Linear(d, a.hidden), nn.GELU(), nn.Linear(a.hidden, a.out_dim))
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=1e-5)
    tx = {n: torch.from_numpy(xs[n]) for n in names}
    tmask = {n: torch.from_numpy(~acts[n]) for n in names}                  # не активные в окне хосты не могут быть соседями
    def encode(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            return norm(model(torch.from_numpy(x)).numpy())

    idw = {n: torch.from_numpy(xs[n]) for n in names}                       # окна для цели «узнавание»: обучающие + train
    idw["train"] = torch.from_numpy(xtr)
    id_acts = {**{n: acts[n] for n in names}, "train": act_tr}
    id_pairs = [(p, q) for p in idw for q in idw if p != q]
    id_both = {pq: np.flatnonzero(id_acts[pq[0]] & id_acts[pq[1]]) for pq in id_pairs} if a.id_weight > 0 else {}

    def identity_top1() -> float:
        ztr = encode(xtr)
        vals = [top1(ztr, encode(xs[h]), np.flatnonzero(act_tr & acts[h])) for h in held]
        return float(np.mean(vals))

    ev = held[: a.eval_windows]
    t1, tot, cnt, tid = time.time(), 0.0, 0, 0.0
    for step in range(1, a.steps + 1):
        model.train()
        n = names[rng.integers(len(names))]
        rows, A, D, _ = win[n]
        sel = rng.choice(len(rows), size=min(a.batch, len(rows)), replace=False)
        ids = rows[sel]
        z = F.normalize(model(F.dropout(tx[n], a.drop)), dim=1)
        sim = z[ids] @ z.T / a.tau
        selfm = torch.zeros(len(ids), N, dtype=torch.bool)
        selfm[torch.arange(len(ids)), torch.from_numpy(ids)] = True          # сам себе не сосед
        sim = sim.masked_fill(selfm | tmask[n][None, :], float("-inf"))
        w = torch.softmax(sim, dim=1)
        num = (w * torch.from_numpy(A[sel].astype(np.float32))).sum(1)
        den = (w * torch.from_numpy(D[sel].astype(np.float32))).sum(1)
        loss = -(torch.log(num + 1e-4) - torch.log(den + 1e-4)).mean()
        if a.id_weight > 0:
            pa, pb = id_pairs[rng.integers(len(id_pairs))]
            both = id_both[(pa, pb)]
            sid = rng.choice(both, size=min(a.id_batch, len(both)), replace=False)
            za = F.normalize(model(F.dropout(idw[pa][sid], a.drop)), dim=1)
            zb = F.normalize(model(F.dropout(idw[pb][sid], a.drop)), dim=1)
            lg = za @ zb.T / a.id_tau
            tg = torch.arange(len(sid))
            lid = (F.cross_entropy(lg, tg) + F.cross_entropy(lg.T, tg)) / 2
            tid += float(lid.detach())
            loss = loss + a.id_weight * lid
        opt.zero_grad()
        loss.backward()
        opt.step()
        tot += float(loss.detach())
        cnt += 1
        if step % 100 == 0 or step == 1:
            log(f"  шаг {step}/{a.steps}: loss {tot / cnt:.3f}" + (f" (из них узнавание {tid / cnt:.3f})" if a.id_weight > 0 else "") + f" ({time.time() - t1:.0f} с)")
            tot, cnt, tid = 0.0, 0, 0.0
        if held and (step % a.eval_every == 0 or step == a.steps):
            m, al = mrr_new(encode, held if step == a.steps else ev)
            log(f"  ШАГ {step}: MRR новых собеседников на контрольных окнах ({'все' if step == a.steps else len(ev)}): {m:.4f} (все собеседники {al:.4f})")
            log(f"  ШАГ {step}: узнавание Top-1 (train -> контрольные окна) головой: {identity_top1():.3f}")
    if held:
        log(f"всего {time.time() - t0:.0f} с. Это проверка; результат не записан (для записи: --holdout none)")
        return

    sub = Path(a.out) / a.name
    sub.mkdir(parents=True, exist_ok=True)
    only = Path(a.out) / f"{a.name}_only"                                  # голова отдельно, без веса: для --remerge
    only.mkdir(parents=True, exist_ok=True)
    mdir = Path(a.merge) if a.merge else None
    wins = ["train"] + sorted(p.stem for p in blocks.glob("*.npz") if not p.stem.startswith("old_") and p.stem != "train")
    for w in wins:
        x, _, _, _ = load_x(blocks, None, w, w, sc=sc, **kw)
        z0 = encode(x)
        mask = np.ones(N, bool) if w == "train" else cohort[f"evaluable_{w}"].to_numpy(bool)
        fn = "embeddings.csv" if w == "train" else f"{w}.csv"
        pd.DataFrame(z0[mask], index=pd.Index(hosts[mask], name="host_id")).add_prefix("head_").to_csv(only / fn, float_format="%.5g")
        z = z0 * (a.head_weight if mdir else 1.0)
        df = pd.DataFrame(z[mask], index=pd.Index(hosts[mask], name="host_id")).add_prefix("head_")
        if mdir:
            base = pd.read_csv(mdir / fn, index_col="host_id", dtype={"host_id": str})
            df = base.join(df, how="inner")
            if len(df) != len(base):
                raise SystemExit(f"{fn}: хосты блока {mdir} и головы не совпали")
        df.to_csv(sub / fn, float_format="%.5g")
        log(f"записано {sub / fn} ({len(df):,} хостов, {df.shape[1]} чисел)")
    log(f"готово за {time.time() - t0:.0f} с; папка {sub} — блок для gnn_combine_v3.py (--extra)")


if __name__ == "__main__":
    main()
