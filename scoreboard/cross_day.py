"""Кросс-дневной энкодер: нейросеть учится узнавать ОДИН И ТОТ ЖЕ хост в РАЗНЫХ окнах (разные часы и дни).

    # проверка идеи без записи результата: день 04 не участвует в обучении, на нём считается Top-1 узнавания
    python -u scoreboard/cross_day.py --holdout d04
    # итоговый блок для склейки: обучение на всех окнах, запись эмбеддингов окон дней 14-16, validation и train
    python -u scoreboard/cross_day.py --holdout none --name xtenc

Режимы входа: --behav-src ref (поведение из runs/blocks_ref и runs/old/blocks_ref, как раньше, окна train, d02, d03, d04)
  или --behav-src own (поведение из самих блоков runs/blocks_all/<окно>.npz, 66 чисел; окна — все old_* из папки блоков).

Вход (всё делают уже имеющиеся скрипты):
  runs/blocks_all/<окно>.npz       build_blocks.py по папке со ссылками на окна (см. команды в инструкции): train, validation,
                                   test, test_day15, test_day16 (дни 14-16) и old_d02, old_d03, old_d04 (часы 12:30-13:30 дней 02-04)
  runs/blocks_ref/<окно>.npz       эталонное поведение для текущих окон (build_behavioral.py)
  runs/old/blocks_ref/<окно>.npz   то же для старых окон (test = день 02, test_day03, test_day04)

Идея. Вектор хоста в окне = [поведение, собеседники (SVD), сервисы (SVD)] с весами. Нейросеть (MLP) переводит его в 128 чисел
так, чтобы один хост в двух разных окнах получался близким, а разные хосты — далёкими (InfoNCE). Пары окон берутся из
train, день 02 (12:30-13:30), день 03, день 04. Вектор хоста по-прежнему считается только по ОДНОМУ часу его окна;
дни 02-04 используются лишь для обучения весов нейросети (допустимо: «любые дни до 12»).
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OLD = {"d02": "test", "d03": "test_day03", "d04": "test_day04"}       # имя в обучении -> имя окна в runs/old/blocks_ref
TRAIN_NAMES = ["train", "d02", "d03", "d04"]                    # окна режима ref; в режиме own — train и все old_* из папки блоков


def log(msg: str) -> None:
    print(msg, flush=True)


def load_x(blocks: Path, behav_dir: Path | None, w: str, bname: str, kp: int, ks: int, wp: float, ws: float, sc: dict | None,
           rp: int = 0, rs: int = 0, wrp: float = 0.5, wrs: float = 0.3):
    """Входной вектор окна: [behav/sb, wp*peers/sp, ws*services/ss]; sc — масштабы (по train) или None -> посчитать.
    behav_dir None -> поведение берётся из самого файла блоков (поле behav)."""
    z = np.load(blocks / f"{w}.npz")
    if behav_dir is None:
        b = z
    else:
        b = np.load(behav_dir / f"{bname}.npz")
        if list(z["hosts"]) != list(b["hosts"]):
            raise SystemExit(f"{w}: порядок хостов в блоках и в поведении не совпадает — пересоберите блоки")
    parts = {"b": b["behav"].astype(np.float64), "p": z["peers"][:, :kp].astype(np.float64), "s": z["services"][:, :ks].astype(np.float64)}
    act = (np.abs(parts["p"]).sum(1) > 0) | (np.abs(parts["s"]).sum(1) > 0)          # хост был активен в окне
    wts = {"b": 1.0, "p": wp, "s": ws}
    if rp > 0:                                                  # «отпечаток»: случайная проекция полной TF-IDF (build_blocks.py --rp-peers)
        if "peers_rp" not in z.files:
            raise SystemExit(f"{w}.npz: нет peers_rp — соберите блоки с --rp-peers")
        parts["r"], wts["r"] = z["peers_rp"][:, :rp].astype(np.float64), wrp
    if rs > 0:
        if "services_rp" not in z.files:
            raise SystemExit(f"{w}.npz: нет services_rp — соберите блоки с --rp-services")
        parts["t"], wts["t"] = z["services_rp"][:, :rs].astype(np.float64), wrs
    if sc is None:
        sc = {k: float(np.sqrt((v ** 2).sum(1).mean())) + 1e-9 for k, v in parts.items()}
    x = np.hstack([wts[k] * parts[k] / sc[k] for k in parts])
    return x.astype(np.float32), act, sc, list(z["hosts"])


def norm(z: np.ndarray) -> np.ndarray:
    return z / (np.linalg.norm(z, axis=1, keepdims=True) + 1e-9)


def top1(za: np.ndarray, zb: np.ndarray, ids: np.ndarray) -> float:
    """Хост из окна b ищется среди хостов окна a по косинусу (только хосты, активные в обоих окнах)."""
    a, b = norm(za[ids]), norm(zb[ids])
    best = (b @ a.T).argmax(1)
    return float((best == np.arange(len(ids))).mean())


def encode(model, x: np.ndarray) -> np.ndarray:
    import torch
    model.eval()
    with torch.no_grad():
        return model(torch.from_numpy(x)).numpy()


def train(xs: dict, acts: dict, names: list[str], out_dim: int, hidden: int, epochs: int, tau: float, drop: float, seed: int,
          batch: int, lr: float):
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    d = xs[names[0]].shape[1]
    model = nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, out_dim))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    pairs = [(a, b) for a in names for b in names if a != b]
    both = {(a, b): np.flatnonzero(acts[a] & acts[b]) for a, b in pairs}
    steps = max(1, int(np.ceil(len(next(iter(both.values()))) / batch)))
    t0 = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        for _ in range(steps):
            a, b = pairs[rng.integers(len(pairs))]
            ids = rng.choice(both[(a, b)], size=min(batch, len(both[(a, b)])), replace=False)
            xa, xb = torch.from_numpy(xs[a][ids]), torch.from_numpy(xs[b][ids])
            xa, xb = F.dropout(xa, drop), F.dropout(xb, drop)
            za, zb = F.normalize(model(xa), dim=1), F.normalize(model(xb), dim=1)
            lg = za @ zb.T / tau
            tgt = torch.arange(len(ids))
            loss = (F.cross_entropy(lg, tgt) + F.cross_entropy(lg.T, tgt)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot += float(loss)
        if ep % 10 == 0 or ep == 1:
            log(f"  эпоха {ep}/{epochs}: loss {tot / steps:.3f} ({time.time() - t0:.0f} с)")
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--blocks", default=str(ROOT / "runs" / "blocks_all"))
    ap.add_argument("--behav-dir", default=str(ROOT / "runs" / "blocks_ref"))
    ap.add_argument("--old-behav-dir", default=str(ROOT / "runs" / "old" / "blocks_ref"))
    ap.add_argument("--processed", default=str(ROOT / "runs" / "processed"))
    ap.add_argument("--out", default=str(ROOT / "runs" / "submissions"))
    ap.add_argument("--name", default="xtenc")
    ap.add_argument("--holdout", default="d04", help="d04/d03/d02 — окно не участвует в обучении (проверка); none — обучать на всех и записать результат")
    ap.add_argument("--exclude", nargs="*", default=[], help="окна, которые не брать в обучение (например d02: оно идёт сразу после validation)")
    ap.add_argument("--behav-src", choices=["ref", "own"], default="ref", help="ref — эталонное поведение (4 окна), own — поведение из блоков, любые окна old_*")
    ap.add_argument("--k-peers", type=int, default=128)
    ap.add_argument("--k-services", type=int, default=64)
    ap.add_argument("--rp-peers", type=int, default=0, help="сколько чисел случайной проекции полной TF-IDF собеседников подать на вход (0 — не подавать)")
    ap.add_argument("--rp-services", type=int, default=0, help="то же для сервисов")
    ap.add_argument("--w-rp-peers", type=float, default=0.5)
    ap.add_argument("--w-rp-services", type=float, default=0.3)
    ap.add_argument("--w-peers", type=float, default=0.5)
    ap.add_argument("--w-services", type=float, default=0.3)
    ap.add_argument("--out-dim", type=int, default=128)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60, help="0 — только базовая оценка без обучения (torch не нужен)")
    ap.add_argument("--tau", type=float, default=0.1)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--seeds", type=int, default=1, help="сколько независимых энкодеров обучить (зёрна seed, seed+1, ...); выходы склеиваются (ансамбль)")
    a = ap.parse_args()
    t0 = time.time()
    blocks = Path(a.blocks)
    own = a.behav_src == "own"
    bref = Path(a.behav_dir)
    oref = Path(a.old_behav_dir)
    kw = dict(kp=a.k_peers, ks=a.k_services, wp=a.w_peers, ws=a.w_services, rp=a.rp_peers, rs=a.rp_services,
              wrp=a.w_rp_peers, wrs=a.w_rp_services)

    if own:
        tags = sorted(p.stem[4:] for p in blocks.glob("old_*.npz"))
        all_names = ["train"] + tags
        log(f"поведение из блоков (own); окна обучения-кандидаты: {all_names}")
    else:
        all_names = list(TRAIN_NAMES)
        # совместимость поведения старых и текущих окон (стандартизация по одному и тому же train)
        tr_cur, tr_old = np.load(bref / "train.npz")["behav"], np.load(oref / "train.npz")["behav"]
        if tr_cur.shape != tr_old.shape or not np.allclose(tr_cur, tr_old, atol=1e-3):
            raise SystemExit("поведение train в runs/blocks_ref и runs/old/blocks_ref различается — старые блоки несовместимы, пересоберите")
        log("поведение train в старой и текущей папках совпадает")

    def spec(n):
        """(имя файла блоков, папка поведения, имя файла поведения) для окна обучения n"""
        if n == "train":
            return "train", (None if own else bref), "train"
        return f"old_{n}", (None if own else oref), OLD.get(n, n)

    xs, acts, sc = {}, {}, None
    hosts0 = None
    for n in all_names:
        w, bd, bn = spec(n)
        x, act, sc_, hosts = load_x(blocks, bd, w, bn, sc=sc, **kw)
        sc = sc or sc_                                         # масштабы — по train (первое окно)
        if hosts0 is None:
            hosts0 = hosts
        elif hosts != hosts0:
            raise SystemExit(f"{n}: порядок хостов отличается от train")
        xs[n], acts[n] = x, act
        log(f"окно {n}: активных хостов {int(act.sum()):,} из {len(act):,}; вход {x.shape[1]} чисел")

    def match(n, prefixes):                                    # окно n входит в список префиксов (d04 -> d04, d04h10, ...)
        return any(n.startswith(p) for p in prefixes)

    held_prefix = [] if a.holdout == "none" else [a.holdout]
    held = [n for n in all_names if n != "train" and match(n, held_prefix)]
    names = [n for n in all_names if n not in held and not match(n, a.exclude)]
    ref_n = next((n for n in names if n != "train"), "train")  # опорное обучающее окно для второй пары

    def evaluate(tag: str, f):
        zs = {n: f(xs[n]) for n in all_names}
        pairs = [("train", h) for h in held] + ([(ref_n, h) for h in held] if ref_n != "train" else [])
        if not pairs:                                          # итоговый режим: просто сколько узнаётся на обучающих парах
            pairs = [("train", n) for n in names if n != "train"][:6]
        log(f"Top-1 узнавания, {tag}:")
        means = []
        for qa, qb in pairs:
            ids = np.flatnonzero(acts[qa] & acts[qb])
            v = top1(zs[qa], zs[qb], ids)
            if qa == "train" and qb in held:
                means.append(v)
            seen = "КОНТРОЛЬ" if qb in held else "обучающая пара"
            log(f"   {qa}->{qb}: {v:.3f} ({len(ids)} хостов, {seen})")
        if means:
            log(f"   СРЕДНЕЕ train->контрольные окна ({len(means)}): {np.mean(means):.3f}   <- число для сравнения вариантов")

    evaluate("вход без обучения (косинус по вектору [поведение, собеседники, сервисы])", lambda x: x)
    if a.epochs == 0:
        return
    log(f"обучение: окна {names}, контроль {held}, эпох {a.epochs}")
    models = []
    for i in range(a.seeds):
        if a.seeds > 1:
            log(f"--- энкодер {i + 1}/{a.seeds} (зерно {a.seed + i})")
        models.append(train(xs, acts, names, a.out_dim, a.hidden, a.epochs, a.tau, a.drop, a.seed + i, a.batch, a.lr))

    def ens(x):
        """Выходы всех энкодеров, каждый нормирован по строкам, склеены и поделены на sqrt(N): косинус = среднее косинусов."""
        return np.hstack([norm(encode(m, x)) for m in models]) / np.sqrt(len(models))

    evaluate(f"после обучения ({a.out_dim * a.seeds} чисел, энкодеров {a.seeds})", ens)
    if held:
        log(f"всего {time.time() - t0:.0f} с. Это проверка; результат не записан (для записи: --holdout none)")
        return

    cohort = pd.read_csv(Path(a.processed) / "cohort.csv", dtype={"host_id": str}).set_index("host_id")
    if list(cohort.index) != hosts0:
        raise SystemExit("порядок хостов в блоках не совпадает с cohort.csv")
    sub = Path(a.out) / a.name
    sub.mkdir(parents=True, exist_ok=True)
    wins = sorted(p.stem for p in blocks.glob("*.npz") if not p.stem.startswith("old_"))
    for w in ["train"] + [w for w in wins if w != "train"]:
        x, _, _, _ = load_x(blocks, None if own else bref, w, w, sc=sc, **kw)
        z = ens(x)
        mask = np.ones(len(cohort), bool) if w == "train" else cohort[f"evaluable_{w}"].to_numpy(bool)
        df = pd.DataFrame(z[mask], index=pd.Index(cohort.index[mask], name="host_id")).add_prefix("emb_")
        df.to_csv(sub / ("embeddings.csv" if w == "train" else f"{w}.csv"), float_format="%.5g")
        log(f"записано {sub / ('embeddings.csv' if w == 'train' else w + '.csv')} ({int(mask.sum()):,} хостов)")
    log(f"готово за {time.time() - t0:.0f} с; папка {sub} — это блок для gnn_combine_v3.py (--extra)")

if __name__ == "__main__":
    main()
