"""Прогноз будущих связей на ОДНОМ замороженном наборе кандидатов для всех методов.

Проверяем, будет ли связь между парой хостов в более позднем окне. Это не классификация ролей.

  python -m evaluation.future_links build --source-flows train.csv \
    --target-flows test.csv --hosts embeddings.csv --output reference/edges.csv
  python -m evaluation.future_links score --embeddings embeddings.csv \
    --edges reference/edges.csv --output results/link_prediction.json

Связь (неориентированная) — хотя бы один поток в любую сторону.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

from evaluation.temporal_reference import load_flows
from evaluation.io import load_embeddings as _load_matrix
from evaluation.uncertainty import auc_with_ci, host_index, reweighted_average_precision


FIELDS = ["host_a", "host_b", "source_window", "target_window", "label", "split"]


def _pair(a: str, b: str) -> tuple[str, str]:
    return (a, b) if a < b else (b, a)


def observed_edges(flows: pd.DataFrame, cohort: set[str]) -> set[tuple[str, str]]:
    return {
        _pair(str(a), str(b))
        for a, b in zip(flows["src"], flows["dst"])
        if a != b and a in cohort and b in cohort
    }


def build_edges(
    source: pd.DataFrame,
    target: pd.DataFrame,
    hosts: list[str],
    *,
    seed: int = 0,
    negatives_per_positive: int = 1,
    split: str = "test",
) -> pd.DataFrame:
    """Детерминированный набор кандидатов для заданной когорты: положительные + случайные отрицательные.

    Отрицательные выбираются случайно с отбраковкой, без перебора всех O(n^2) пар.
    PR-AUC на такой выборке — не PR-AUC «в бою».
    """
    if not 0 < negatives_per_positive <= 100:
        raise ValueError("negatives_per_positive должен быть от 1 до 100")
    if len(hosts) != len(set(hosts)) or len(hosts) < 3 or any(not h for h in hosts):
        raise ValueError("нужно хотя бы 3 уникальных непустых хоста")
    if target["ts"].min() <= source["ts"].max():
        raise ValueError("source- и target-окна пересекаются или соприкасаются")
    if split not in {"test", "validation"}:
        raise ValueError("split должен быть validation или test")
    hosts = sorted(hosts)
    population = set(hosts)
    positives = sorted(observed_edges(target, population))
    if not positives:
        raise ValueError("в target-окне нет ни одной связи внутри когорты")
    all_pairs_count = len(hosts) * (len(hosts) - 1) // 2
    remaining = all_pairs_count - len(positives)
    requested = len(positives) * negatives_per_positive
    if remaining < requested:
        raise ValueError(
            f"не хватает отрицательных пар: нужно {requested}, есть {remaining}; "
            "уменьшите negatives_per_positive или расширьте когорту"
        )
    rng = np.random.default_rng(seed)
    negatives: set[tuple[str, str]] = set()
    pos_set = set(positives)
    # отбраковка быстрая, пока отрицательных не слишком много
    # эмбеддинги в выборе не участвуют
    retries = 0
    max_retries = max(10000, requested * 100)
    while len(negatives) < requested:
        a, b = rng.choice(len(hosts), size=2, replace=False)
        pair = _pair(hosts[int(a)], hosts[int(b)])
        if pair not in pos_set:
            negatives.add(pair)
        retries += 1
        if retries > max_retries:
            # пространство почти исчерпано: полный перебор допустим только
            # для небольших сетей
            if all_pairs_count > 1_000_000:
                raise RuntimeError(
                    "не удалось набрать отрицательные пары: уменьшите negatives_per_positive "
                    "или используйте разреженный набор кандидатов"
                )
            for i, a in enumerate(hosts):
                for b in hosts[i + 1:]:
                    pair = (a, b)
                    if pair not in pos_set:
                        negatives.add(pair)
                    if len(negatives) >= requested:
                        break
                if len(negatives) >= requested:
                    break
            break

    source_window = (
        source["ts"].min().isoformat() + "Z/" +
        source["ts"].max().isoformat() + "Z"
    )
    target_window = (
        target["ts"].min().isoformat() + "Z/" +
        target["ts"].max().isoformat() + "Z"
    )
    rows = [
        (a, b, source_window, target_window, 1, split)
        for a, b in positives
    ] + [
        (a, b, source_window, target_window, 0, split)
        for a, b in sorted(negatives)
    ]
    out = pd.DataFrame(rows, columns=FIELDS)
    # связи, которые уже были в source-окне, предсказать гораздо проще, чем новые;
    # сохраняем флаг, чтобы считать их отдельно
    seen = observed_edges(source, population)
    out["seen_in_source"] = [
        _pair(a, b) in seen for a, b in zip(out["host_a"], out["host_b"])
    ]
    return out


def load_candidate_edges(file: Path) -> pd.DataFrame:
    df = pd.read_csv(
        file, dtype={"host_a": "string", "host_b": "string",
                     "source_window": "string", "target_window": "string",
                     "split": "string", "label": "string"},
    )
    missing = set(FIELDS) - set(df.columns)
    if missing:
        raise ValueError(f"в edges.csv нет колонок {sorted(missing)}")
    if df.empty or df[FIELDS].isna().any().any():
        raise ValueError("edges.csv пуст или в нём пропуски")
    if not df["label"].isin(["0", "1"]).all():
        raise ValueError("метки связей должны быть 0/1")
    if not df["split"].isin(["test", "validation"]).all():
        raise ValueError("неизвестный split")
    if not (df.host_a < df.host_b).all():
        raise ValueError("пары должны быть упорядочены (host_a < host_b) и из разных хостов")
    if df.duplicated(["host_a", "host_b", "source_window", "target_window", "split"]).any():
        raise ValueError("повторяющиеся пары-кандидаты")
    return df


def score_embeddings(
    embedding: pd.DataFrame, edges: pd.DataFrame,
    *, split: str = "test", n_boot: int = 0, seed: int = 0,
) -> dict:
    """Оценить РОВНО сохранённых кандидатов. Неизвестный хост — ошибка, а не пропуск."""
    edges = edges[edges["split"] == split]
    if edges.empty:
        raise ValueError(f"нет кандидатов для split={split}")
    if edges[["source_window", "target_window"]].drop_duplicates().shape[0] != 1:
        raise ValueError("в одной оценке нельзя смешивать разные пары окон source/target")
    if embedding.index.has_duplicates:
        raise ValueError("повторяющиеся host_id")
    known = set(embedding.index)
    used = set(edges["host_a"]) | set(edges["host_b"])
    if not used.issubset(known):
        raise ValueError(f"в эмбеддинге нет хостов-кандидатов: {sorted(used - known)[:5]}")
    labels = edges["label"].astype(int).to_numpy()
    if len(np.unique(labels)) < 2:
        raise ValueError("нужны и положительные, и отрицательные кандидаты")
    emb = embedding.to_numpy(dtype=float)
    if not np.isfinite(emb).all():
        raise ValueError("в эмбеддинге есть бесконечные или пустые координаты")
    if (np.linalg.norm(emb, axis=1) == 0).any():
        raise ValueError("нулевой вектор хоста: косинус не определён")
    index = {host: j for j, host in enumerate(embedding.index)}
    a = np.array([index[x] for x in edges["host_a"]])
    b = np.array([index[x] for x in edges["host_b"]])
    # косинус по парам без полной матрицы N*N
    norms = np.linalg.norm(emb, axis=1)
    cosine = np.einsum("ij,ij->i", emb[a], emb[b]) / (norms[a] * norms[b])
    window = edges.iloc[0]
    auc = auc_with_ci(labels, cosine, n_boot=n_boot, seed=seed, hosts=host_index(edges.host_a, edges.host_b))
    n_hosts = len(known)
    natural = float(labels.sum() / (n_hosts * (n_hosts - 1) / 2))
    extra: dict = {
        "roc_auc_ci": auc["roc_auc_ci"], "pr_auc_ci": auc["pr_auc_ci"], "ci_method": auc["ci_method"],
        "pr_auc_lift": auc["pr_auc"] / float(labels.mean()),
        # PR-AUC при настоящей доле положительных (связи в целевом окне / все пары когорты);
        # убирает только эффект выборки 1:1
        "natural_prevalence": natural,
        "pr_auc_at_natural_prevalence": (reweighted_average_precision(labels, cosine, natural)
                                         if 0 < natural < 1 else None),
    }
    if "seen_in_source" in edges.columns:
        seen = edges["seen_in_source"].astype(str).str.lower().eq("true").to_numpy()
        extra["share_positives_seen_in_source"] = float(seen[labels == 1].mean())
        new = ~seen
        if new.any() and len(np.unique(labels[new])) == 2:
            extra["new_links"] = {
                "roc_auc": float(roc_auc_score(labels[new], cosine[new])),
                "pr_auc": float(average_precision_score(labels[new], cosine[new])),
                "n_examples": int(new.sum()), "n_positive": int(labels[new].sum()),
                "positive_rate": float(labels[new].mean()),
                "note": "candidates not observed in the source window (new links vs unseen non-links)",
            }
    return {
        **extra,
        "roc_auc": float(roc_auc_score(labels, cosine)),
        "pr_auc": float(average_precision_score(labels, cosine)),
        "n_examples": int(len(labels)),
        "n_positive": int(labels.sum()),
        "positive_rate": float(labels.mean()),
        "n_candidate_hosts": len(used),
        "n_embedding_hosts": len(known),
        "host_coverage": float(len(used) / len(known)),
        "split": split,
        "source_window": window["source_window"],
        "target_window": window["target_window"],
        "interpretation": (
            "PR-AUC depends on the frozen sampled candidate distribution; "
            "this is NOT full-population deployment average precision."
        ),
    }


def _window_bounds(window: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    a, b = window.split("/")
    return pd.Timestamp(a.rstrip("Z")), pd.Timestamp(b.rstrip("Z"))


def score_learned(embedding: pd.DataFrame, fit_edges: pd.DataFrame, eval_edges: pd.DataFrame, *,
                  fit_split: str = "validation", eval_split: str = "test", seed: int = 0,
                  n_boot: int = 0) -> dict:
    """Классификатор связей на замороженных эмбеддингах: LogisticRegression на [|e_a - e_b|, e_a * e_b].

    Учится на кандидатах БОЛЕЕ РАННЕГО целевого окна (validation), оценивается на позднем (test);
    эмбеддинг в обоих случаях один — из source-окна. Косинус для трафика плох: клиенты связаны
    с серверами, а не с похожими клиентами. Окно обучения должно закончиться до окна оценки и иметь
    тот же source, иначе — отказ. C=1 фиксирован и на test не подбирается.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    fit = fit_edges[fit_edges["split"] == fit_split]
    ev = eval_edges[eval_edges["split"] == eval_split]
    if fit.empty or ev.empty:
        raise ValueError(f"нужны кандидаты и для обучения ({fit_split!r}), и для оценки ({eval_split!r})")
    fw, ew = fit.iloc[0], ev.iloc[0]
    if fw["source_window"] != ew["source_window"]:
        raise ValueError("у кандидатов обучения и оценки разные source-окна")
    if _window_bounds(fw["target_window"])[1] > _window_bounds(ew["target_window"])[0]:
        raise ValueError("окно обучения должно закончиться до начала окна оценки (утечка по времени)")
    used = set(fit.host_a) | set(fit.host_b) | set(ev.host_a) | set(ev.host_b)
    if not used <= set(embedding.index):
        raise ValueError(f"в эмбеддинге нет хостов-кандидатов: {sorted(used - set(embedding.index))[:5]}")
    scaler = StandardScaler().fit(embedding.to_numpy(float))
    z = pd.DataFrame(scaler.transform(embedding.to_numpy(float)), index=embedding.index)

    def feats(df):
        a, b = z.loc[df.host_a].to_numpy(), z.loc[df.host_b].to_numpy()
        return np.hstack([np.abs(a - b), a * b])

    yf, ye = fit["label"].astype(int).to_numpy(), ev["label"].astype(int).to_numpy()
    if len(np.unique(yf)) < 2 or len(np.unique(ye)) < 2:
        raise ValueError("и в обучении, и в оценке нужны оба класса")
    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(feats(fit), yf)
    score = clf.predict_proba(feats(ev))[:, 1]
    out = auc_with_ci(ye, score, n_boot=n_boot, seed=seed, hosts=host_index(ev.host_a, ev.host_b))
    # случайный уровень: тот же классификатор на случайных векторах той же размерности (запоминает хабы)
    z = pd.DataFrame(np.random.default_rng(seed).standard_normal(z.shape), index=z.index)
    rnd = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(feats(fit), yf)
    out["random_roc_auc"] = float(roc_auc_score(ye, rnd.predict_proba(feats(ev))[:, 1]))
    out.update(n_fit=int(len(fit)), n_eval=int(len(ev)), fit_split=fit_split, eval_split=eval_split,
               fit_target_window=fw["target_window"], eval_target_window=ew["target_window"],
               pr_auc_lift=out["pr_auc"] / float(ye.mean()))
    return out


def persistence_baseline(edges: pd.DataFrame, split: str = "test") -> dict | None:
    """Точка отсчёта без эмбеддинга: 1, если пара уже общалась в source-окне."""
    e = edges[edges["split"] == split]
    if "seen_in_source" not in e.columns:
        return None
    y = e["label"].astype(int).to_numpy()
    seen = e["seen_in_source"].astype(str).str.lower().eq("true").to_numpy().astype(float)
    if len(np.unique(y)) < 2:
        return None
    return {"roc_auc": float(roc_auc_score(y, seen)), "pr_auc": float(average_precision_score(y, seen)),
            "note": "trivial predictor: pair seen in the source window; embeddings should be read against it"}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sp = ap.add_subparsers(dest="command", required=True)
    build = sp.add_parser("build")
    build.add_argument("--source-flows", type=Path, required=True)
    build.add_argument("--target-flows", type=Path, required=True)
    build.add_argument("--hosts", type=Path, required=True,
                       help="Canonical embeddings.csv used ONLY to define fixed host cohort")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--seed", type=int, default=0)
    build.add_argument("--negatives-per-positive", type=int, default=1)
    score = sp.add_parser("score")
    score.add_argument("--embeddings", type=Path, required=True)
    score.add_argument("--edges", type=Path, required=True)
    score.add_argument("--output", type=Path)
    args = ap.parse_args()
    if args.command == "build":
        cohort = _load_matrix(args.hosts, "host_id", "emb_").index.astype(str).tolist()
        df = build_edges(load_flows(args.source_flows), load_flows(args.target_flows),
                         cohort, seed=args.seed,
                         negatives_per_positive=args.negatives_per_positive)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"Saved {len(df)} candidate edges to {args.output}")
    else:
        emb = _load_matrix(args.embeddings, "host_id", "emb_")
        edges = load_candidate_edges(args.edges)
        report = score_embeddings(emb, edges)
        report["edges_sha256"] = hashlib.sha256(args.edges.read_bytes()).hexdigest()
        report["ground_truth_host_roles"] = False
        output = json.dumps(report, indent=2, ensure_ascii=False)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(output + "\n", encoding="utf-8")
        print(output)


if __name__ == "__main__":
    main()
