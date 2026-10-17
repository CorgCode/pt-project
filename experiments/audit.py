"""Аудит методологии прогона (docs/METHODOLOGY.md). Каждая функция отвечает на один вопрос:

  weight_sensitivity   устойчив ли поведенческий эталон (и порядок методов) к другим весам признаков?
  reference_stability  насколько меняется поведенческое соседство хоста между окнами?
  random_null          сколько набирает чистый шум? (случайный уровень по каждой метрике)
  split_agreement      одинаково ли validation и test упорядочивают методы?
  prevalence           доля положительных среди кандидатов против настоящей
  OVERLAP              какие пары «метод — эталон» делят информацию по построению
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from evaluation.evaluate import run
from evaluation.knn_consistency import build_reference_features, evaluate_embedding
from evaluation.temporal_reference import load_flows

BLOCKS = ("volume", "peers", "ratio", "ports", "proto", "hourly")

# Что вход метода делит с ИСТОЧНИКОМ эталона по построению.
OVERLAP = {
    "random": {"neighbors": "none", "edges": "none", "pairs": "none"},
    "behavioral": {"neighbors": "same feature definitions, earlier window (feature overlap)",
                   "edges": "none", "pairs": "weak: behaviour correlates with peers"},
    "behavioral_pca": {"neighbors": "same feature definitions, earlier window (feature overlap)",
                       "edges": "none", "pairs": "weak: behaviour correlates with peers"},
    "behavioral_svd": {"neighbors": "same feature definitions, earlier window (feature overlap)",
                       "edges": "none", "pairs": "weak: behaviour correlates with peers"},
    "behavioral_portfree": {"neighbors": "same feature definitions MINUS the port block, earlier window (reduced feature overlap)",
                            "edges": "none", "pairs": "weak: behaviour correlates with peers"},
    "behavioral_portfree_pca": {"neighbors": "same feature definitions MINUS the port block, earlier window (reduced feature overlap)",
                                "edges": "none", "pairs": "weak: behaviour correlates with peers"},
    "graph_spectral": {"neighbors": "weak: degree/volume leaks into the spectrum",
                       "edges": "strong: link persistence (peers seen in train recur)",
                       "pairs": "strong: peer sets persist across windows"},
    "reference_ceiling": {"neighbors": "identical by construction (target-window features)",
                          "edges": "none", "pairs": "weak"},
}


def weight_sensitivity(processed_dir: Path, hosts: list[str], target: str,
                       embeddings: dict[str, pd.DataFrame], *, n_configs: int = 12, seed: int = 0,
                       k: int = 10) -> dict:
    """Пересобрать поведенческий эталон со случайно изменёнными весами блоков.

    Каждая конфигурация умножает вес блока на exp(N(0, 0.5)), в трети конфигураций ещё обнуляет случайный блок.
    Для каждой: совпадение соседей с основным эталоном и Consistency@k каждого метода по новому эталону;
    согласие порядка методов с основным — тау Кендалла.
    """
    flows = load_flows(Path(processed_dir) / "flows" / f"{target}.csv")
    canon = build_reference_features(flows, hosts=hosts)
    rng = np.random.default_rng(seed)

    def scores(ref):
        return {m: float(evaluate_embedding(e.loc[hosts], ref, ks=(k,)).loc[(k, "consistency"), "mean"])
                for m, e in embeddings.items()}

    base = scores(canon)
    rows = []
    for c in range(n_configs):
        w = {b: float(np.exp(rng.normal(0, 0.5))) for b in BLOCKS}
        dropped = None
        if c % 3 == 2:
            dropped = BLOCKS[int(rng.integers(len(BLOCKS)))]
            w[dropped] = 1e-9
        alt = build_reference_features(flows, hosts=hosts, weights=w)
        agree = float(evaluate_embedding(alt.loc[hosts], canon, ks=(k,)).loc[(k, "consistency"), "mean"])
        s = scores(alt)
        names = list(base)
        tau = kendalltau([base[m] for m in names], [s[m] for m in names]).statistic if len(names) > 2 else float("nan")
        rows.append({"config": c, "dropped_block": dropped, "reference_agreement": agree,
                     "kendall_tau": None if np.isnan(tau) else float(tau),
                     "scores": s, "weights": w})
    taus = [r["kendall_tau"] for r in rows if r["kendall_tau"] is not None]
    return {"k": k, "canonical_scores": base, "configs": rows,
            "mean_reference_agreement": float(np.mean([r["reference_agreement"] for r in rows])),
            "min_reference_agreement": float(np.min([r["reference_agreement"] for r in rows])),
            "mean_kendall_tau": float(np.mean(taus)) if taus else None,
            "min_kendall_tau": float(np.min(taus)) if taus else None,
            "chance_agreement": k / (len(hosts) - 1)}


def reference_stability(processed_dir: Path, hosts: list[str], windows: list[str], k: int = 10) -> dict:
    """Совпадение соседей между поведенческими эталонами разных окон."""
    feats = {w: build_reference_features(load_flows(Path(processed_dir) / "flows" / f"{w}.csv"), hosts=hosts)
             for w in windows}
    out = {}
    for i, a in enumerate(windows):
        for b in windows[i + 1:]:
            r = evaluate_embedding(feats[a], feats[b], ks=(k,))
            out[f"{a}->{b}"] = {m: float(r.loc[(k, m), "mean"]) for m in ("consistency", "jaccard", "ndcg")}
    return {"k": k, "pairs": out, "chance_consistency": k / (len(hosts) - 1),
            "meaning": "Consistency@k between neighbour lists from two windows' behavioural profiles: "
                       "an upper-ish reference for how well ANY earlier-window embedding can match a later one"}


def random_null(reference_dir: Path, hosts: list[str], tmp_dir: Path, *, dim: int = 16,
                n: int = 30, seed: int = 0, fit_reference_dir: Path | None = None) -> dict:
    """Распределение метрик у случайных эмбеддингов на настоящем эталоне (случайный уровень).

    Обученный классификатор связей у случайных векторов НЕ 0.5: случайный вектор — это идентификатор хоста,
    и классификатор может запомнить хабы из окна обучения.
    """
    tmp_dir = Path(tmp_dir)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        e = pd.DataFrame(np.random.default_rng(seed + i).standard_normal((len(hosts), dim)),
                         index=pd.Index(hosts, name="host_id")).add_prefix("emb_")
        p = tmp_dir / f"null_{i}.csv"
        e.to_csv(p)
        rows.append(run(p, reference_dir, f"null{i}", fit_reference_dir=fit_reference_dir)["metrics"])
    df = pd.DataFrame(rows)
    keep = [c for c in df.columns if c.endswith("_mean") or not c.endswith(("_median", "_std"))]
    return {"n_draws": n, "metrics": {c: {"mean": float(df[c].mean()), "std": float(df[c].std(ddof=1)),
                                          "p95": float(df[c].quantile(0.95)), "p5": float(df[c].quantile(0.05))}
                                      for c in keep}}


def split_agreement(results: dict[str, dict[str, dict]], metrics: list[str]) -> dict:
    """Тау Кендалла между порядком методов на validation и на test, по каждой метрике."""
    out = {}
    val, test = results.get("validation", {}), results.get("test", {})
    names = sorted(set(val) & set(test))
    for m in metrics:
        a = [val[n]["metrics"].get(m) for n in names]
        b = [test[n]["metrics"].get(m) for n in names]
        if len(names) > 2 and None not in a + b:
            tau = kendalltau(a, b).statistic
            out[m] = None if np.isnan(tau) else float(tau)
    return out


def prevalence(results_test: dict[str, dict]) -> dict:
    any_res = next(iter(results_test.values()))
    d = any_res.get("metric_details", {})
    link, pair = d.get("future_link_prediction", {}), d.get("pairwise_auc", {})
    return {
        "future_links": {"sampled_positive_rate": link.get("positive_rate"),
                         "natural_prevalence": link.get("natural_prevalence"),
                         "share_positives_seen_in_source": link.get("share_positives_seen_in_source"),
                         "note": "PR-AUC is only comparable at equal prevalence; use lift or the natural-prevalence value"},
        "pairs": {"sampled_positive_rate": pair.get("positive_rate"),
                  "judged_population_prevalence": pair.get("judged_population_prevalence"),
                  "population_positive_rate": pair.get("population_positive_rate"),
                  "unjudged_share": pair.get("unjudged_share"),
                  "note": "the judged prevalence is conditional on a pair being judged (J >= tau or J == 0); "
                          "unjudged pairs are excluded, so it is not the positive rate of all host pairs"},
    }


def dump(path: Path, obj) -> None:
    Path(path).write_text(json.dumps(obj, indent=2, sort_keys=True, default=float) + "\n", encoding="utf-8")
