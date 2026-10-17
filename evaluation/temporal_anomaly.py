
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon


DATA_DIR = Path("runs/synthetic_demo/processed/flows")
OUTPUT_DIR = Path("runs/clustering_anomaly")

# Продолжительность окон в часах по диапазонам ts.
WINDOW_HOURS = {
    "train": 6.0,
    "validation": 3.0,
    "test": 3.0,
}

FEATURES = [
    "events_per_hour",
    "bytes_per_hour",
    "packets_per_hour",
    "unique_destinations_per_hour",
]


def load_window(split):
    path = DATA_DIR / f"{split}.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing flow file: {path}")

    df = pd.read_csv(path)
    required = {"src", "dst", "bytes", "packets", "ts"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    for column in ["bytes", "packets", "ts"]:
        df[column] = pd.to_numeric(df[column], errors="raise")

    if df[list(required)].isna().any().any():
        raise ValueError(f"Missing values in required columns: {path}")

    return df


def host_features(df, split):
    # Use source-host activity consistently across all windows.
    grouped = df.groupby("src", sort=True).agg(
        events=("src", "size"),
        bytes=("bytes", "sum"),
        packets=("packets", "sum"),
        unique_destinations=("dst", "nunique"),
    )

    hours = WINDOW_HOURS[split]
    result = pd.DataFrame(index=grouped.index)
    result.index.name = "host_id"

    result["events_per_hour"] = grouped["events"] / hours
    result["bytes_per_hour"] = grouped["bytes"] / hours
    result["packets_per_hour"] = grouped["packets"] / hours
    result["unique_destinations_per_hour"] = (
        grouped["unique_destinations"] / hours
    )

    return result


def robust_scale(values):
    median = np.median(values)
    mad = np.median(np.abs(values - median))
    scale = 1.4826 * mad
    return max(float(scale), 1e-6)


def evaluate():
    windows = {
        split: load_window(split)
        for split in WINDOW_HOURS
    }

    features = {
        split: host_features(df, split)
        for split, df in windows.items()
    }

    all_hosts = sorted(
        set().union(*(set(frame.index) for frame in features.values()))
    )

    # Missing source activity is treated as zero source events.
    features = {
        split: frame.reindex(all_hosts, fill_value=0.0)
        for split, frame in features.items()
    }

    train_log = np.log1p(features["train"][FEATURES])
    baseline_scales = {
        feature: robust_scale(train_log[feature].to_numpy())
        for feature in FEATURES
    }

    score_frames = []
    summary_rows = []

    for split in ["validation", "test"]:
        current_log = np.log1p(features[split][FEATURES])
        difference = (current_log - train_log).abs()

        normalized = difference.copy()
        for feature in FEATURES:
            normalized[feature] /= baseline_scales[feature]

        scores = normalized.mean(axis=1)

        host_scores = pd.DataFrame({
            "host_id": all_hosts,
            "split": split,
            "temporal_change_score": scores.to_numpy(),
        })

        for feature in FEATURES:
            host_scores[f"train_{feature}"] = (
                features["train"][feature].to_numpy()
            )
            host_scores[f"{split}_{feature}"] = (
                features[split][feature].to_numpy()
            )
            host_scores[f"abs_log_change_{feature}"] = (
                difference[feature].to_numpy()
            )

        host_scores["change_rank"] = (
            host_scores["temporal_change_score"]
            .rank(method="min", ascending=False)
            .astype(int)
        )
        score_frames.append(host_scores)

        # Distribution-level change in log event-rate across hosts.
        train_rates = train_log["events_per_hour"].to_numpy()
        current_rates = current_log["events_per_hour"].to_numpy()
        bins = np.histogram_bin_edges(
            np.concatenate([train_rates, current_rates]), bins=10
        )
        p, _ = np.histogram(train_rates, bins=bins)
        q, _ = np.histogram(current_rates, bins=bins)

        # Add a small smoothing constant to avoid zero-probability bins.
        p = (p + 1e-8) / (p.sum() + 1e-8 * len(p))
        q = (q + 1e-8) / (q.sum() + 1e-8 * len(q))

        summary_rows.append({
            "split": split,
            "n_hosts": len(all_hosts),
            "median_temporal_change_score": float(scores.median()),
            "p95_temporal_change_score": float(scores.quantile(0.95)),
            "js_divergence_event_rate": float(jensenshannon(p, q) ** 2),
            "interpretation": (
                "Unlabelled temporal change; not confirmed anomaly accuracy"
            ),
        })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    host_output = OUTPUT_DIR / "host_temporal_scores.csv"
    summary_output = OUTPUT_DIR / "temporal_anomaly_results.csv"

    pd.concat(score_frames, ignore_index=True).to_csv(
        host_output, index=False
    )
    pd.DataFrame(summary_rows).to_csv(summary_output, index=False)

    print(pd.DataFrame(summary_rows).to_string(index=False))
    print(f"\nSaved host scores: {host_output}")
    print(f"Saved temporal summary: {summary_output}")


if __name__ == "__main__":
    evaluate()
