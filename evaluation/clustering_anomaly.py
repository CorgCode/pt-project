
import numpy as np
import pandas as pd

from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
from sklearn.metrics import (
    adjusted_rand_score,
    silhouette_score,
    davies_bouldin_score,
    precision_score,
    recall_score,
    f1_score,
)


def generate_synthetic_embeddings(seed=42):
    """Generate host embeddings with known groups and synthetic anomalies."""
    rng = np.random.default_rng(seed)

    normal_embeddings, normal_labels = make_blobs(
        n_samples=300,
        centers=3,
        n_features=6,
        cluster_std=0.55,
        random_state=seed,
    )

    anomalies = rng.uniform(
        low=12,
        high=16,
        size=(20, 6),
    )

    embeddings = np.vstack([normal_embeddings, anomalies])
    is_anomaly = np.concatenate([
        np.zeros(len(normal_embeddings), dtype=int),
        np.ones(len(anomalies), dtype=int),
    ])

    host_ids = [f"host_{i:03d}" for i in range(len(embeddings))]

    df = pd.DataFrame(
        embeddings,
        columns=[f"emb_{i}" for i in range(6)],
    )
    df.insert(0, "host_id", host_ids)
    df["is_synthetic_anomaly"] = is_anomaly

    return df, normal_labels


def evaluate_synthetic_embeddings(seed=42):
    df, normal_labels = generate_synthetic_embeddings(seed)

    embedding_columns = [
        column for column in df.columns
        if column.startswith("emb_")
    ]
    X = df[embedding_columns].to_numpy()
    y_true = df["is_synthetic_anomaly"].to_numpy()

    model = DBSCAN(eps=1.8, min_samples=5)
    cluster_labels = model.fit_predict(X)

    y_pred = (cluster_labels == -1).astype(int)

    non_noise_mask = cluster_labels != -1
    non_noise_labels = cluster_labels[non_noise_mask]
    non_noise_X = X[non_noise_mask]
    unique_clusters = np.unique(non_noise_labels)

    if 2 <= len(unique_clusters) < len(non_noise_X):
        silhouette = silhouette_score(non_noise_X, non_noise_labels)
        davies_bouldin = davies_bouldin_score(
            non_noise_X, non_noise_labels
        )
    else:
        silhouette = np.nan
        davies_bouldin = np.nan


    normal_mask = y_true == 0
    ari = adjusted_rand_score(
        normal_labels,
        cluster_labels[normal_mask],
    )

    metrics = {
        "method": "synthetic_dbscan",
        "n_hosts": len(df),
        "n_synthetic_anomalies": int(y_true.sum()),
        "n_clusters_excluding_noise": len(unique_clusters),
        "n_detected_outliers": int(y_pred.sum()),
        "silhouette": silhouette,
        "davies_bouldin": davies_bouldin,
        "normal_cluster_ari": ari,
        "anomaly_precision": precision_score(
            y_true, y_pred, zero_division=0
        ),
        "anomaly_recall": recall_score(
            y_true, y_pred, zero_division=0
        ),
        "anomaly_f1": f1_score(
            y_true, y_pred, zero_division=0
        ),
        "seed": seed,
        "dbscan_eps": 1.8,
        "dbscan_min_samples": 5,
    }

    return df, metrics

def evaluate_embedding_csv(path, eps=1.8, min_samples=5):
    """Evaluate DBSCAN clustering on an existing embedding CSV."""
    from pathlib import Path

    df = pd.read_csv(path)
    embedding_columns = [
        column for column in df.columns if column.startswith("emb_")
    ]

    if "host_id" not in df.columns:
        raise ValueError("CSV must contain a host_id column.")
    if not embedding_columns:
        raise ValueError("CSV must contain columns named emb_0, emb_1, ...")
    if df["host_id"].duplicated().any():
        raise ValueError("host_id values must be unique.")
    if df[embedding_columns].isna().any().any():
        raise ValueError("Embedding coordinates must not contain missing values.")

    X = df[embedding_columns].to_numpy(dtype=float)
    if not np.isfinite(X).all():
        raise ValueError("Embedding coordinates must all be finite.")

    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)

    non_noise = labels != -1
    cluster_ids = np.unique(labels[non_noise])
    n_non_noise = int(non_noise.sum())

    silhouette = np.nan
    davies_bouldin = np.nan
    if 2 <= len(cluster_ids) < n_non_noise:
        X_clustered = X[non_noise]
        labels_clustered = labels[non_noise]
        silhouette = silhouette_score(X_clustered, labels_clustered)
        davies_bouldin = davies_bouldin_score(
            X_clustered, labels_clustered
        )

    results = {
        "method": Path(path).stem,
        "evaluator": "DBSCAN",
        "n_hosts": len(df),
        "n_dimensions": len(embedding_columns),
        "n_clusters_excluding_noise": len(cluster_ids),
        "n_detected_outliers": int((labels == -1).sum()),
        "outlier_fraction": float((labels == -1).mean()),
        "silhouette": float(silhouette),
        "davies_bouldin": float(davies_bouldin),
        "dbscan_eps": eps,
        "dbscan_min_samples": min_samples,
        "input_file": str(path),
        "anomaly_precision": np.nan,
        "anomaly_recall": np.nan,
        "anomaly_f1": np.nan,
    }

    return pd.DataFrame([results]), pd.DataFrame({
        "host_id": df["host_id"],
        "cluster": labels,
        "is_dbscan_outlier": labels == -1,
    })

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(
        description="Evaluate DBSCAN on synthetic or existing embeddings."
    )
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to an existing embeddings CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="runs/clustering_anomaly",
        help="Directory for evaluation results.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.embeddings:
        results, assignments = evaluate_embedding_csv(args.embeddings)
        assignments_path = output_dir / "cluster_assignments.csv"
    else:
        embeddings, metrics = evaluate_synthetic_embeddings()
        results = pd.DataFrame([metrics])
        assignments = embeddings
        assignments_path = output_dir / "synthetic_embeddings.csv"

    result_path = output_dir / "clustering_anomaly_results.csv"

    results.to_csv(result_path, index=False)
    assignments.to_csv(assignments_path, index=False)

    print(results.to_string(index=False))
    print(f"\nSaved results: {result_path}")
    print(f"Saved assignments/data: {assignments_path}")

