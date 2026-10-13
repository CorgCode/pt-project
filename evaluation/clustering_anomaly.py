
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


if __name__ == "__main__":
    embeddings, metrics = evaluate_synthetic_embeddings()

    pd.DataFrame([metrics]).to_csv(
        "clustering_anomaly_results.csv",
        index=False,
    )
    embeddings.to_csv("synthetic_embeddings.csv", index=False)

    print(pd.Series(metrics).to_string())
    print("\nSaved clustering_anomaly_results.csv")
    print("Saved synthetic_embeddings.csv")
