"""Логика демо без интерфейса. Метрики — из общей оценки."""
from __future__ import annotations

import contextlib
import io
import tempfile
import warnings
from pathlib import Path

import pandas as pd
from sklearn.decomposition import PCA

from benchmark.baselines.make_synthetic_reference import make
from benchmark.baselines.run_all import generate
from evaluation.validate import validate_submission
from evaluation.evaluate import run

KS = (5, 10, 20)
METRICS = {"consistency": "k-NN Consistency", "jaccard": "Jaccard", "ndcg": "nDCG"}
UPLOADED = "uploaded"
BASELINES = {"random": "random", "pca": "pca", "svd": "svd", "reference_identity": "reference ceiling"}


class UploadError(ValueError):
    """Загруженный файл некорректен; сообщение можно показать пользователю."""


def make_demo_reference(directory: Path, n_hosts: int = 300, seed: int = 0) -> Path:
    """Синтетический reference/features.csv (только для демо)."""
    make(Path(directory), n_hosts=n_hosts, seed=seed)
    return Path(directory)


def validate_upload(data: bytes, workdir: Path) -> Path:
    """Сохранить загруженный файл и проверить его общим валидатором."""
    if not data.strip():
        raise UploadError("The file is empty.")
    path = Path(workdir) / "embeddings.csv"
    path.write_bytes(data)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            validate_submission(str(path))
    except (ValueError, pd.errors.ParserError, pd.errors.EmptyDataError, UnicodeDecodeError) as exc:
        raise UploadError(f"Invalid embeddings.csv: {exc}") from exc
    return path


def check_hosts(path: Path, reference_dir: Path) -> None:
    emb = set(pd.read_csv(path)["host_id"].astype(str))
    ref = set(pd.read_csv(Path(reference_dir) / "features.csv")["host_id"].astype(str))
    if emb != ref:
        raise UploadError(
            "host_id values do not match the demo reference "
            f"({len(emb - ref)} unknown, {len(ref - emb)} missing out of {len(ref)} reference hosts). "
            "Use the sample file as a template."
        )


def run_comparison(path: Path, reference_dir: Path, workdir: Path, seed: int = 0) -> dict[str, dict]:
    """Оценить загруженный файл и бейзлайны (той же размерности) общей оценкой."""
    path, workdir = Path(path), Path(workdir)
    dim = len([c for c in pd.read_csv(path, nrows=1).columns if c.startswith("emb_")])
    with warnings.catch_warnings():  # для pca/svd размерность ограничена числом признаков
        warnings.simplefilter("ignore")
        baseline_paths = generate(reference_dir, workdir / "baselines", dim=dim, seed=seed)
    results = {UPLOADED: run(path, reference_dir, UPLOADED)}
    for name, p in baseline_paths.items():
        results[BASELINES[name]] = run(p, reference_dir, BASELINES[name])
    return results


def leaderboard(results: dict[str, dict]) -> pd.DataFrame:
    """Методы x metric@k (среднее), по убыванию Consistency@10."""
    rows = []
    for method, res in results.items():
        row = {"method": method}
        for m in METRICS:
            for k in KS:
                row[f"{m}@{k}"] = res["metrics"].get(f"{m}@{k}_mean")
        rows.append(row)
    df = pd.DataFrame(rows).set_index("method")
    return df.sort_values("consistency@10", ascending=False)


def metric_long(board: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Длинный формат (метод, k, значение) для графика одной метрики."""
    cols = [f"{metric}@{k}" for k in KS]
    out = board[cols].copy()
    out.columns = [str(k) for k in KS]
    return out.reset_index().melt(id_vars="method", var_name="k", value_name=METRICS[metric])


def unavailable(results: dict[str, dict]) -> list[str]:
    return results[UPLOADED]["unavailable_metrics"]


def pca_projection(path: Path) -> pd.DataFrame:
    """2D PCA загруженного эмбеддинга, индекс — host_id."""
    df = pd.read_csv(path)
    X = df[[c for c in df.columns if c.startswith("emb_")]].to_numpy(float)
    if X.shape[1] < 2 or X.shape[0] < 2:
        raise UploadError("Need at least 2 hosts and 2 embedding dimensions for a 2D projection.")
    xy = PCA(n_components=2, random_state=0).fit_transform(X)
    return pd.DataFrame({"host_id": df["host_id"].astype(str), "PC1": xy[:, 0], "PC2": xy[:, 1]})


def sample_embedding(reference_dir: Path, workdir: Path, dim: int = 8, seed: int = 0) -> bytes:
    """Корректный пример файла (бейзлайн PCA), чтобы попробовать демо."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        paths = generate(reference_dir, Path(workdir) / "sample", dim=dim, seed=seed, methods=["pca"])
    return paths["pca"].read_bytes()


def evaluate_upload(data: bytes, reference_dir: Path, workdir: Path | None = None) -> dict:
    """Проверка -> хосты -> сравнение. Возвращает всё, что нужно интерфейсу."""
    with tempfile.TemporaryDirectory() if workdir is None else contextlib.nullcontext(workdir) as wd:
        wd = Path(wd)
        path = validate_upload(data, wd)
        check_hosts(path, reference_dir)
        results = run_comparison(path, reference_dir, wd)
        return {
            "results": results,
            "board": leaderboard(results),
            "unavailable": unavailable(results),
            "projection": pca_projection(path) if _has_2d(path) else None,
        }


def _has_2d(path: Path) -> bool:
    return len([c for c in pd.read_csv(path, nrows=1).columns if c.startswith("emb_")]) >= 2
