import argparse
from pathlib import Path

import pandas as pd


def validate_submission(path: str) -> None:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Submission not found: {file_path}")

    df = pd.read_csv(file_path)

    if "host_id" not in df.columns:
        raise ValueError("Missing required column: host_id")

    embedding_columns = [column for column in df.columns if column.startswith("emb_")]
    if not embedding_columns:
        raise ValueError("No embedding columns found. Expected columns named emb_0, emb_1, ...")

    if df["host_id"].isna().any():
        raise ValueError("host_id contains missing values")

    if df["host_id"].duplicated().any():
        duplicates = df.loc[df["host_id"].duplicated(), "host_id"].tolist()
        raise ValueError(f"Duplicate host_id values: {duplicates[:5]}")

    if df[embedding_columns].isna().any().any():
        raise ValueError("Embedding contains NaN or missing values")

    non_numeric = [
        column for column in embedding_columns
        if not pd.api.types.is_numeric_dtype(df[column])
    ]
    if non_numeric:
        raise ValueError(f"Embedding columns must be numeric: {non_numeric}")

    print(f"OK: {len(df)} hosts, embedding dimension = {len(embedding_columns)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate a host embedding submission")
    parser.add_argument("submission", help="Path to embeddings.csv")
    args = parser.parse_args()
    validate_submission(args.submission)


if __name__ == "__main__":
    main()
