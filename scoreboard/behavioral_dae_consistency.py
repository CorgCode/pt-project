"""Score the consistency-regularised behavioural denoising autoencoder.

    PT_DATA=/path/to/lanl/processed python scoreboard/behavioral_dae_consistency.py

Uses two independently noised views of each train profile and a consistency penalty
between their latent embeddings. Writes embeddings to its own submission directory.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from methods.baselines.autoencoder import BehavioralDAEConsistency  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402

NAME = "behavioral_dae_consistency"
AUTHOR = "kalivixZ"
DIM = 16
HIDDEN_DIM = 64
EPOCHS = 30
BATCH_SIZE = 256
SEED = 17
CONSISTENCY_WEIGHT = 0.1


def write_embeddings(model: BehavioralDAEConsistency, data: ProcessedData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    submit.keep_vectors_local(out)

    jobs = [("train", list(data.cohort), "embeddings.csv")]
    jobs.extend((window, data.evaluable(window), submit.window_file(window))
                for window in data.test_windows)
    if "validation" in data.windows:
        jobs.append(("validation", data.evaluable("validation"), "validation.csv"))

    for window, hosts, filename in jobs:
        embedding = model.embed(data, window, hosts)
        if embedding.shape != (len(hosts), DIM):
            raise RuntimeError(
                f"{NAME}: окно {window}: ожидалась форма {(len(hosts), DIM)}, получено {embedding.shape}"
            )
        embedding.to_csv(out / filename, float_format="%.7g")
        print(f"{window}: {len(embedding)} hosts x {embedding.shape[1]} dimensions")

    (out / "SHARED_COORDINATES").touch()
    if AUTHOR.strip():
        (out / "AUTHOR").write_text(AUTHOR.strip().lstrip("@") + "\n", encoding="utf-8")


def main() -> None:
    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("Для behavioral_dae_consistency нужен PyTorch: pip install torch") from exc

    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    model = BehavioralDAEConsistency(
        dim=DIM,
        hidden_dim=HIDDEN_DIM,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        seed=SEED,
        consistency_weight=CONSISTENCY_WEIGHT,
    ).fit(data)
    print(
        f"Обучение завершено: reconstruction MSE "
        f"{model.reconstruction_loss_history_[0]:.5f} -> "
        f"{model.reconstruction_loss_history_[-1]:.5f}; "
        f"consistency MSE {model.consistency_loss_history_[0]:.5f} -> "
        f"{model.consistency_loss_history_[-1]:.5f}"
    )

    out = ROOT / "scoreboard" / "submissions" / NAME
    write_embeddings(model, data, out)
    result = submit.score(out)
    submit.render()
    print("\n" + NAME + ": " + ", ".join(
        f"{item['title']} = {item['value']:.3f}"
        for item in result["summary"] if item["value"] is not None
    ))
    print("Общая таблица обновлена локально: scoreboard/README.md")


if __name__ == "__main__":
    main()
