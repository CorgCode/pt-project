"""Temporal denoising autoencoder: temporal contrastive learning plus reconstruction.

Run:
    PT_DATA=/path/to/lanl/processed python scoreboard/temporal_dae.py --tune

The model is fitted only on train. Hyperparameters are selected on validation;
test windows are evaluated once after selection.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from pipeline.data import ProcessedData  # noqa: E402
from temporal_encoder import (  # noqa: E402
    GRID, RECIPE, SEED, PCA_DIM, BATCH, TEMPERATURE,
    TemporalEncoder, _encoder, _recipes,
)
from sklearn.preprocessing import normalize  # noqa: E402

NAME = "temporal_dae"
RECONSTRUCTION_GRID = (0.03, 0.1, 0.3)


class TemporalDAE(TemporalEncoder):
    """TemporalEncoder with a decoder that preserves information in each temporal view."""

    def __init__(self, *args, reconstruction_weight: float = 0.1, **kwargs):
        super().__init__(*args, **kwargs)
        if reconstruction_weight < 0:
            raise ValueError("reconstruction_weight must be >= 0")
        self.reconstruction_weight = float(reconstruction_weight)

    def _fit_mlp(self, data, xs, ok, active):
        import torch
        from torch import nn
        from torch.nn import functional as F

        torch.set_num_threads(2)
        xs = [torch.from_numpy(x) for x in xs]
        pair_rng = np.random.default_rng(SEED + 1)

        self.net = _encoder(xs[0].shape[1])
        # The encoder output remains the embedding; this auxiliary decoder is used only in training.
        self.net.decoder = nn.Sequential(
            nn.Linear(128, 256), nn.GELU(), nn.Linear(256, xs[0].shape[1])
        )
        opt = torch.optim.AdamW(self.net.parameters(), lr=1e-3, weight_decay=1e-4)

        behavior = normalize(self.beh.embed(data, "train", self.hosts).to_numpy()).astype(np.float32)
        services = normalize(self._train["services"]).astype(np.float32)
        teacher = torch.from_numpy(np.hstack([behavior, services]) / np.float32(np.sqrt(2)))
        rng = np.random.default_rng(SEED)

        for _ in range(self.epochs):
            self.net.train()
            order = rng.permutation(active)
            for off in range(0, len(order), BATCH):
                batch = order[off:off + BATCH]
                i, j = (0, 1) if self.splits == 2 else pair_rng.choice(self.splits, 2, replace=False)
                ix = torch.from_numpy(batch[ok[i, batch] & ok[j, batch]])
                if len(ix) < 2:
                    continue

                xa, xb = xs[i][ix], xs[j][ix]
                za, zb = self.net(xa), self.net(xb)
                logits, labels = za @ zb.T / TEMPERATURE, torch.arange(len(ix))
                contrastive = 0.5 * (
                    F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)
                )
                target = teacher[ix] @ teacher[ix].T
                relational = 0.5 * (
                    F.mse_loss(za @ za.T, target) + F.mse_loss(zb @ zb.T, target)
                )
                reconstruction = 0.5 * (
                    F.mse_loss(self.net.decoder(za), xa) +
                    F.mse_loss(self.net.decoder(zb), xb)
                )
                loss = contrastive + 0.5 * relational + self.reconstruction_weight * reconstruction

                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
                opt.step()


def write_embeddings(model: TemporalDAE, data: ProcessedData, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    submit.keep_vectors_local(out)
    jobs = [("train", data.cohort, "embeddings.csv")]
    jobs.extend((w, data.evaluable(w), submit.window_file(w)) for w in data.test_windows)
    if "validation" in data.windows:
        jobs.append(("validation", data.evaluable("validation"), "validation.csv"))
    for window, hosts, filename in jobs:
        emb = model.embed(data, window, hosts)
        emb.to_csv(out / filename, float_format="%.5g")
        print(f"{window}: {len(emb)} hosts x {emb.shape[1]} dimensions")


def tune(data: ProcessedData):
    """Tune reconstruction weight, temporal splits and block weights on validation only."""
    rows = []
    best = None
    with tempfile.TemporaryDirectory() as tmp:
        for reconstruction_weight in RECONSTRUCTION_GRID:
            for splits in GRID["splits"]:
                model = TemporalDAE(
                    reconstruction_weight=reconstruction_weight, splits=splits, pca_dim=PCA_DIM
                ).fit(data)
                for recipe in _recipes(GRID):
                    model.recipe = recipe
                    model.fit_pca(data)
                    out = Path(tmp) / "candidate"
                    out.mkdir(exist_ok=True)
                    model.embed(data, "train", data.cohort).to_csv(out / "embeddings.csv")
                    model.embed(data, "validation", data.evaluable("validation")).to_csv(out / "validation.csv")
                    score, tasks = submit.validation_score(out / "embeddings.csv", out / "validation.csv")
                    row = {
                        "reconstruction_weight": reconstruction_weight,
                        "splits": splits,
                        **{k: recipe[k] for k in ("behavioral", "neural", "diffusion")},
                        "validation_score": round(score, 3),
                        "tasks": {k: round(v, 2) for k, v in tasks.items()},
                    }
                    rows.append(row)
                    print(row, flush=True)
                    if best is None or score > best[0]:
                        best = (score, reconstruction_weight, splits, recipe)

    score, reconstruction_weight, splits, recipe = best
    print(
        f"Best validation score={score:.2f}; reconstruction_weight={reconstruction_weight}; "
        f"splits={splits}; recipe={recipe}",
        flush=True,
    )
    model = TemporalDAE(
        reconstruction_weight=reconstruction_weight, splits=splits, recipe=recipe, pca_dim=PCA_DIM
    ).fit(data)
    return model, rows, {
        "reconstruction_weight": reconstruction_weight,
        "splits": splits,
        "recipe": recipe,
        "pca_dim": PCA_DIM,
        "tuned_on": "validation only",
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tune", action="store_true", help="select settings using validation")
    args = parser.parse_args()

    try:
        import torch  # noqa: F401
    except ImportError as exc:
        raise SystemExit("temporal_dae requires PyTorch: pip install torch") from exc

    data = ProcessedData(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    if "validation" not in data.windows:
        raise SystemExit("temporal_dae tuning requires a validation window")

    if args.tune:
        model, rows, params = tune(data)
    else:
        model = TemporalDAE().fit(data)
        rows = []
        params = {
            "reconstruction_weight": model.reconstruction_weight,
            "splits": model.splits,
            "recipe": model.recipe,
            "pca_dim": model.pca_dim,
            "tuned_on": "fixed defaults; not tuned",
        }

    out = ROOT / "scoreboard" / "submissions" / NAME
    write_embeddings(model, data, out)
    (out / "SHARED_COORDINATES").touch()
    (out / "AUTHOR").write_text("kalivixZ\\n", encoding="utf-8")
    (out / "params.json").write_text(json.dumps(params, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    if rows:
        (out / "tuning.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\\n", encoding="utf-8")
    result = submit.score(out)
    submit.render()
    print(
        NAME + ": " + ", ".join(
            f"{item['title']} = {item['value']:.3f}"
            for item in result["summary"] if item["value"] is not None
        )
    )


if __name__ == "__main__":
    main()
