# Baseline embeddings

Simple baselines for testing and calibrating the common benchmark before real models exist.
They write the standard submission format (`host_id, emb_0, emb_1, ...`) and are scored by the
existing evaluator; no metric logic lives here. Column names and the host-ID convention come from
`benchmark/config.json`.

| method | what it is | role |
|---|---|---|
| `random` | seeded Gaussian noise, `dim` columns | floor; should sit at chance level |
| `pca` | PCA of `reference/features.csv`, `dim` columns | meaningful reduced baseline |
| `svd` | TruncatedSVD of `reference/features.csv`, `dim` columns | meaningful reduced baseline |
| `reference_identity` | `ref_*` features copied to `emb_*` unchanged | ceiling (leaks by design) |

Notes:
- `dim` is configurable (`--dim`, default 16) and capped to the data's maximum with a warning.
  `reference_identity` always keeps all reference columns.
- `--seed` (default 0) makes every method reproducible.
- `pca` and `svd` give the same embedding up to sign when the reference features are
  already column-centered (e.g. standardized); they differ on uncentered data.
- Baselines built from the reference features score high on neighbourhood metrics **by
  construction**. They are calibration points, not results to compare real models against
  as if independent.

## Usage

From the repository root:

```bash
python -m benchmark.baselines.run_all --reference-dir reference --dim 16 --seed 0 --evaluate
python benchmark/build_leaderboard.py
```

This writes `submissions/baselines/<method>/embeddings.csv`, then (with `--evaluate`) runs the
common evaluator and writes `benchmark/results/baseline_<method>.json`. Options: `--methods`,
`--out-dir`, `--results-dir`. Metrics whose reference artifacts or implementations are not
ready (#22-#25) stay listed under `unavailable_metrics`; nothing is invented.

## Synthetic reference (calibration / CI only)

Until the real `reference/features.csv` exists (#19), generate a synthetic one:

```bash
python -m benchmark.baselines.make_synthetic_reference --output-dir /tmp/synthetic_reference
python -m benchmark.baselines.run_all --reference-dir /tmp/synthetic_reference --out-dir /tmp/sub --results-dir /tmp/res --evaluate
```

Do not commit synthetic references, generated embeddings or results as real benchmark output.

## Tests

```bash
python -m pytest -q
```


## Port-free behavioural baseline (issue #43)

`temporal.py` also provides `BehavioralPortFree` and `BehavioralPortFreePCA` (names `behavioral_portfree`,
`behavioral_portfree_pca`): the train-window behavioural profile with the port block removed entirely. They are
off by default; `--port-free-baselines` (in `experiments.run_experiment` / `end_to_end`) adds them as extra rows on the
same references, hosts and windows. The full feature ablation (every block, paired intervals, a port-free-reference
diagnostic and the methodological caveats) is `python -m experiments.feature_ablation`; see `docs/METHODOLOGY.md` 4.12.
