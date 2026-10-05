# Benchmark demo (Streamlit prototype)

Upload an `embeddings.csv` and immediately see how it scores against simple baselines,
using the shared benchmark code (`benchmark/validate_submission.py`, `evaluation/evaluate.py`,
`benchmark/baselines`). No metric logic lives in the app.

## Run

From the repository root:

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py
```

## Use

1. Optionally download **sample embeddings.csv** from the sidebar to see the format.
2. Upload your `embeddings.csv` (`host_id, emb_0, emb_1, ...`).
3. Press **Run evaluation**.

You get:
- k-NN Consistency, Jaccard and nDCG at k = 5 / 10 / 20 for your embedding;
- grouped bar charts comparing it with `random`, `pca`, `svd` and `reference ceiling`
  (baselines use your embedding's dimension, capped by the reference size);
- a leaderboard table sorted by Consistency@10;
- a 2D PCA projection of your embedding; hover a point to see its `host_id`;
- a list of metrics that are not available yet (#22-#25), shown as unavailable, never estimated.

Malformed files (empty, missing `host_id`, no `emb_*` columns, duplicate IDs, NaN or
non-numeric values, unreadable text) produce a clear error message instead of a crash.

## Important

- The default reference is **synthetic demo data** (300 hosts `h0`..`h299`, 5 roles),
  generated on start-up. Your file's `host_id` values must match it; use the sample file
  as a template. Scores are for exploring the benchmark, not real results.
- The reference ceiling and PCA/SVD are built from the reference features, so they score
  high by construction.
- Nothing is saved: results are not written to the repository.

## Layout

- `streamlit_app.py` - UI only.
- `core.py` - validation, evaluation, leaderboard and projection (no Streamlit import).
- `test_app.py` - tests; run `python -m pytest -q`.
