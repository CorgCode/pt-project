# Experiments

Run every method on the SAME cohort, windows, references and candidate pairs.

```bash
# everything from raw netflow (real data) ...
python -m experiments.end_to_end --netflow data/raw/netflow_day-0*.bz2 \
    --config my_config.json --workdir runs/lanl [--submissions-dir submissions]

# ... or a smoke run on a synthetic LANL-format fixture (no download)
python -m experiments.end_to_end --fixture --workdir runs/synthetic_demo
```

Stages (also usable separately): `pipeline.prepare` → `pipeline.reference` →
`experiments.run_experiment`. Output of the last stage in `<workdir>/run`:

| path | content |
|---|---|
| `report.md`, `figures/` | tables with bootstrap CIs, scorecards, identity and audit plots |
| `results/<target>/<method>.json` | full evaluator output (metrics, CIs, details, reference fingerprints) |
| `identity/<target>/<method>.json` | identity retrieval, raw and Procrustes |
| `leaderboard_<target>.csv` | built by `benchmark.build_leaderboard` (refuses mixed references) |
| `audit.json` | weight sensitivity, window stability, random-null bands, prevalence, overlap |
| `manifest.json` | data kind (synthetic/real), cohort/reference hashes, seeds, environment, git SHA |

Methods: `random`, `behavioral`, `behavioral_pca`, `behavioral_svd`, `graph_spectral` (all fitted on the
train window only), the leaking `reference_ceiling`, and anything in `--submissions-dir/<method>/embeddings.csv`
(source-window embedding; `<method>/validation.csv` / `<method>/test.csv` enable identity retrieval;
an empty `SHARED_COORDINATES` file declares one coordinate system). Expected methods that were not supplied
(node2vec, deepwalk, graphsage, struc2vec, rolx) are listed as missing, never simulated.

Read `docs/METHODOLOGY.md` before interpreting any number.
