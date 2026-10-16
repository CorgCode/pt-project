# Experiments

Install dependencies once: `pip install -r requirements.txt`.

Run every method on the SAME cohort, windows, references and candidate pairs.

```bash
# everything from raw netflow (real data) ...
python -m experiments.end_to_end --netflow data/raw/netflow_day-0*.bz2 \
    --config my_config.json --workdir runs/lanl [--submissions-dir submissions] \
    --data-kind lanl --source-note "LANL form, <date>, <who>"

# ... without --data-kind lanl the run is labelled `unverified`, never LANL.
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
| `scoreboard_<target>.csv` | built by `benchmark.build_leaderboard` (refuses mixed references) |
| `audit.json` | weight sensitivity, window stability, random-null bands, prevalence, overlap |
| `manifest.json` | data kind (synthetic/real), cohort/reference hashes, seeds, environment, git SHA |

**Provenance label.** Every run is `synthetic_fixture`, `unverified` (default) or `lanl_declared`; the report,
the manifest and the dashboard state which, and that provenance is *declared, never verified*.

**Cohort.** `--cohort-policy train_only` (default) evaluates the hosts active in the train window and scores
each later window on the hosts still active there (coverage is reported); `all_windows` is the transductive
stable-host variant and the report says so.

Methods: `random`, `behavioral`, `behavioral_pca`, `behavioral_svd`, `graph_spectral` (all fitted on the
train window only), the leaking `reference_ceiling`, and anything in `--submissions-dir/<method>/embeddings.csv`
(source-window embedding; `<method>/validation.csv` / `<method>/test.csv` enable identity retrieval;
an empty `SHARED_COORDINATES` file declares one coordinate system). A `provenance.json`
`{"trained_on_windows": ["train"]}` is required to claim the method used no later window: a declaration that
includes `validation` or `test` is refused (temporal leakage); a missing file is flagged as *undeclared* in the
report. These declarations cannot be verified by code. Expected methods that were not supplied
(node2vec, deepwalk, graphsage, struc2vec, rolx) are listed as missing, never simulated.

Read `docs/METHODOLOGY.md` before interpreting any number.

## Feature ablation (port-free baseline, issue #43)

```bash
python -m experiments.feature_ablation --processed-dir data/processed --reference-dir data/reference --out-dir runs/ablation
```

Compares the behavioural profile with and without each feature block on identical hosts, windows and references; writes
`ablation.json` and `ablation.md` (paired confidence intervals for full vs port-free, leave-one-block-out table, a
port-free-reference diagnostic, and the methodological implications). The report carries the data-kind banner
(synthetic / unverified / declared LANL).
