# Data pipeline: LANL netflow → windows → graph → frozen references

Raw LANL logs → canonical flows → train / validation / test windows → fixed host cohort →
per-window communication graphs → frozen evaluation references. Every step is a CLI and is
covered by tests. Nothing is invented about the LANL files: only documented fields are used.

## Source format (LANL Unified Host and Network Dataset, 2017, `netflow_day-NN.bz2`)

Documented at <https://csr.lanl.gov/data/2017/> (access requires the form on that page):
bzip2, comma-separated, **no header**, one record per line, fields in this order:

| # | field | meaning (verbatim from the source) |
|---|---|---|
| 1 | `Time` | start time of the event, "epoch time format" |
| 2 | `Duration` | duration in seconds |
| 3 | `SrcDevice` | the device that likely initiated the event |
| 4 | `DstDevice` | the receiving device |
| 5 | `Protocol` | the protocol number |
| 6 | `SrcPort` | port used by SrcDevice |
| 7 | `DstPort` | port used by DstDevice |
| 8 | `SrcPackets` | packets sent by SrcDevice |
| 9 | `DstPackets` | packets sent by DstDevice |
| 10 | `SrcBytes` | bytes sent by SrcDevice |
| 11 | `DstBytes` | bytes sent by DstDevice |

Devices are anonymised (`Comp132598`, some named such as `Mail`); non-well-known ports are
anonymised (`Port12597`). **The unit/origin of `Time` is not specified by the source**, so the
pipeline never converts it to a calendar date: windows are given in the same units as `Time`.
Use `describe` to read the real range before declaring windows.

The Windows host-event logs (`wls_day-NN.bz2`) are **not** supported here; the benchmark
works from netflow only.

### Canonical flows

`src, dst, dst_port, proto, bytes, packets, ts` (+ `src_port, duration, resp_bytes, resp_packets`):

`src=SrcDevice, dst=DstDevice, dst_port=DstPort, proto=Protocol, bytes=SrcBytes,
packets=SrcPackets, ts=Time`. Direction is originator → responder; the responder's counters
are kept as `resp_*` but the behavioural features (`evaluation.knn_consistency`) use only the
originator direction. `host_id` is the device string as found (stripped; case-sensitive;
`007` and `7` are different hosts).

Invalid rows (blank device, non-numeric/negative values, blank protocol/port) are dropped
and **counted by reason** in `manifest.json`; wrong field counts raise an error unless
`--lenient` is given (then they are counted).

## Run

```bash
# 1. look at the real Time range (no output files)
python -m pipeline.prepare describe --netflow data/raw/netflow_day-0*.bz2

# 2. declare windows BEFORE looking at any model scores (see pipeline/example_config.json)
python -m pipeline.prepare run --netflow data/raw/netflow_day-0*.bz2 \
    --config my_config.json --output-dir data/processed \
    --data-kind lanl --source-note "LANL download form, <date>, <who>"   # omit both => 'unverified'

# 3. freeze the evaluation references (validation and test targets)
python -m pipeline.reference --processed-dir data/processed --output-dir data/reference
```

Output of step 2 (`data/processed`): `flows/{train,validation,test}.csv`, `cohort.csv` (events per window and `evaluable_<window>`),
`graph/edges_<window>.csv`, `manifest.json` (schema, config, per-reason invalid counts,
SHA-256 of every input file, window statistics, leakage checks).

Output of step 3 (`data/reference/{validation,test}`): `features.csv`, `neighbors.csv`,
`edges.csv`, `pairs.csv`, `triplets.csv`, `eval_hosts.csv`, `manifest.json` (+ `labels.csv` only when external
labels are supplied with `--labels F --labels-source "…"`).

Models read only `ProcessedData.training_flows()` / `training_graph()` and the cohort:

```python
from pipeline.data import ProcessedData
d = ProcessedData("data/processed")
flows, graph, hosts = d.training_flows(), d.training_graph(), d.cohort
```

## Provenance: what the pipeline can and cannot claim

LANL publishes no checksums this pipeline can compare against, so origin is **never verified**.
Every processed dataset, reference and result carries one of three kinds (`pipeline/provenance.py`):

| `data_kind` | meaning |
|---|---|
| `synthetic_fixture` | `pipeline.fixtures` output (marker file, or the generator's device-ID signature even if the marker was deleted) |
| `unverified` | **default for any other input**: nothing is claimed about where the bytes came from |
| `lanl_declared` | the operator ran `--data-kind lanl --source-note "where/when obtained"`; file names must look like `netflow_day-NN.bz2` and the data must not match the fixture signature |

`provenance_verified` is always `false`; a declaration is recorded as a declaration, with the note,
the sanity-check results and per-file format statistics (`manifest.json: provenance`). Contradictory claims
(a marker or fixture signature together with `--data-kind lanl`, a LANL claim without a source note) are
refused. Reports and the dashboard print which kind applies; unverified or synthetic input is never
presented as LANL.

## Cohort policy and selection bias

Which hosts are evaluated is decided in `prepare` before any model exists:

| `cohort_policy` | cohort | evaluable hosts per later window |
|---|---|---|
| `train_only` (**default**) | hosts active in the **train** window only | the cohort hosts also active in that window; references and metrics use them, coverage is reported |
| `all_windows` | hosts active in **every** window | all (by construction) |

`all_windows` lets *future* activity decide who is in the benchmark: it is a transductive **stable-host**
benchmark. Models still receive no future flows or edges, but scores condition on host survival and
overstate persistence, identity and link persistence for prospective use. It is kept for comparison.
With `train_only`, hosts that vanish are simply not evaluable in that window; the manifest reports the
coverage and a survivorship diagnostic (median train-window activity of hosts active in every window vs
the others). A host-level cold-start or "disappeared host" analysis is not part of the metrics.

## Temporal design and leakage prevention

| window | used for |
|---|---|
| `train` | the only window models may fit on (flows, graph, features) |
| `validation` | tuning decisions; references in `reference/validation` |
| `test` | final reporting only; references in `reference/test` |

- Windows are half-open `[start, end)`, strictly ordered, non-overlapping (validated).
- Flows are assigned by start time and **dropped if still running at the window end**
  (`Time + Duration >= end`), so no flow carries information across a boundary.
- One **cohort** is fixed in `prepare`, before any model exists (policy above); every method is evaluated
  on the same hosts. The default policy uses no information from the validation or test windows.
- References are built from the *target* window (later than `train`), from logs only.
- Reference views are different questions and are never mixed (see `evaluation/` docs):
  behavioural neighbours (`neighbors.csv`), future links (`edges.csv`), shared-peer structure
  (`pairs.csv`, `triplets.csv`), and external labels (`labels.csv`, only if they exist).
- `pairs`/`triplets` judge only clear cases (positive: peer-set Jaccard ≥ τ, negative: 0);
  everything between is **unjudged and never treated as negative**.
- No role labels are generated. A linear probe is *unavailable* unless real external labels
  are supplied.

## Synthetic fixture (tests, CI, smoke runs)

`python -m pipeline.fixtures --output-dir data/fixture` writes files in the **exact LANL
layout** plus a `SYNTHETIC_FIXTURE` marker; everything built from them is stamped
`synthetic: true` and the marker cannot be overridden with `--data-kind lanl`. Results on
the fixture say nothing about LANL.

## What is needed to run the real experiment

1. Download `netflow_day-NN.bz2` files via the LANL form (one file is one day; a few days
   suffice for a first run). The download links issued by the form are tokenised and expire.
2. `describe` them, choose train/validation/test windows in `Time` units, run steps 2–3.
3. Provide embeddings from the real methods (`host_id, emb_*`, one file per method trained
   on the `train` window only), or use the baselines in `benchmark/baselines`.

## Credit

The first outline of a "normalised events → graph → reference" builder came from PR #37
(`kalivixZ`); this pipeline replaces its single-window prototype with a LANL reader,
temporal windows, a fixed cohort and leakage checks.

## Tests

`python -m pytest -q pipeline`
