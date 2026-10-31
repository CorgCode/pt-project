"""Подготовить разреженную память и проверить гипотезу на прежнем validation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from evaluation.identity import identity_retrieval
from experiments.fingerprint_memory import FingerprintBank, read_fingerprints
from experiments.multiday_gain import check_chronology
from experiments.overnight.model import T
from pipeline.data import ProcessedData


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    work = args.work
    work.mkdir(parents=True, exist_ok=True)
    old = Path("/Volumes/KINGSTON/PT_PROJECT_ML/multiday_gain")
    base = ProcessedData(Path("/Volumes/KINGSTON/PT_PROJECT_ML/multiday/processed"))
    paths = [base.root, Path("/Volumes/KINGSTON/PT_PROJECT_ML/multiday2d/day03"),
             *[old / f"day{day:02d}" for day in range(4, 10)]]
    training = [ProcessedData(p) for p in paths]
    check_chronology(training, 9 * 86400 + 12.5 * 3600)
    observations = []
    for day, data in zip(range(2, 10), training):
        counts = read_fingerprints(data, "train", base.cohort, work / "count_cache")
        observations.append(counts)
        print(f"day{day:02d}: отпечатки готовы, context {len(counts['counts']['context']):,}", flush=True)
        (work / "extraction_state.json").write_text(json.dumps({"last_completed_day": day, "total_days": 8}) + "\n")
    # Пилот использует ТОЛЬКО дни02–07, несмотря на наличие более поздних данных.
    pilot = FingerprintBank().fit(observations[:6], base.cohort)
    validation = ProcessedData(old / "paired_validation")
    hosts = validation.evaluable("validation")
    rows = pd.Index(base.cohort).get_indexer(hosts)
    later = pilot.transform(read_fingerprints(validation, "validation", base.cohort, work / "count_cache"))[rows]
    earlier = pilot.prototype[rows]
    results = []
    for mode, temperature in [("direct", 0.05), ("memory", 0.05), ("memory", 0.1)]:
        if mode == "direct":
            a, b = pilot.direct(earlier), pilot.direct(later)
            accuracy = None
        else:
            a = pilot.posterior(earlier, temperature)[0]
            b, predicted = pilot.posterior(later, temperature)
            accuracy = float(np.mean(predicted == rows))
        result = identity_retrieval(pd.DataFrame(a, index=hosts), pd.DataFrame(b, index=hosts),
                                    alignment="procrustes", seed=0, n_boot=0)
        row = {"mode": mode, "temperature": temperature, "identity_top1": result["top1"],
               "classification_accuracy": accuracy, "note": "diagnostic on validation08, not final test"}
        results.append(row)
        print(json.dumps(row), flush=True)
    (work / "probe_validation08.json").write_text(json.dumps(results, indent=2) + "\n")
    del pilot
    bank = FingerprintBank().fit(observations, base.cohort)
    with (work / "bank.pkl").open("wb") as stream:
        cloudpickle.dump(bank, stream)
    (work / "bank_metadata.json").write_text(json.dumps({"training_sources": list(map(str, paths)),
        "training_days": list(range(2, 10)), "hosts": len(base.cohort),
        "peer_vocabulary": len(bank.vocab['peers']), "context_vocabulary": len(bank.vocab['context'])}, indent=2) + "\n")
    print("Память восьми дней сохранена; день10/11 не использован для её обучения", flush=True)


if __name__ == "__main__":
    main()
