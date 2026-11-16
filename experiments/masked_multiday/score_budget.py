"""Counterfactual score budget, NOT a trained model or a performance estimate.

Run from the repository root:
    python -m experiments.masked_multiday.score_budget
"""
from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path

from evaluation.scoreboard import overall_score


def budget(result: dict, target: float = 85.0) -> dict:
    windows = result.get("by_window")
    if not windows:
        raise ValueError("A per-window result is required")
    columns = [s["metric"] for s in next(iter(windows.values()))["summary"]]
    required = {"identity_top1", "partner_new_mrr"}
    if not required.issubset(columns):
        raise ValueError("Identity and New partners must be required metrics")
    for window in windows.values():
        if any(window["metrics"].get(k) is None for k in columns):
            raise ValueError("Incomplete baseline")

    def score(identity=None, mrr=None):
        candidate = copy.deepcopy(result)
        for window in candidate["by_window"].values():
            for key, value in (("identity_top1", identity), ("partner_new_mrr", mrr)):
                if value is not None:
                    window["metrics"][key] = value
        return float(overall_score([candidate], columns).iloc[0])

    scenarios = []
    for identity in (0.75, 0.80, 0.85, 0.90, 1.0):
        low, high = 0.0, 1.0
        feasible = score(identity, high) >= target
        if feasible:
            for _ in range(50):
                middle = (low + high) / 2
                if score(identity, middle) < target:
                    low = middle
                else:
                    high = middle
        scenarios.append({"identity_each_window": identity,
                          "required_mrr_each_window": high if feasible else None,
                          "score_at_mrr_0.55": score(identity, 0.55)})
    return {"kind": "counterfactual_not_measurement", "target": target,
            "assumptions": "Keep baseline Behavior, Links, Pairs, dimension and chance levels per window; set the same Identity/MRR in every window.",
            "baseline_score": score(), "perfect_identity_only_score": score(1.0),
            "scenarios": scenarios}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--result", type=Path)
    parser.add_argument("--commit", default="70347a3")
    parser.add_argument("--target", type=float, default=85.0)
    args = parser.parse_args()
    if not 0 < args.target <= 100:
        parser.error("target must be in (0, 100]")
    if args.result:
        result = json.loads(args.result.read_text())
    else:
        root = Path(__file__).resolve().parents[2]
        result = json.loads(subprocess.check_output(
            ["git", "show", f"{args.commit}:scoreboard/results/multiday_codebook_plus.json"], cwd=root))
    print(json.dumps(budget(result, args.target), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
