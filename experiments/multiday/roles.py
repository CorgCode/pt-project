"""Профиль сервис/роль по обоим портам и статистические отпечатки потоков."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

from experiments.multiday.gain import MultiDayEncoder, RECIPES, check_chronology, evaluate, counts_from_view
from experiments.overnight.model import T
from experiments.overnight.search import safe_to_continue
from pipeline.data import ProcessedData


def service_roles(flows, hosts):
    """Правило Родиона: меньший числовой порт <1024, иначе other/анонимный dst."""
    sp = pd.to_numeric(flows.src_port, errors="coerce")
    dp = pd.to_numeric(flows.dst_port, errors="coerce")
    dst_server = (dp < 1024) & (sp.isna() | (dp < sp))
    src_server = (sp < 1024) & (dp.isna() | (sp < dp))
    anonymous = sp.isna() & dp.isna()
    service = pd.Series("other", index=flows.index, dtype=object)
    service.loc[dst_server] = dp.loc[dst_server].astype(int).astype(str)
    service.loc[src_server] = sp.loc[src_server].astype(int).astype(str)
    service.loc[anonymous] = "a:" + flows.loc[anonymous, "dst_port"].astype(str)
    role_src = np.where(src_server, "server", np.where(dst_server | anonymous, "client", "neutral"))
    role_dst = np.where(dst_server | anonymous, "server", np.where(src_server, "client", "neutral"))
    service = flows.proto.astype(str) + "/" + service
    parts = []
    index = pd.Index(hosts)
    for end, role in [("src", role_src), ("dst", role_dst)]:
        keep = flows[end].isin(index)
        parts.append(pd.DataFrame({"h": flows.loc[keep, end].astype(str),
                                   "t": service.loc[keep] + "/" + role[keep.to_numpy()]}))
    return pd.concat(parts).groupby(["h", "t"], observed=True, sort=False).size()


class RoleMixin:
    def view(self, flows, hosts, seconds):
        out = super().view(flows, hosts, seconds)
        out["counts"]["services"] = service_roles(flows, hosts)
        moments = []
        for end in ["src", "dst"]:
            stats = flows.groupby(end, observed=True)[list(T.STATS)].agg(["mean", "std"])
            stats = stats.reindex(hosts).fillna(0)
            moments.append(np.log1p(stats.to_numpy()))
        out["activity"] = np.hstack([out["activity"], *moments])
        return out


class RoleEncoder(RoleMixin, MultiDayEncoder):
    def _features(self, view):
        out = super()._features(view)
        out["_counts"] = counts_from_view(view, self.hosts, self.vocab["peers"])
        return out

    def train_head(self, data, kind, epochs):
        if kind != "contrastive":
            return super().train_head(data, kind, epochs)
        self.encoder, self.epochs, self.posterior_mix = "mlp", epochs, 0.0
        old_splits = self.splits
        xs, ok = self._xs[::3], self._ok[::3]
        self.splits = len(xs)
        active = np.flatnonzero(ok.all(0) if self.splits == 2 else ok.sum(0) >= 2)
        try:
            self._fit_mlp(data, xs, ok, active)
        finally:
            self.splits = old_splits
        self._parts_cache = {}
        return self


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--extra", nargs="+", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("runs/multiday_roles"))
    ap.add_argument("--validation-window", default="test_day04")
    ap.add_argument("--reference-dir", type=Path, default=Path("scoreboard/data/reference/test_day04"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    data = ProcessedData(args.base)
    training = [data, *[ProcessedData(p) for p in args.extra]]
    check_chronology(training, T._span(data, args.validation_window)[0])
    model = RoleEncoder().fit_days(training)
    ref = args.reference_dir
    best = -np.inf
    for kind, epochs in [("contrastive", 40), ("classifier", 80)]:
        if not safe_to_continue():
            raise RuntimeError("Недостаточно ресурсов")
        model.train_head(data, kind, epochs)
        for mix in ([0.0] if kind == "contrastive" else [0.0, 1.0]):
            model.posterior_mix, model._parts_cache = mix, {}
            for i, recipe in enumerate(RECIPES[:2]):
                model.recipe, model.pca_dim = recipe, 384
                model.fit_pca(data)
                name = f"roles_{kind}_{mix}_{i}"
                score, result = evaluate(model, data, args.validation_window, ref, args.out / name, name)
                row = {"name": name, "score": score, "identity": result["metrics"]["identity_top1"],
                       "kind": kind, "epochs": epochs, "mix": mix, "recipe": recipe,
                       "has_training_graphs": all(c is not None for c in model._training_counts)}
                with (args.out / "trials.jsonl").open("a") as out:
                    out.write(json.dumps(row) + "\n")
                print(json.dumps(row), flush=True)
                if score > best:
                    best = score
                    with (args.out / "best.pkl").open("wb") as out:
                        cloudpickle.dump(model, out)
                    (args.out / "best.json").write_text(json.dumps(row, indent=2) + "\n")


if __name__ == "__main__":
    from experiments.multiday.roles import main as run_main
    run_main()
