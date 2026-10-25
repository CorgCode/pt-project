#!/bin/bash
# Train на двух днях: день 02 11:00–12:00 + день 03 11:00–12:00. Когорта, эталоны и test — как в основной таблице.
#   NOTE="форма LANL, <дата>, <кто>" PY=/usr/local/bin/python3 experiments/lanl/two_days.sh /путь/к/multiday
# Нужна папка после multiday.sh (processed/ и raw/netflow_day-03.bz2). Таблица основной ветки не меняется:
# обучение идёт во временном git worktree, итог — <папка>/two_days/README.md.
# Честное сравнение — только день 04: test дня 02 лежит между часами train, test дня 03 — сразу после.
set -eu
M=${1:?папка multiday}
PY=${PY:-python}
NOTE=${NOTE:?NOTE — откуда данные: «форма LANL, <дата>, <кто>»}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
B=$M/two_days
mkdir -p "$B"
cd "$REPO"

cat > "$B/config_day03.json" <<'EOF'
{"windows": {"train": [212400, 216000], "validation": [216000, 217800], "test": [217800, 221400]},
 "min_events_per_window": 300, "min_cohort_size": 25, "drop_straddling": true, "seed": 0, "cohort_policy": "train_only"}
EOF
[ -f "$B/day03/flows/train.csv" ] || "$PY" -m pipeline.prepare run --netflow "$M/raw/netflow_day-03.bz2" \
    --config "$B/config_day03.json" --output-dir "$B/day03" --data-kind lanl \
    --source-note "$NOTE" --overwrite

P=$B/processed
if [ ! -f "$P/flows/train.csv" ]; then
  mkdir -p "$P/flows" "$P/graph"
  cp "$M/processed/cohort.csv" "$P/"
  for f in validation test test_day03 test_day04; do
    ln -sf "$M/processed/flows/$f.csv" "$P/flows/$f.csv"
    ln -sf "$M/processed/graph/edges_$f.csv" "$P/graph/edges_$f.csv"
  done
  "$PY" - "$M/processed" "$B/day03" "$P" <<'EOF'
import json, sys
import pandas as pd
M, D3, P = sys.argv[1:]
a = pd.read_csv(f"{M}/flows/train.csv", dtype=str)
b = pd.read_csv(f"{D3}/flows/train.csv", dtype=str)
# метки дня 03 сдвигаются на −25 ч (в 10:00–11:00 дня 02): окно train сплошное, 2 часа, и temporal_encoder
# режет его ровно на «день 03» и «день 02» — один хост в двух днях становится положительной парой
b["ts"] = (b.ts.astype(int) - 86400 - 3600).astype(str)
f = pd.concat([b, a], ignore_index=True)
f.to_csv(f"{P}/flows/train.csv", index=False)
m = json.load(open(f"{M}/manifest.json"))
m["windows"]["train"].update(start=122400.0, min_time=float(f.ts.astype(int).min()), n_flows=int(len(f)))
m["config"]["windows"]["train"] = [122400.0, 129600.0]
m["two_days"] = "train: 10:00–11:00 — день 03 11:00–12:00 (метки −25 ч), 11:00–12:00 — день 02"
json.dump(m, open(f"{P}/manifest.json", "w"), indent=2, ensure_ascii=False)
lo, hi = f[["src", "dst"]].min(1), f[["src", "dst"]].max(1)
e = pd.DataFrame({"host_a": lo, "host_b": hi, "bytes": f.bytes.astype(float), "packets": f.packets.astype(float)})
e = e[e.host_a != e.host_b].groupby(["host_a", "host_b"]).agg(flows=("bytes", "size"), bytes=("bytes", "sum"),
                                                              packets=("packets", "sum")).reset_index()
e.to_csv(f"{P}/graph/edges_train.csv", index=False)
print(f"train: {len(a)} потоков дня 02 + {len(b)} дня 03")
EOF
fi

T=$(mktemp -d)/two_days
git worktree add -q --detach "$T" HEAD
trap 'git worktree remove --force "$T"' EXIT
cd "$T"
export PT_DATA=$P PT_COMPACT=1
cp "$P/cohort.csv" "$P/manifest.json" scoreboard/data/processed/
find scoreboard/submissions -name '*.csv' -delete
find scoreboard/submissions -name '.build' -delete
rm -f scoreboard/results/*.json
mkdir -p scoreboard/submissions/temporal_cca
sed 's/temporal_cca_tuned/temporal_cca/g' scoreboard/submissions/temporal_cca_tuned/README.md > scoreboard/submissions/temporal_cca/README.md
"$PY" -m evaluation.submit --builtin behavioral --force
"$PY" scoreboard/temporal_encoder.py --encoder cca
"$PY" scoreboard/hybrid_tfidf.py
if "$PY" -c "import torch" 2>/dev/null; then "$PY" scoreboard/temporal_encoder.py; fi
"$PY" -m evaluation.submit --all
cp scoreboard/README.md "$B/README.md"
echo "готово: $B/README.md (смотреть столбец дня 04 в разделе «По дням»)"
