#!/bin/bash
# Ночной прогон на нескольких днях LANL (#60): скачать → подготовить → обучить → перенести итоги.
#   LANL_TOKEN=<токен> experiments/lanl/night.sh /путь/к/папке 03 04 05
#   DRY=1 experiments/lanl/night.sh /tmp/x 03        — только показать команды
# Окна — те же часы суток, что и в дне 02 (config_day02.json), сдвиг на сутки: выбраны до просмотра результатов.
# День, который упал, не останавливает остальные. Прогресс: LANL_WORKDIR=<папка>/dayNN experiments/lanl/status.sh --watch
set -u
W=${1:?папка для данных}; shift
DAYS=${*:-03 04 05}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python}
METHODS=${METHODS:-deepwalk node2vec graphsage rolx}   # struc2vec дольше всех и хуже всех — по желанию: METHODS="... struc2vec"
KEEP_RAW=${KEEP_RAW:-0}                                 # 1 — не удалять архив дня после подготовки
BASE=${BASE:-$REPO/experiments/lanl/config_day02.json}   # окна дня 02, от них сдвиг
KIND=${KIND:-lanl}                                      # synthetic_fixture — только для проверки скрипта

run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }
walks5() {                                              # num_walks = 5, как в дне 02, — только для выбранных методов
  local out="" m; for m in deepwalk node2vec; do case " $METHODS " in *" $m "*) out+="${out:+, }\"$m\": {\"num_walks\": 5}";; esac; done
  echo "{$out}"
}

day() {
  local D=$1 B=$W/day$1
  mkdir -p "$B/raw" "$B/logs"
  export LANL_WORKDIR=$B
  local OFF=$(( (10#$D - 2) * 86400 ))
  "$PY" - "$BASE" "$OFF" > "$B/config.json" <<'EOF'
import json, sys
c = json.load(open(sys.argv[1])); off = int(sys.argv[2])
c["windows"] = {k: [a + off, b + off] for k, (a, b) in c["windows"].items()}
print(json.dumps(c, indent=2))
EOF
  echo "=== день $D: окна $(tr -d ' \n' < "$B/config.json" | grep -o '"windows":{[^}]*}')"
  [ -f "$B/lanl/processed/manifest.json" ] || [ -f "$B/raw/netflow_day-$D.bz2" ] || run "$REPO/experiments/lanl/fetch.sh" "$D" "$B/raw" || return 1
  cd "$REPO" || return 1
  [ -f "$B/lanl/processed/manifest.json" ] || run "$PY" -m pipeline.prepare run --netflow "$B/raw/netflow_day-$D.bz2" \
      --config "$B/config.json" --output-dir "$B/lanl/processed" --data-kind "$KIND" \
      --source-note "форма LANL, netflow_day-$D, ночной прогон $(date +%F)" > "$B/logs/prepare.log" 2>&1 || return 1
  [ "$KEEP_RAW" = 1 ] || run rm -f "$B/raw/netflow_day-$D.bz2"
  [ -f "$B/lanl/reference/test/manifest.json" ] || run "$PY" -m pipeline.reference --processed-dir "$B/lanl/processed" \
      --output-dir "$B/lanl/reference" --pairs-per-anchor 10 > "$B/logs/reference.log" 2>&1 || return 1
  # как в дне 02: все методы, затем граф host–service (v2) в отдельную папку
  run "$PY" -m experiments.run_experiment --processed-dir "$B/lanl/processed" --reference-dir "$B/lanl/reference" \
      --out-dir "$B/lanl/run" --methods $METHODS --port-free-baselines --reuse-embeddings \
      --method-config "$(walks5)" > "$B/logs/experiment.log" 2>&1 || return 1
  run "$PY" -m experiments.run_experiment --processed-dir "$B/lanl/processed" --reference-dir "$B/lanl/reference" \
      --out-dir "$B/lanl/run_v2" --methods deepwalk node2vec --dim 32 --reuse-embeddings \
      --method-config "$REPO/experiments/lanl/methods_v2.json" > "$B/logs/experiment_v2.log" 2>&1 || return 1
  run "$PY" experiments/lanl/publish.py "$B/lanl" "results/lanl_day$D" || return 1
  mkdir -p "$B/v2" && ln -sfn "$B/lanl/processed" "$B/v2/processed" && ln -sfn "$B/lanl/reference" "$B/v2/reference" \
      && ln -sfn "$B/lanl/run_v2" "$B/v2/run"                # publish.py берёт <папка>/run
  run "$PY" experiments/lanl/publish.py "$B/v2" "results/lanl_day${D}_services" || return 1
}

export PYTHONUNBUFFERED=1
for D in $DAYS; do
  start=$(date +%s)
  if day "$D"; then echo "=== день $D готов за $(( ($(date +%s) - start) / 60 )) мин"
  else echo "=== день $D упал — смотрите $W/day$D/logs/"; fi
done
echo "Итоги — в results/lanl_day*. Утром: закоммитить их в ветку и открыть PR."
