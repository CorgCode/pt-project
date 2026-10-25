#!/bin/bash
# Таблица с проверкой на нескольких днях: train — день 02, test — по часу в днях 02, 03, 04 (день 05 не трогаем).
#   LANL_TOKEN=<токен> NOTE="форма LANL, <дата>, <кто>" experiments/lanl/multiday.sh /путь/к/папке
#   CONFIG=experiments/lanl/config_multiday_train6h.json ...   — train 6 часов (нужна машина помощнее, BIG_MACHINE.md)
#   DRY=1 ... — только показать команды
# Архивы дней, которых нет в <папка>/raw, скачиваются (нужен LANL_TOKEN). Итог — scoreboard/ этой ветки.
set -eu
W=${1:?папка для данных}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
CONFIG=${CONFIG:-$REPO/experiments/lanl/config_multiday.json}
PY=${PY:-python}
NOTE=${NOTE:?NOTE — откуда данные: «форма LANL, <дата>, <кто>»}
run() { echo "+ $*"; [ -n "${DRY:-}" ] || "$@"; }
cd "$REPO"

for D in 02 03 04; do
  [ -f "$W/raw/netflow_day-$D.bz2" ] || run experiments/lanl/fetch.sh "$D" "$W/raw"
done

run "$PY" -m pipeline.prepare run --netflow "$W"/raw/netflow_day-0{2,3,4}.bz2 --config "$CONFIG" \
    --output-dir "$W/processed" --data-kind lanl --source-note "$NOTE" --overwrite
run "$PY" -m pipeline.reference --processed-dir "$W/processed" --output-dir "$W/reference" --pairs-per-anchor 10

# новые окна — в таблицу: старые векторы и результаты построены по другим окнам
run cp "$W/processed/cohort.csv" "$W/processed/manifest.json" scoreboard/data/processed/
run rm -rf scoreboard/data/reference
run cp -r "$W/reference" scoreboard/data/reference
run find scoreboard/submissions -name '*.csv' -delete
run find scoreboard/submissions -name '.build' -delete
run rm -f scoreboard/results/*.json
export PT_DATA=$W/processed

run "$PY" -m evaluation.submit --builtin all
run "$PY" scoreboard/hybrid_tfidf.py --tune          # DIM и веса — на validation, с узнаваемостью
run "$PY" scoreboard/temporal_encoder.py --encoder cca --tune   # лёгкая версия без torch: секунды
if "$PY" -c "import torch" 2>/dev/null; then
  run "$PY" scoreboard/temporal_encoder.py           # рецепт выбран на validation дня 02, здесь не перенастраивается
  run "$PY" scoreboard/temporal_encoder.py --tune    # те же части, веса и куски train — на validation этого прогона
else
  echo "temporal_encoder пропущен: нет torch (pip install torch)"
fi
run "$PY" scoreboard/combine.py behavioral node2vec_services graph_spectral --name hybrid_services --tune-weights
run "$PY" scoreboard/combine.py behavioral node2vec graph_spectral --name hybrid_weighted --tune-weights
run "$PY" scoreboard/combine.py behavioral deepwalk --name hybrid_behavioral_deepwalk
run "$PY" scoreboard/combine.py behavioral node2vec --name hybrid_behavioral_node2vec
run "$PY" scoreboard/combine.py behavioral node2vec graph_spectral --name hybrid_behavioral_node2vec_spectral
run "$PY" -m evaluation.submit --all
echo "готово: таблица — scoreboard/README.md, по дням — раздел «По дням»"
