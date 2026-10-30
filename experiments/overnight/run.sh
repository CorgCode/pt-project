#!/bin/bash
# Ночной перебор на одной машине (16 ГБ памяти хватает): данные → перебор → итог.
#   experiments/overnight/run.sh /путь/к/папке [часов]
#   PT_DATA=/готовая/processed experiments/overnight/run.sh /путь/к/папке   — если данные уже подготовлены
#   LANL_TOKEN=<токен> NOTE="форма LANL, <дата>, <кто>" ...                  — если данных нет: скачать дни 02–04
#   SMOKE=1 ...                                                               — проверка за ~10 минут
# Безопасность: низкий приоритет (nice), не больше половины ядер (максимум 4), машина не засыпает (caffeinate),
# между пробами — проверка памяти, диска и перегрева (search.py). Повторный запуск продолжает с места остановки.
set -eu
W=${1:?папка для данных и логов}
HOURS=${2:-8}
REPO=$(cd "$(dirname "$0")/../.." && pwd)
PY=${PY:-python3}
mkdir -p "$W/logs"
cd "$REPO"

CORES=$(sysctl -n hw.ncpu 2>/dev/null || nproc)
THREADS=$(( CORES / 2 )); [ "$THREADS" -gt 4 ] && THREADS=4; [ "$THREADS" -lt 1 ] && THREADS=1
export OMP_NUM_THREADS=$THREADS OPENBLAS_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS VECLIB_MAXIMUM_THREADS=$THREADS
export PT_COMPACT=1 PYTHONUNBUFFERED=1

FREE=$(df -Pk "$W" | awk 'NR==2 {print int($4/1048576)}')
echo "ядер $CORES, потоков $THREADS, свободно на диске $FREE ГБ"

# 1. данные: готовые (PT_DATA) или скачать и подготовить дни 02–04 с окнами таблицы
if [ -z "${PT_DATA:-}" ]; then
  PT_DATA=$W/processed
  if [ ! -f "$PT_DATA/manifest.json" ]; then
    [ "$FREE" -ge 15 ] || { echo "нужно ≥ 15 ГБ свободного места для скачивания"; exit 1; }
    : "${LANL_TOKEN:?нет данных: задайте PT_DATA или LANL_TOKEN (форма https://csr.lanl.gov/data/2017/)}"
    : "${NOTE:?NOTE — откуда данные: «форма LANL, <дата>, <кто>»}"
    for D in 02 03 04; do
      [ -f "$W/raw/netflow_day-$D.bz2" ] || experiments/lanl/fetch.sh "$D" "$W/raw"
    done
    nice -n 10 "$PY" -m pipeline.prepare run --netflow "$W"/raw/netflow_day-0{2,3,4}.bz2 \
        --config experiments/lanl/config_multiday.json --output-dir "$PT_DATA" --data-kind lanl \
        --source-note "$NOTE" > "$W/logs/prepare.log" 2>&1
  fi
fi
# когорта должна совпадать с таблицей, иначе баллы несравнимы
cmp -s <(cut -d, -f1 "$PT_DATA/cohort.csv" | sort) <(cut -d, -f1 scoreboard/data/processed/cohort.csv | sort) \
  || { echo "когорта в $PT_DATA не совпадает с scoreboard/data/processed — данные другие"; exit 1; }
export PT_DATA

# 2. перебор и итог (caffeinate — только на macOS: не даёт уснуть, пока идёт процесс)
ARGS=(--hours "$HOURS"); [ -n "${SMOKE:-}" ] && ARGS=(--smoke --hours 1)
KEEP=(); command -v caffeinate >/dev/null && KEEP=(caffeinate -i)
${KEEP[@]+"${KEEP[@]}"} nice -n 15 "$PY" -m experiments.overnight.search "${ARGS[@]}" 2>&1 | tee -a "$W/logs/overnight.log"
echo "готово: runs/overnight/REPORT.md, лог: $W/logs/overnight.log"
