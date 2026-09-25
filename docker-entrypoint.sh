#!/usr/bin/env bash
set -e

DATA_DIR="${DATA_DIR:-/app/data}"
ARTIFACTS_DIR="${ARTIFACTS_DIR:-/app/artifacts}"
N_WINDOWS="${N_WINDOWS:-5}"

if [ ! -f "${ARTIFACTS_DIR}/summary.json" ]; then
  echo "[entrypoint] Артефактов нет — генерирую данные и запускаю пайплайн…"
  if [ ! -f "${DATA_DIR}/meta.json" ]; then
    python -m hostembed.datagen --out "${DATA_DIR}" --windows "${N_WINDOWS}"
  fi
  python -m hostembed.pipeline --data "${DATA_DIR}" --artifacts "${ARTIFACTS_DIR}"
fi

echo "[entrypoint] Запускаю веб-сервис на :8000"
exec uvicorn app.server:app --host 0.0.0.0 --port 8000
