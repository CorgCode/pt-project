#!/bin/bash
# Двойной щелчок в Finder или запуск из Terminal. Обучение не изменяется.
cd "$(dirname "$0")" || exit 1
if [ ! -x .venv/bin/python ]; then
  echo 'Не найден Python проекта: .venv/bin/python'
  exit 1
fi
export TERM="${TERM:-xterm-256color}"
exec .venv/bin/python -m experiments.training_monitor "$@"
