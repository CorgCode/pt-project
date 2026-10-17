#!/bin/bash
# Статус прогона на LANL. Один раз: ./status.sh   Постоянно: ./status.sh --watch
# Папка прогона — LANL_WORKDIR (в ней raw/, lanl/, logs/).
D=${LANL_WORKDIR:-/Volumes/KINGSTON/PT_PROJECT_ML}
show() {
  echo "=== $(date '+%H:%M:%S')  PT-05 · LANL ==="
  echo
  echo "Шаги:"
  for s in "processed/manifest.json:1. подготовка (окна, когорта, графы)" \
           "reference/test/manifest.json:2. эталоны" \
           "run/manifest.json:3. обучение и оценка методов" \
           "run/scoreboard_test.md:4. таблица результатов"; do
    f=${s%%:*}; n=${s#*:}
    [ -e "$D/lanl/$f" ] && echo "  [готово] $n" || echo "  [ ... ] $n"
  done
  echo
  echo "Сейчас работает:"
  ps -axo pid,etime,rss,command | grep -E " -m (pipeline|experiments|evaluation)" | grep -vE "grep| -c |zsh|bash" |
    awk '{mem=$3/1024/1024; cmd=""; for(i=4;i<=NF;i++) if($i ~ /^(pipeline|experiments|evaluation)\./) {cmd=$i" "$(i+1); break}
          printf "  %-34s идёт %s, память %.1f ГБ\n", cmd, $2, mem}' | head -5
  echo
  used=$(du -sk "$D" 2>/dev/null | cut -f1)
  awk -v u=$used 'BEGIN{printf "Место на SSD: %.2f ГБ на диске\n", u/1024/1024}'
  for f in "$D"/lanl/processed/flows/[!.]*.csv; do [ -e "$f" ] && printf "  flows/%s: %d МБ\n" "$(basename "$f")" $(( $(stat -f %z "$f") / 1048576 )); done
  [ -d "$D/lanl/run/results/test" ] && echo "  методов оценено: $(ls $D/lanl/run/results/test | wc -l | tr -d ' ')"
  echo
  last=$(ls -t "$D"/logs/*.log 2>/dev/null | head -1)
  if [ -n "$last" ]; then
    echo "Лог: $(basename "$last")"
    tail -n 8 "$last" | sed 's/^/  /'
  fi
}
if [ "$1" = "--watch" ]; then while true; do clear; show; sleep 5; done; else show; fi
