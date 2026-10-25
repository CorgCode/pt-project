#!/bin/bash
# Скачать netflow_day-DD.bz2 из LANL параллельными кусками (сервер даёт ~0.3 МБ/с на соединение).
#   LANL_TOKEN=<токен из формы> ./fetch.sh 02 /путь/к/raw [12]
# Токен выдаёт форма на https://csr.lanl.gov/data/2017/ (email + цель использования). В git его не кладём.
set -e
DAY=$1; OUT=${2:-.}; N=${3:-12}
[ -z "$LANL_TOKEN" ] && { echo "нужен LANL_TOKEN (см. комментарий в начале файла)"; exit 1; }
mkdir -p "$OUT"; cd "$OUT"
U="https://csr.lanl.gov/data-fence/$LANL_TOKEN/unified-host-network-dataset-2017/netflow/netflow_day-$DAY.bz2"
F=netflow_day-$DAY.bz2
TOTAL=$(curl -sI "$U" | tr -d '\r' | awk 'tolower($1)=="content-length:"{print $2}')
[ -z "$TOTAL" ] && { echo "сервер не отдал размер файла — проверьте токен"; exit 1; }
size() { [ -f "$1" ] && (stat -f %z "$1" 2>/dev/null || stat -c %s "$1") || echo 0; }
HAVE=$(size $F)
[ "$HAVE" -ge "$TOTAL" ] && { echo "$F уже скачан"; exit 0; }
STEP=$(( (TOTAL-HAVE+N-1)/N ))
# куски докачиваются, пока все не дойдут до конца: сервер иногда обрывает соединение
for attempt in $(seq 1 30); do
  left=0
  for i in $(seq 0 $((N-1))); do
    S=$((HAVE+i*STEP)); E=$((S+STEP-1)); [ $E -ge $TOTAL ] && E=$((TOTAL-1))
    P=$F.part$i; GOT=$(size $P)
    [ $((S+GOT)) -le $E ] && { left=$((left+1)); curl -s --retry 10 --retry-delay 3 -r $((S+GOT))-$E "$U" >> $P & }
  done
  wait
  [ $left = 0 ] && break
  [ $attempt -gt 1 ] && echo "повтор $attempt: оборвались $left кусков, докачиваю"
done
touch $F
for i in $(seq 0 $((N-1))); do cat $F.part$i >> $F; rm $F.part$i; done
[ "$(size $F)" -eq "$TOTAL" ] && bzip2 -t $F && echo "$F: $TOTAL байт, bz2 в порядке" \
  || { echo "$F: размер $(size $F) из $TOTAL или архив битый — удалите файл и запустите снова"; exit 1; }
