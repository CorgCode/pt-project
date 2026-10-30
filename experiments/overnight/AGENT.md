# Инструкция для агента: ночной перебор

Задача — запустить `experiments/overnight/run.sh` на этой машине на ночь, следить, чтобы она не перегрелась
и не закончилась память, утром оформить итог. Подробности о переборе — в `README.md` рядом.

## Правила

- Машина чужая и рабочая: не запускать ничего тяжёлого параллельно с перебором, не менять системные настройки,
  не ставить пакеты глобально через `sudo`.
- Данные и токены не коммитить. Токен LANL не нужен: данные уже лежат на SSD.
- В коммитах и PR — никаких строк про ИИ (`Co-Authored-By`, `Generated with` и т. п.). Сообщения коммитов — на русском.
- PR не открывать, issues и чужие ветки не трогать, в `main` не пушить.
- test не использовать для подбора: перебор сам смотрит только validation, test — один раз в итоге.
- Если что-то сломалось и непонятно почему — остановить перебор и написать, что видно, а не чинить наугад.

## 1. Подготовка

```bash
git clone https://github.com/CorgCode/pt-project.git ~/pt-project   # если репозитория ещё нет
cd ~/pt-project && git fetch && git checkout physcorgi/overnight-search-139 && git pull
```

Найти данные на SSD (папка `PT_PROJECT_ML/multiday/processed`, внутри `cohort.csv`, `flows/`, `graph/`, `manifest.json`):

```bash
ls -d /Volumes/*/PT_PROJECT_ML/multiday/processed      # macOS
ls -d /media/*/*/PT_PROJECT_ML/multiday/processed      # Linux
```

Брать именно `multiday/processed`. Папка `multiday2d` — обучение на 2 часах, по правилам таблицы не подходит.

Python 3.10+ с зависимостями — в отдельном окружении:

```bash
python3 -m venv ~/pt-venv && ~/pt-venv/bin/pip install -r requirements.txt torch psutil
~/pt-venv/bin/python -c "import pandas, sklearn, torch, psutil; print('ok')"
```

Перед запуском: ноутбук на зарядке, свободно ≥ 10 ГБ на диске с репозиторием (`df -h .`).

## 2. Проверка (~10 минут)

```bash
export PT_DATA=<путь к multiday/processed> PY=~/pt-venv/bin/python
SMOKE=1 experiments/overnight/run.sh ~/overnight 1
```

Должно закончиться строкой `готово` и файлом `runs/overnight_smoke/REPORT.md` с баллом около 63.
Ошибка «когорта не совпадает» — выбрана не та папка данных.

## 3. Ночной запуск

В фоне, чтобы не зависеть от сессии агента:

```bash
mkdir -p ~/overnight && nohup experiments/overnight/run.sh ~/overnight 8 > ~/overnight/run.out 2>&1 &
echo $! > ~/overnight/pid
```

Следить раз в 20–30 минут, не чаще:

```bash
tail -n 5 ~/overnight/logs/overnight.log
wc -l runs/overnight/trials.jsonl                     # сделанные пробы, всего ~240
pmset -g therm; vm_stat | head -5                      # macOS: перегрев и память
```

Перебор сам ставит паузу при нехватке памяти, места или перегреве. Если машина всё же заметно греется
или тормозит у владельца — `kill $(cat ~/overnight/pid)`. Повторный запуск той же командой продолжит с места остановки.

## 4. Утром

Итог — `runs/overnight/REPORT.md` и отправка `scoreboard/submissions/overnight_best/`.

```bash
git checkout -b physcorgi/overnight-result
git add scoreboard/submissions/overnight_best scoreboard/results/overnight_best.json
git commit -m "Таблица: посылка overnight_best (ночной перебор)"
git push -u origin physcorgi/overnight-result
```

Владельцу написать коротко: балл `overnight_best` на test и место в таблице, балл на validation, сколько проб сделано,
были ли паузы из-за памяти или перегрева, ссылка на ветку. `runs/` не коммитить.
