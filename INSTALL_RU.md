# Как добавить `behavioral_dae` в `CorgCode/pt-project`

Архив содержит только новые файлы и не меняет GitHub, не создаёт ветку и не открывает PR.

## 1. Скопировать файлы в корень клона проекта

Распакуй архив в корень репозитория так, чтобы новые файлы легли по этим путям:

```text
methods/baselines/autoencoder.py
methods/baselines/test_autoencoder.py
methods/baselines/autoencoder.md
scoreboard/behavioral_dae.py
scoreboard/submissions/behavioral_dae/README.md
```

Ничего из существующего не перезаписывается. `methods/baselines/stable.py` из старого архива можно оставить как отдельный не-нейросетевой вариант; он не заменяет этот метод.

## 2. Установить зависимость

В активированном окружении проекта:

```bash
pip install torch
```

Остальные зависимости должны быть установлены как в README проекта (`pip install -r requirements.txt`).

## 3. Прогнать тесты

```bash
python -m pytest -q methods/baselines/test_autoencoder.py
```

Тесты используют проектный fixture `processed`. Они проверяют train-only обучение, формат выходных эмбеддингов, детерминированность при одинаковом seed и снижение ошибки реконструкции. Если PyTorch не установлен, тестовый модуль будет пропущен; поэтому сначала установи torch.

## 4. Прогнать на LANL

В проекте уже есть pipeline и скрипты для LANL, но сырые потоки не хранятся в Git. Доступ нужно получить через официальный портал LANL, см. `experiments/lanl/README.md` в репозитории.

После подготовки данных по инструкции проекта:

```bash
# Linux / macOS / Git Bash: путь должен указывать на папку processed с manifest.json, cohort.csv и flows/
PT_DATA=/path/to/lanl/processed python scoreboard/behavioral_dae.py
```

PowerShell:

```powershell
$env:PT_DATA = "D:\data\lanl\processed"
python scoreboard/behavioral_dae.py
```

Скрипт обучит сеть только на train, создаст `embeddings.csv`, validation и test-файлы в `scoreboard/submissions/behavioral_dae/`, затем вызовет проектную оценку. Не коммить сырые LANL-данные и сгенерированные CSV: скрипт создаёт `.gitignore` для векторов.

## 5. Перед сравнением

1. Сначала сохрани исходные изменения в своей ветке.
2. Запусти один раз с начальными настройками и зафиксируй score и метрики.
3. Если будешь подбирать параметры, делай это только на validation. Не меняй параметры по test.
4. При добавлении результатов в командную таблицу обязательно укажи свой реальный GitHub login в `AUTHOR` внутри `scoreboard/behavioral_dae.py`.

## Важно

Файлы подготовлены под интерфейс текущего `main` (по состоянию репозитория на 26 октября 2026 года). В этой среде нет доступа к исходным LANL-потокам и нельзя клонировать репозиторий для полного интеграционного прогона, поэтому настоящий LANL score не выдумывался. После распаковки проверь тесты и выполни эксперимент у себя.
