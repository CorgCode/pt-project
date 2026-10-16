# Дашборд (Streamlit)

Показывает результат прогона (`python -m experiments.end_to_end`) и позволяет оценить свой файл эмбеддингов.
Главные числа не пересчитываются, а читаются из сохранённых файлов, поэтому дашборд не может разойтись с оценкой.

```bash
pip install -r app/requirements.txt
streamlit run app/streamlit_app.py                            # синтетический пример из demo/synthetic_run
PT_RESULTS_DIR=runs/lanl streamlit run app/streamlit_app.py   # или выбрать прогон в боковой панели
```

Прогоны ищутся в `demo/synthetic_run` и `runs/<имя>`. Пример в репозитории — **синтетический**, и дашборд пишет
это вверху каждой страницы.

## Вкладки

| вкладка | что показывает |
|---|---|
| Overview | когорта, методы, неоценённые методы, размеры окон, отбракованные строки |
| Leaderboard | метрики по группам, validation или test, 95% интервалы, недоступные метрики |
| Link prediction | косинус и обученный классификатор связей, бейзлайн «связь уже была», случайный уровень |
| Neighbourhoods | 2D-проекция хостов, цвет — совпадение с эталоном; соседи выбранного хоста |
| Temporal | validation против test, узнаваемость хоста во времени, дрейф соседства |
| Validity audit | случайный уровень, чувствительность к весам признаков, пересечение с эталонами |
| Provenance | манифесты, хеши файлов, правила разметки |
| Try your embedding | загрузить свой `embeddings.csv` для демо-когорты и сравнить с бейзлайнами |

## Правила

- синтетика или реальные данные — видно всегда; непосчитанная метрика показывается как недоступная, а не 0;
- потолок `reference_ceiling` помечен как «утекающий» и не участвует в выборе лучшего;
- на графиках есть случайный уровень (пунктир);
- пропуски — «-», а не `None`/`nan`.

## Проверка в браузере

`python -m app.browser_check --out app/screenshots --stress` (нужны `pip install playwright && playwright install chromium`)
поднимает сервер и проходит все вкладки в headless Chromium на ширине 1440 и 390 px: ошибки консоли, исключения
Streamlit, горизонтальная прокрутка, `nan` в тексте, совпадение чисел на странице с файлами результатов.
В CI не входит (нужен браузер); в CI идут `AppTest` из `test_app.py`.

| компьютер | телефон |
|---|---|
| ![leaderboard](screenshots/desktop_leaderboard.png) | ![leaderboard](screenshots/phone_leaderboard.png) |
| ![link prediction](screenshots/desktop_link_prediction.png) | ![overview](screenshots/phone_overview.png) |
| ![neighbourhoods](screenshots/desktop_neighbourhoods.png) | |

Код: `results.py` (данные, без Streamlit), `views.py` (вкладки), `streamlit_app.py` (раскладка), `core.py` (загрузка файла).
Тесты: `python -m pytest -q app`.
