# PT-05: эмбеддинги хостов по сетевому трафику

Исследовательский проект Positive Technologies × МФТИ. По NetFlow строим векторы
компьютеров, сравниваем поведение, прогнозируем контакты и проверяем узнаваемость
хоста в более позднем окне. Данные — [LANL 2017](https://csr.lanl.gov/data/2017/).
Настоящей разметки ролей и инцидентов у используемого набора нет.

## Что есть в main

Доступны подготовка данных, бейзлайны, графовые методы, единая оценка и Streamlit-дашборд.
Это исследовательский стенд: он не выдаёт подтверждённые роли устройств и вердикты об атаках.
Состав и ограничения — [PROJECT_STATUS.md](PROJECT_STATUS.md).

## Сайт и лаборатория

Запустите `python3 noomo-ui/serve.py` и откройте `http://127.0.0.1:8845/`.
Интерактивная лаборатория, разбор модели и инструкции по запуску описаны в
[noomo-ui/README.md](noomo-ui/README.md). Отдельный локальный режим для оператора
описан в [app/operator/README.md](app/operator/README.md).
Сборка Docker, HTTPS и подключение инференса описаны в [deploy/README.md](deploy/README.md).

## Запуск с нуля

Python 3.11 или 3.12. Команды выполняются из корня репозитория:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m experiments.end_to_end --fixture --workdir runs/demo
python -m streamlit run app/streamlit_app.py
```

`--fixture` создаёт синтетические данные и проверяет конвейер; полученные числа
не являются результатами на LANL. Дашборд по умолчанию читает опубликованный LANL-прогон.
Чтобы посмотреть свой: `PT_RESULTS_DIR=runs/demo python -m streamlit run app/streamlit_app.py`.
Для нейросетевых экспериментов PyTorch устанавливается отдельно:
`python -m pip install -r requirements-ml.txt`.

## Короткие команды

Если установлен `make`: `make setup`, `make check`, `make test`, `make demo`,
затем `make dashboard`. Последняя команда открывает результаты demo.
Все команды можно посмотреть через `make help`; Python по умолчанию — 3.12.
Для 3.11: `make setup PYTHON=python3.11`.

## Проверки

```bash
python tools/check_repo.py
python -m pytest -q
```

Первая команда проверяет структуру, ссылки основных документов и контрольные суммы
сохранённых эталонов. Вторая собирает тесты всего репозитория.
Сведения о версиях и воспроизводимости — [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md).

## Актуальные результаты

[Активная таблица](scoreboard/README.md) использует протокол `lanl-source02-test88-90-v1`: test — дни 88–90, source — неизменённый час дня 02. Лучший из переоценённых методов — `multiday_codebook`, средний score **66.4**.

Результат `multiday_codebook_plus` 70,0 относится к [историческому бенчмарку дней 14–16](scoreboard/history/days14-16/README.md). Он не является оценкой на дне 90. Состав переоценённых методов, покрытие и ограничения приведены в активной таблице.

## Навигация

| Задача | Где |
|---|---|
| Повторить обработку LANL | [experiments/lanl/](experiments/lanl/README.md) |
| Подготовить окна и эталоны | [pipeline/](pipeline/README.md) |
| Посмотреть методы | [methods/](methods/README.md) |
| Понять метрики и score | [evaluation/](evaluation/README.md) |
| Добавить метод | [CONTRIBUTING.md](CONTRIBUTING.md) |
| Посмотреть отчёты | [results/](results/README.md) |
| Запустить дашборд | [app/](app/README.md) |
| Найти исследования участников | [research/](research/README.md) |

## Команда

Даниил @physcorgi · Дима @DrMogger · Тихон @Tikhon2783 · Даня @kalivixZ · Даша @ElliOrNora.
История конкретных вкладов — коммиты и [pull requests](https://github.com/CorgCode/pt-project/pulls).
