# Как мы работаем

## Шпаргалка: от задачи до слияния

```bash
# 1. свежий main и своя ветка: <логин>/<тип>-<номер issue>-<коротко>
git switch main && git pull
git switch -c physcorgi/feature-68-my-method

# 2. работа, тесты
python -m pytest -q evaluation methods app pipeline experiments

# 3. коммит и push
git add -A
git commit -m "Методы: node2vec с подбором p, q"
git push -u origin physcorgi/feature-68-my-method

# 4. PR — откроется шаблон, заполните его
gh pr create --base main
```

Дальше: ревью коллеги и @physcorgi, зелёный CI, **Squash merge**, ветку удалить.

## Шаблоны

| что | как называть | пример |
|---|---|---|
| ветка | `<логин>/<тип>-<номер issue>-<коротко>` | `physcorgi/feature-68-eval` |
| тип ветки | `research`, `data`, `feature`, `fix`, `docs`, `infra` | |
| issue | «Область: что сделать» — есть шаблоны для метода, метрики, ошибки | «Методы: GraphSAGE на признаках» |
| PR | «Область: что сделано» + `Closes #<номер>` | «Оценка: метрика кластеров» |
| коммит | что и зачем, по-русски | «Пайплайн: окна по 30 минут» |

Ветку и ссылку на задачу проверяет CI (workflow **PR format**) и пишет, что поправить.

## Свой метод в таблицу

```bash
cp scoreboard/template_method.py scoreboard/my_method.py   # поменять NAME, AUTHOR, fit и embed
python scoreboard/my_method.py                             # обучит, сохранит, положит заготовку карточки
# заполнить scoreboard/submissions/my_method/README.md и запустить ещё раз — покажет место
```

Дальше commit, push — CI добавит метод в [таблицу](scoreboard/README.md). Все учатся на одних данных
(только train-окно) и сравниваются с одними бейзлайнами — шаблон это обеспечивает.

Уже есть эмбеддинг из своего кода? Папка `scoreboard/submissions/<метод>/`:

- `embeddings.csv` — `host_id, emb_0, emb_1, …` по train-окну (обязательно);
- `test.csv` — те же хосты по test-окну, добавит колонку Identity;
- `SHARED_COORDINATES` — пустой файл, если у окон одна система координат;
- `AUTHOR` — ваш логин на GitHub, для колонки submitted by;
- `README.md` — **карточка метода (обязательно)** по [`scoreboard/CARD_TEMPLATE.md`](scoreboard/CARD_TEMPLATE.md):
  кратко, архитектура, какие данные и как используются, обучение и подбор, ограничения. Без неё или с TODO CI
  отправку не примет. На карточку ведёт ссылка из таблицы.

Подбирать параметры по таблице нельзя — для этого `experiments.tune_methods` (validation).

## Код

- Комментарии — по-русски и коротко. Имена в коде — по-английски.
- Тест рядом с модулем: `test_<модуль>.py`.
- Новый метод — `methods/graph/`, новая метрика — `evaluation/`.

## Нельзя

- Писать напрямую в `main`.
- Коммитить пароли, токены, ключи, `.env`, данные (`data/`, `runs/`).
