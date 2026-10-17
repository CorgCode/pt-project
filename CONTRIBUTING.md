# Как мы работаем

## Шпаргалка: от задачи до слияния

```bash
# 1. свежий main и своя ветка: <логин>/<тип>-<номер issue>-<коротко>
git switch main && git pull
git switch -c physcorgi/feature-68-my-method

# 2. работа, тесты
python -m pytest -q evaluation benchmark app pipeline experiments

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

Положить `scoreboard/submissions/<метод>/embeddings.csv`, commit, push — CI добавит метод в
[таблицу](scoreboard/README.md). PR открывать по шаблону выше.

## Код

- Комментарии — по-русски и коротко. Имена в коде — по-английски.
- Тест рядом с модулем: `test_<модуль>.py`.
- Новый метод — `benchmark/methods/`, новая метрика — `evaluation/`.

## Нельзя

- Писать напрямую в `main`.
- Коммитить пароли, токены, ключи, `.env`, данные (`data/`, `runs/`).
