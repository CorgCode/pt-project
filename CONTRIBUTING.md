# Как мы работаем

1. Есть issue — что сделать и когда готово.
2. Ветка от свежего `main`: `<логин>/<тип>-<номер issue>-<коротко>`, например `physcorgi/feature-68-eval`.
   Типы: `research`, `data`, `feature`, `fix`, `docs`, `infra`.
3. Коммиты небольшие, сообщение — что и зачем.
4. Pull request в `main` с `Closes #<номер>`.
5. Ревью коллеги и владельца кода (@physcorgi), CI зелёный.
6. Squash merge, ветку удалить.

Напрямую в `main` писать нельзя.

## Код

- Комментарии — по-русски и коротко. Имена в коде — по-английски.
- Тест рядом с модулем: `test_<модуль>.py`.
- Перед PR: `python -m pytest -q evaluation benchmark app pipeline experiments`.
- Новый метод — `benchmark/methods/`, новая метрика — `evaluation/`.

## Нельзя коммитить

Пароли, токены, ключи, `.env`, VPN-конфиги, данные (`data/`, `runs/`).
