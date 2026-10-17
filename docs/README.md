# Сайт и методология

- [`METHODOLOGY.md`](METHODOLOGY.md) — как оцениваем эмбеддинги и насколько можно верить числам.
- `index.html`, `assets/`, `data/` — сайт https://corgcode.github.io/pt-project/ (публикуется сам при изменении `docs/`).

Обновить данные сайта после нового прогона:

```bash
python docs/build_data.py --bundle /путь/к/lanl      # папка с run/, reference/, processed/
python docs/check_pages.py                          # проверка страницы
```
