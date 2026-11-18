# Сохранённые модели

Веса хранятся отдельно от Git. Локальная папка артефактов может находиться на внешнем диске;
её абсолютный путь не является частью кода и задаётся при проверке.

## multiday_codebook

Найден и проверен исходный checkpoint основы:

- Относительный путь: `fingerprint_codebook_recent/final/model.pkl`.
- SHA-256: `a1c4bfa15ed0c227aefd981f59b87769aece146efb232da90617e4f561b10211`.
- Эталонный SHA записан в `scoreboard/submissions/multiday_codebook/params.json`.
- Рядом сохранены `frozen_selection.json`, `summary.json`, `REPORT.md`.

Проверка не загружает pickle и не запускает модель:

```bash
python tools/check_artifacts.py /path/to/PT_PROJECT_ML
```

Совпадение SHA подтверждает идентичность checkpoint. Оно не заменяет проверку
inference и не восстанавливает отсутствующие Python-модули. Исторические scores
в соседних отчётах могут относиться к другому протоколу оценки.

## multiday_codebook_plus

В проверенной папке не найдены отдельные веса и реализация дополнительной головы Plus.
Найденный checkpoint основы не является полной Plus-моделью.
Состояние опубликованного пакета — [REPRODUCIBILITY.md](REPRODUCIBILITY.md).
