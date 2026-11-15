# Восстановление оригинальной основы multiday_codebook_plus

> Публикуется документ локального исследования. Указанные команды требуют локального кода `experiments/leader_recovery`, который не входит в данный PR и отсутствует в его базовом main. Артефакты доступны только владельцу локальной рабочей копии; опубликованный отчёт не гарантирует их доступность другим участникам.

**Оригинальный multiday_codebook найден и загружен. Дополнительная голова
plus не найдена.** Реконструкция не выполнялась; старый пилот не используется.

## Подтверждение происхождения

Параметры `70347a3:scoreboard/submissions/multiday_codebook_plus/params.json`
фиксируют base_checkpoint_sha256:
`a1c4bfa15ed0c227aefd981f59b87769aece146efb232da90617e4f561b10211`.

Тот же SHA256 имеют реальные локальные файлы:
- PT_PROJECT_ML/fingerprint_codebook_recent/final/model.pkl
- PT_PROJECT_ML/fingerprint_codebook_recent/candidates/codebook1_e400/best.pkl

Checkpoint успешно загружен: CodebookEncoder, source6453×896, base, adapter.
Его source совпадает с late_test/vectors/multiday_codebook/embeddings.csv
с максимальной абсолютной ошибкой4,9998e-9 (округление CSV).

Исходный pickle использует прежние имена experiments.fingerprint_*. Код
восстановлен из локального git commit86d709d в runs/original_multiday_code.
load_checkpoint.py перенаправляет старые имена в experiments.multiday.*,
не изменяя оригинальный pickle. Установлены недостающие psutil и pyarrow.

## Повторная оценка основы

Пять метрик повторно рассчитаны на сохранённых source/late CSV и эталонах
дней14–16, без обучения и без подбора параметров по этим дням.

| Окно | Повторный score |
|---|---:|
| День14 | 63,25525 |
| День15 | 63,21744 |
| День16 | 68,24970 |
| Среднее | **64,90746** |

Опубликованное среднее основы:64,93617. Полного численного совпадения нет:
Links AUC отличается примерно на0,00016–0,00029, Identity дня14 — на один
оцениваемый запрос (0,64304 против0,64430). Остальные четыре/три метрики
совпадают. Причина малых расхождений не установлена; возможны численные
различия библиотек/округления/SVD, это предположение, не доказанный вывод.
SHA всех семи файлов первого test-эталона и всех семи fit/probe-файлов
совпали с reference_info опубликованного JSON. Оценка Links использует
перенесённый memory-bounded evaluator той же C=1 задачи, не буквальный
старый бинарный runtime. Это восстановленная оригинальная основа, но не
побитовое воспроизведение всего исторического окружения.

Артефакты: runs/original_codebook_recovery/summary.json, test*.json,
checkpoint_source.json и reference_audit.json.

## Чего не хватает для plus70,04

Ожидаемый SHA256 partner-head:
`e67a64739fa7b828b813f7824c223bd3ae49821a5293c7c5bd90e90170aca6be`.

Проверено133 реальных файла моделей/массивов с расширениями pkl, pt, npy,
npz, joblib (без служебных flow/count-cache); совпадения нет. Инвентаризация:
runs/leader_checkpoint_inventory.json. Ссылки-дубликаты не хешировались.
GitHub Releases проверены через gh: опубликован только
 temporal-stable-hybrid-day02-v1. Код головы/сборки отсутствует в снимке70347a3
и в локальной ветке origin/physcorgi/multiday-codebook-plus.

Следовательно, 70,04 этому восстановлению не приписывается. Для точного plus
нужны сохранённая partner-head с указанным SHA, код её загрузки/сборки и
graph-блок либо его воспроизводимый генератор. Переобученная голова будет
новой конфигурацией и потребует собственной проверки.

## Команды

```bash
mkdir -p runs/original_multiday_code
git archive 86d709d | tar -x -C runs/original_multiday_code
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m experiments.leader_recovery.load_checkpoint
OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 .venv/bin/python -m experiments.leader_recovery.reproduce
```

Большие файлы не копировались в Git. Внешних сообщений авторам не отправлялось.
