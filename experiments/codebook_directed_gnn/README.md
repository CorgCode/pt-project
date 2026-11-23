# Codebook Directed GNN

Направленная GNN с GRU и двумя нулевыми residual-проекциями в896 координат
оригинального multiday_codebook. Source=base+Ws×history_graph;
late=base_late+Wl×current_graph. Час разделён на6 частей по10 минут.
Оригинальная основа и её Ridge обязательны: это не самостоятельный
checkpoint с исходными векторами лидера plus.

[Математика основы, GNN и функций потерь](../../research/DrMogger/reports/methods_since_2026_11_17.md).
[Актуальный пересчёт и контроли](../../research/DrMogger/nov17/README.md).

## Оригинальная основа

PT_DATA/fingerprint_codebook_recent/final/model.pkl должен иметь SHA256
a1c4bfa15ed0c227aefd981f59b87769aece146efb232da90617e4f561b10211.
Загрузка проверяет SHA и очищает оконные кэши. pickle берётся только из этого
доверенного локального артефакта; веса в Git не включены.

Архивный код нужен для восстановления старых имён классов:

~~~sh
python -m pip install -r requirements.txt torch joblib psutil pyarrow cloudpickle
mkdir -p runs/original_multiday_code
git archive 86d709d experiments/multiday experiments/overnight scoreboard | tar -x -C runs/original_multiday_code
~~~

Эта операция извлекает код из истории, а не обучает основу. Передайте каталог
через --archive. Полный старый Git commit должен присутствовать локально.
Две Bag-головы и кодировщик оригинала подробно разобраны в общем отчёте.

## Повторение

PT_RUN — новый каталог, PT_DEV_REFERENCE — локальный reference дня13.
Для zero_check нужны сохранённые source и late CSV оригинального896-мерного
codebook на дне13; late разрешено содержать только активных хостов.

~~~sh
python -m experiments.codebook_directed_gnn.prepare --data "$PT_DATA" --output "$PT_RUN/windows" --part train
python -m experiments.codebook_directed_gnn.prepare --data "$PT_DATA" --output "$PT_RUN/windows" --part development
python -m experiments.codebook_directed_gnn.zero_check --data "$PT_DATA" --archive runs/original_multiday_code --source "$BASE_SOURCE" --late "$BASE_LATE13" --reference "$PT_DEV_REFERENCE" --output "$PT_RUN/zero"
python -m experiments.codebook_directed_gnn.train_identity --data "$PT_DATA" --archive runs/original_multiday_code --windows "$PT_RUN/windows" --output "$PT_RUN/train" --steps 12 --batch 256 --seed 17
python -m experiments.codebook_directed_gnn.evaluate_identity --data "$PT_DATA" --windows "$PT_RUN/windows" --training "$PT_RUN/train" --reference "$PT_DEV_REFERENCE" --zero "$PT_RUN/zero" --output "$PT_RUN/dev"
python -m experiments.codebook_directed_gnn.train_donors --data "$PT_DATA" --windows "$PT_RUN/windows" --training "$PT_RUN/train" --zero "$PT_RUN/zero" --output "$PT_RUN/donors" --steps 6 --batch 4
python -m experiments.codebook_directed_gnn.evaluate_donors --data "$PT_DATA" --training "$PT_RUN/donors" --late "$PT_RUN/dev/late.csv" --reference "$PT_DEV_REFERENCE" --output "$PT_RUN/dev_donors"
~~~

Контроли первого этапа: --no-graph и --directionless-bytes. Во втором
направления байтов усредняются при сохранении общего объёма. Для нового late
можно вызвать evaluate_identity.apply(checkpoint, prepared_graph, base_late):
сеть не вызывает fit и использует только собственный граф этого окна.

Первые12 шагов обучают только late-поправку. Следующие6 шагов фиксируют
графовый encoder и учат source-проекцию по полезным историческим заменам.
Дни14–16 не оценивались, улучшения полного development-score не получено.
Повторный inference из сохранённых checkpoint дал точное совпадение late CSV
для полной, no_graph и directionless веток.
