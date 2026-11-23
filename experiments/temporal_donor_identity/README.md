# Direct Identity Codebook

Поздний вектор оригинального codebook896 корректируется линейным residual:
y = y0 + W[y0; context] + bias. В обученном варианте context — 64 нуля;
W и bias стартуют с нуля. Source не меняется. Цель — episodic Procrustes/CE
и штраф отклонения от исходного вектора. Обучение 03–12, выбор на13.
MLP и исторические донорские профили в утилитах — контроли, не выбранная модель.

Нужны оригинальная основа и архивный код, описанные в
[Codebook Directed GNN](../codebook_directed_gnn/README.md), а также сохранённые
исторические codebook-векторы в PT_PROJECT_ML. Веса и векторы локальные.

```sh
python -m experiments.temporal_donor_identity.train_identity --archive runs/original_multiday_code --output runs/direct_new --steps 160 --batch 512 --seed 17
python -m experiments.temporal_donor_identity.infer_identity --checkpoint runs/direct_new/model.pt --late base_late.csv --output runs/direct_new/late.csv
python -m experiments.temporal_donor_identity.evaluate_fixed --run runs/direct_new
```

Последняя команда — оценка уже изученных дней14–16 после фиксации выбора.
В общей локальной таблице сохранена актуальная штатная оценка прежней ветки,
повторное обучение при объединении не проводилось. Независимого holdout нет.
