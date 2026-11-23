# Temporal GATv2

Самостоятельный encoder:12 пятиминутных графов, два edge-aware направленных
GATv2-слоя256 с4 головами, двухслойный Transformer часа, двухслойный Transformer
истории, выход512. Внешние прямые партнёры входят в граф; cap16 применяется
после точной агрегации. Байты узлов включают все контакты выбранных узлов.

Обучающие эпизоды:02+10→11,02+10→12,02+10+11→12. Loss:
Identity+0,15vote+0,2links+0,2behavior+0,3masked-bytes. Own-ID не является
входом, конкретные peers входят в хешированный контактный скетч.

[Математика](../../research/DrMogger/reports/methods_since_2026_11_17.md).
[Актуальная таблица](../../research/DrMogger/nov17/README.md).

## Запуск

PT_DATA указывает на существующий PT_PROJECT_ML, PT_RUN — новый каталог.
Нужны исходные flows и подготовленные source02/future NPZ с согласованным
словарём в fingerprint_transfer, fingerprint_aligned и fingerprint_codebook_recent.
Порядок словаря фиксируется SHA в новых checkpoint; Episode не хранит старые
индексы между повторными созданием/fit.

~~~sh
python -m experiments.temporal_gatv2.prepare_windows --data "$PT_DATA" --output "$PT_RUN/windows" --part train
python -m experiments.temporal_gatv2.prepare_windows --data "$PT_DATA" --output "$PT_RUN/windows" --part development
python -m experiments.temporal_gatv2.train --data "$PT_DATA" --windows "$PT_RUN/windows" --output "$PT_RUN/train" --steps 30 --width 256 --batch 64 --seed 17
python -m experiments.temporal_gatv2.evaluate_dev --data "$PT_DATA" --windows "$PT_RUN/windows" --training "$PT_RUN/train" --reference "$PT_DEV_REFERENCE" --output "$PT_RUN/dev"
~~~

Абляции запускаются отдельно с --disable-graph, --disable-bytes или
--disable-vote. Для сопоставимости используется13 шагов у полной сети и каждого
контроля. --warmup-steps позволяет отложить donor-loss, но такой режим не
выдаётся за проведённый эксперимент.

Пилот не оценивался на14–16. Полная модель после30 шагов имеет development
35,2263; короткий бюджет и один seed не устанавливают предел архитектуры.
Обучающий Procrustes по части batch64 имеет существенно меньший ранг, чем
512-мерное пространство. Прямой contrastive-компонент стабилизирует часть
неопределённых направлений, но не устраняет различия с официальной оценкой.
