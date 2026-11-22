# Temporal Graph Mixer

Самостоятельная модель из LANL NetFlow: четыре блока, фиксированная направленная диффузия без GNN, временной MLP-Mixer и защищённая поведенческая часть. Веса/codebook предыдущих моделей не загружаются. Результаты текущего локального запуска — [RESULTS.md](RESULTS.md). Достижение70/90 не следует из архитектуры.

## Что получает модель

Один пример — хост за объявленное часовое окно, четыре15-минутных участка в исходном порядке. Направленные объёмы считаются корректно: у src TX=bytes,RX=resp_bytes; у dst TX=resp_bytes,RX=bytes; аналогично packets. Включаются контакты вне когорты, исключаются self-flow. Own-ID служит только ключом и меткой положительной пары, не входит входным токеном. Конкретные peers являются разрешёнными токенами трафика. Векторы признаков не являются обучаемой таблицей host-ID.

Для прогнозной задачи участок не видит поток, завершающийся после его конца. События, пересекающие квартал часа, исключены из ВСЕХ quarter-признаков/графов. Для полного-hour active>=300, source voting и проверки эталона они сохраняются, если завершаются до конца часа. Это явное различие определения признаков и population; не подмена частот оценщика. Точки ровно на конце окна принадлежат следующему окну, duration используется строго ts+duration<end.

Четыре128-мерных блока:

1. **Own:**18 числовых TX/RX/packet/duration/частотных признаков по двум endpoint-ролям,128 хешированных направленных protocol-частот,128 координат распределения сервисов и8 сводок разнообразия контактов/сервисов:282 входа, train StandardScaler и PCA128. Собственные peer-ID сюда не входят: используются cardinality, entropy, maximum share и средняя частота. Это собственный поведенческий профиль этой модели, не буквальный штатный behavioral57 и не истинная разметка ролей. В первой версии было146 входов; лишь24 компоненты имели дисперсию выше1e-8 максимальной. Во второй версии этому порогу соответствуют все128 компонент. Старые transforms читаются через сохранённую схему, без изменения их признаков.
2. **Peers:**8192 хешированных направленных peer-токена, log часовых интенсивностей, IDF по различным source-хостам всей разрешённой истории, row-normalization/SVD128.
3. **Services:**1024 хеш-координаты direction/proto/port плюс3072 координаты совместных direction/peer/proto/port, тот же TF-IDF/SVD128. Возможны коллизии.
4. **Graph:**для каждого quarter четыре канала: out/in по числу flows и по реальным переданным bytes. Log1p рёбер, ослабление destination-хабов, row-нормировка P. По32 компонентам own/service считаются PX,P²X,|PX−P²X|, всего768чисел; train scaler/PCA128 **после диффузии**. Первичная проекция до64 каналов перед диффузией нужна для памяти; вариант диффузии всех несжатых sparse-словарей не реализован.

Итого X: N×4quarters×4blocks×128. Центры/масштабы блоков фиксируются на train, поздние данные ничего не fit. У неактивных хостов нормированные признаки могут быть ненулевыми из-за train-centering; для построения historical source используются настоящие маски events>=300, не проверка нормы признаков.

## Сеть и цели

Каждый quarter512→hidden256; два residual MixerLayer. Слой смешивает четыре позиции MLP4→16→4 и каналы256→512→256 с LayerNorm/GELU. Stable256 строится по среднему hidden, Dynamic128 — по last-minus-mean. Итог512: `[2×unit(own128); unit(stable256);0,5×unit(dynamic128)]`, затем общая нормировка. Own зависит только от фиксированного preprocessing и трафика, не от градиентов сети. Это защищает его направление, **не гарантирует неизменность итоговых соседств**.

Первый этап, seed17,600stochastic steps на режим, AdamW0,001: положительная пара одного хоста в ДВУХ различных train-днях, симметричный InfoNCE(temp0,12,batch128), независимое15% block masking, восстановление среднего профиля часа, прогноз будущих quarter на горизонтах15/30/45минут, сохранение pairwise own-сходств, variance/covariance anti-collapse penalty. Forecast-head получает латент только исходного quarter до временного mixing, поэтому не видит будущие quarter. Градиент проходит в общие input/stable/dynamic-проекции. Восстановление не обязано сохранить каждый конкретный редкий контакт.

У первого этапа per-coordinate SmoothL1 мала после блочной нормировки. Дополнительный заранее описанный этап `continue_identity` масштабирует reconstruction/forecast на128 и вводит full-gallery proxy CE среди6453классов, ещё600steps,lr0,0005. Proxy-веса нужны только для supervision и **не сохраняются/не используются при inference**. Вход сети и на этом этапе — трафик, не собственный ID. Это изменение recipe после просмотра первых development-результатов явно учитывается в оценке; development не независим.

Прогноз реализован внутри часа, cross-day supervision — узнавание хоста. Отдельный прогноз по горизонту нескольких дней и архитектура со скользящим12-дневным контекстом не реализованы: разрешённый набор02–12 содержит11разных дней, а не несколько независимых12-дневных последовательностей. Это ограничение текущего эксперимента, не утверждение о невозможности работы с другими данными.

## Историческое голосование

`adapt` начинает с graph checkpoint первого этапа:200steps,AdamW0,0001,seed17. На каждом шаге выбирается равновероятно историческое target-окно03–12; source всегда02. Forward использует настоящие top10 активных доноров, положительные косинусы и RAW source02 counts. Собственный/старые партнёры исключены, вселенная построена по активным source-донорам. Smooth first-ANY logsumexp на log1p голосов; backward straight-through softmax, **не точный градиент дискретного MRR**. Добавлены KL к предобученной геометрии и междневной Identity. Полный ансамбль доноров участвует, не только батч32запроса. Сеть остаётся общей индуктивной функцией; свободной per-host embedding-таблицы нет.

`continue_identity` сохраняет и этот вариант, но сам продолжает исходные no_graph/graph, а не adapted. Поэтому новое full-gallery обучение не объявляется совместным partner-обучением. Классификатор Links обучается штатным C=1 probe только при оценке; отдельного temporal Links-loss в train пока нет. Pairs сохраняется косвенно через геометрию/контекст, не через вымышленные role-labels.

## Контроли и выбор

- `plain`:склейка тех же четырёх блоков без временной сети,512координат.
- `no_graph`:тот же Mixer и число steps, графовый вход занулён; preprocessing может строить Graph, но сеть его не получает.
- `graph`:полный Mixer.
- `adapted`:graph после исторического voting.
- `*_recent`:в source остаётся own блока исходного02, stable/dynamic усредняются по разрешённым train-наблюдениям с полураспадом3дня, затем нормируются. Late вектор строится по текущему окну тем же encoder. Source один, фиксированный для всех benchmark-окон. Все historical-коды построены одной финальной сетью; отдельного межмодельного выравнивания не требуется.
- `*_global`:контроль полного-gallery CE, исходные checkpoint сохранены.
- Для global historical source отдельно проверяется узкая сетка весов own/stable/dynamic:[1;2;0,5],[1;2;1],[1;1;1]; исходные[2;1;0,5] остаются. Настройки выбираются только на development.

Выбор по полному development-score, строгий guard raw Neighbors не менее95% plain. Для каждого source заново пересчитываются все пять метрик; при другой размерности пересчитывался бы и random Links, но здесь все512. Сначала все source/model/selection SHA фиксируются, затем запускается benchmark. Все экспортируемые CSV пишутся float64 %.17g и **перечитываются** до оценки. Истинный Identity использует штатный Procrustes, direct-направления публикуются отдельно.

Поддерживается inference на произвольном списке новых хостов: никакой финальный словарь own-ID не требуется; граф строится среди переданных query-узлов, другие endpoints остаются контактными признаками. Это архитектурная возможность, не измеренная host-disjoint accuracy. Для строгого host-disjoint теста надо переобучить и preprocessing без отложенных хостов. Нужны rolling-prefix configs с отдельным fit всей цепочки; готовый full-history checkpoint не выдаётся за prefix-модель.

## Данные и воспроизведение

JSON задаёт реальные пути и полуоткрытые окна; переменные окружения разворачиваются. `example.json` отражает бюджет11часов02–12; заменить пути. На13 — выбор без градиентов,14–16 — только замороженная оценка. Все эти дни уже изучались командой: это сопоставимый benchmark, не новый закрытый период. Версия2 разработана после просмотра benchmark версии1; её результат также нельзя называть независимым. Входные flows/SHA/границы/активные когорты/частоты и эталонные CSV проверяются; AppleDouble служебные файлы не являются эталонами. Канонические потолки применяются только при совпадении manifests; для13 нужен локальный manifest с собственным neighbor_ceiling.

```bash
python -m pip install -r requirements.txt torch joblib
export PT_REPO_ROOT=/repo PT_DATA_ROOT=/data PT_RUNS=/runs
export OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2
python -m experiments.temporal_graph_mixer.run --config config.json --output runs/tgm01
# После фиксации на development; флаг запускает уже изученные benchmark14–16:
python -m experiments.temporal_graph_mixer.run --config config.json --output runs/tgm02 --benchmark
# Поэтапно: data --part train; fit; adapt; add_history; continue_identity; add_weights;
# evaluate --part development; затем evaluate --part benchmark с config.selection.
python -m experiments.temporal_graph_mixer.infer --trained runs/tgm01/final_trained \
  --selection runs/tgm01/development/frozen_selection.json --flows late.csv \
  --cohort query_hosts.csv --start START --end END --day DAY --output runs/late.csv
python -m experiments.temporal_graph_mixer.bundle --trained runs/tgm01/final_trained \
  --selection runs/tgm01/development/frozen_selection.json --output runs/tgm_bundle
```

Все output-папки должны быть новыми. `feature_dir` — явно заданный кэш с SHA/схемой; можно переиспользовать, входные файлы не меняются. Пример требует numpy/pandas/scipy/scikit-learn/joblib/torch и проектного evaluator. Versions и SHA хранятся с артефактами. Нужны реальные потоки окна для late-инференса; сами эталоны не являются входом encoder.

## Научная основа

Это авторская комбинация, не полное воспроизведение статей и не основание гарантировать score. [GraphMixer](https://arxiv.org/abs/2302.11636) мотивирует простое MLP temporal learning; [TS2Vec](https://arxiv.org/abs/2106.10466) — несколько масштабов времени; [diffusion scattering](https://arxiv.org/abs/1806.08829) — многомасштабную графовую обработку; [VICReg](https://arxiv.org/abs/2105.04906) — anti-collapse regularization. Теоремы для исходных graph-scattering операторов автоматически на наши направленные каналы не переносятся.

В Git только код/пример/отчёт; flows, feature-cache, torch/joblib checkpoint и CSV остаются в runs. Для передачи выбранной модели: transforms.joblib, selected.pt, frozen_training.json, frozen_selection.json, source.csv и versions; имена selected.pt фактически соответствуют label из selection. Decoder/proxy не нужны для выдачи embedding; proxy-веса вообще не входят в checkpoint.
