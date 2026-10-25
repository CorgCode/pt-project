# Таблица результатов

Каждый метод описывает компьютер в сети набором чисел — вектором. Таблица показывает, насколько эти векторы полезны.
Данные — LANL, корпоративная сеть, компьютеров: 5445. Вектор строится по 1 ч трафика (день 02, 11:00–12:00) и проверяется на 3 отрезках: день 02, 12:30–13:30; день 03, 12:30–13:30; день 04, 12:30–13:30. Итог — среднее по ним.

**Score** — итог от 0 до 100: 0 — как у случайных векторов, 100 — идеал. Места — по нему.
Название метода — ссылка на его карточку: устройство, какие данные берёт, как обучается.

| # | method | submitted by | **score** | dim | Neighbors@10 | Links, AUC | New partners, MRR | Pairs, AUC | Identity, Top-1 | Tikhon LP, MRR* | Tikhon transfer, MRR* | Tikhon soft transfer, MRR* |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [`temporal_encoder`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/temporal_encoder/README.md) | @physcorgi | **39.1** | 384 | 0.129 [0.12–0.14] | 0.987 [0.99–0.99] | 0.308 [0.30–0.31] | 0.976 [0.97–0.98] | **0.409 [0.36–0.50]** | 0.010 [0.01–0.01] | 0.912 [0.91–0.92] | 0.876 [0.87–0.88] |
| 2 | [`temporal_encoder_tuned`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/temporal_encoder_tuned/README.md) | @physcorgi | 38.9 | 384 | 0.131 [0.12–0.14] | 0.986 [0.99–0.99] | **0.310 [0.29–0.33]** | 0.975 [0.97–0.98] | 0.397 [0.33–0.50] | 0.009 [0.01–0.01] | 0.911 [0.91–0.92] | 0.877 [0.87–0.88] |
| 3 | [`temporal_cca_tuned`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/temporal_cca_tuned/README.md) | @physcorgi | 38.2 | 384 | 0.135 [0.12–0.14] | 0.985 [0.98–0.99] | 0.301 [0.29–0.31] | 0.975 [0.97–0.98] | 0.362 [0.31–0.44] | 0.009 [0.01–0.01] | 0.905 [0.90–0.91] | 0.864 [0.86–0.87] |
| 4 | [`hybrid_tfidf`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_tfidf/README.md) | @physcorgi | 37.7 | 569 | 0.131 [0.12–0.14] | 0.982 [0.98–0.98] | 0.296 [0.27–0.31] | 0.978 [0.97–0.98] | 0.364 [0.33–0.42] | 0.009 [0.01–0.01] | 0.910 [0.91–0.92] | 0.869 [0.87–0.87] |
| 5 | [`hybrid_services`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_services/README.md) | @physcorgi | 34.1 | 105 | 0.141 [0.13–0.15] | **0.994 [0.99–0.99]** | 0.301 [0.28–0.31] | 0.983 [0.98–0.99] | 0.210 [0.18–0.23] | 0.009 [0.01–0.01] | 0.901 [0.90–0.90] | 0.858 [0.86–0.86] |
| 6 | [`hybrid_weighted`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_weighted/README.md) | @physcorgi | 33.0 | 89 | 0.138 [0.13–0.15] | 0.994 [0.99–0.99] | 0.292 [0.27–0.30] | 0.985 [0.98–0.99] | 0.193 [0.17–0.22] | 0.009 [0.01–0.01] | 0.907 [0.90–0.91] | 0.869 [0.87–0.87] |
| 7 | [`behavioral`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/behavioral/README.md) | @physcorgi | 31.9 | 57 | **0.145 [0.13–0.16]** | 0.984 [0.98–0.99] | 0.262 [0.24–0.27] | 0.983 [0.98–0.99] | 0.176 [0.16–0.19] | 0.008 [0.01–0.01] | 0.891 [0.89–0.89] | 0.808 [0.81–0.81] |
| 8 | [`hybrid_behavioral_node2vec`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_behavioral_node2vec/README.md) | @physcorgi | 28.8 | 73 | 0.097 [0.09–0.10] | 0.993 [0.99–0.99] | 0.288 [0.26–0.30] | 0.987 [0.98–0.99] | 0.159 [0.14–0.17] | 0.025 [0.02–0.03] | 0.928 [0.93–0.93] | **0.909 [0.91–0.91]** |
| 9 | [`hybrid_behavioral_node2vec_spectral`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_behavioral_node2vec_spectral/README.md) | @physcorgi | 28.6 | 89 | 0.096 [0.09–0.10] | 0.994 [0.99–0.99] | 0.289 [0.26–0.30] | 0.989 [0.99–0.99] | 0.156 [0.14–0.17] | 0.022 [0.02–0.02] | **0.928 [0.93–0.93]** | 0.899 [0.90–0.90] |
| 10 | [`hybrid_behavioral_deepwalk`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_behavioral_deepwalk/README.md) | @physcorgi | 28.3 | 73 | 0.103 [0.09–0.11] | 0.992 [0.99–0.99] | 0.289 [0.27–0.30] | 0.992 [0.99–0.99] | 0.139 [0.12–0.15] | 0.023 [0.02–0.02] | 0.924 [0.92–0.93] | 0.887 [0.88–0.89] |
| 11 | [`behavioral_pca`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/behavioral_pca/README.md) | @physcorgi | 27.0 | 16 | 0.130 [0.12–0.14] | 0.983 [0.98–0.98] | 0.256 [0.24–0.27] | 0.984 [0.98–0.99] | 0.101 [0.10–0.11] | 0.007 [0.01–0.01] | 0.891 [0.89–0.89] | 0.800 [0.80–0.81] |
| 12 | [`behavioral_svd`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/behavioral_svd/README.md) | @physcorgi | 26.9 | 16 | 0.130 [0.12–0.14] | 0.983 [0.98–0.98] | 0.256 [0.24–0.27] | 0.984 [0.98–0.99] | 0.101 [0.10–0.10] | 0.007 [0.01–0.01] | 0.891 [0.89–0.89] | 0.800 [0.80–0.81] |
| 13 | [`node2vec_services`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/node2vec_services/README.md) | @physcorgi | 19.8 | 32 | 0.051 [0.05–0.05] | 0.831 [0.82–0.84] | 0.264 [0.24–0.28] | 0.935 [0.92–0.95] | 0.111 [0.10–0.12] | **0.177 [0.17–0.18]** | 0.887 [0.88–0.89] | 0.767 [0.76–0.77] |
| 14 | [`node2vec`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/node2vec/README.md) | @physcorgi | 16.2 | 16 | 0.045 [0.04–0.05] | 0.815 [0.81–0.82] | 0.288 [0.26–0.30] | 0.992 [0.99–0.99] | 0.048 [0.05–0.05] | 0.091 [0.09–0.09] | 0.916 [0.91–0.92] | 0.578 [0.55–0.59] |
| 15 | [`deepwalk_services`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/deepwalk_services/README.md) | @physcorgi | 16.1 | 32 | 0.049 [0.05–0.05] | 0.822 [0.81–0.83] | 0.263 [0.24–0.28] | 0.952 [0.94–0.96] | 0.051 [0.05–0.06] | 0.145 [0.14–0.15] | 0.893 [0.89–0.89] | 0.777 [0.77–0.78] |
| 16 | [`deepwalk`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/deepwalk/README.md) | @physcorgi | 13.7 | 16 | 0.044 [0.04–0.05] | 0.812 [0.80–0.82] | 0.273 [0.25–0.29] | 0.992 [0.99–0.99] | 0.027 [0.02–0.03] | 0.084 [0.08–0.09] | 0.921 [0.92–0.92] | 0.804 [0.80–0.80] |
| 17 | [`graph_spectral`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/graph_spectral/README.md) | @physcorgi | 13.5 | 16 | 0.046 [0.04–0.05] | 0.986 [0.99–0.99] | 0.252 [0.23–0.26] | **0.999 [1.00–1.00]** | 0.018 [0.01–0.03] | 0.008 [0.01–0.01] | 0.896 [0.89–0.90] | 0.370 [0.36–0.38] |
| 18 | [`graphsage`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/graphsage/README.md) | @physcorgi | 11.7 | 16 | 0.075 [0.07–0.08] | 0.916 [0.91–0.92] | 0.232 [0.23–0.24] | 0.866 [0.84–0.89] | 0.005 [0.00–0.01] | 0.021 [0.02–0.02] | 0.899 [0.90–0.90] | 0.842 [0.84–0.85] |
| 19 | [`rolx`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/rolx/README.md) | @physcorgi | 11.6 | 16 | 0.041 [0.04–0.04] | 0.981 [0.98–0.98] | 0.208 [0.19–0.22] | 0.907 [0.90–0.92] | 0.016 [0.01–0.02] | 0.008 [0.01–0.01] | 0.839 [0.83–0.84] | 0.714 [0.71–0.72] |
| 20 | [`struc2vec`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/struc2vec/README.md) | @physcorgi | 4.2 | 16 | 0.018 [0.02–0.02] | 0.844 [0.84–0.85] | 0.170 [0.17–0.17] | 0.539 [0.54–0.54] | 0.001 [0.00–0.00] | 0.010 [0.01–0.01] | 0.779 [0.77–0.78] | 0.602 [0.60–0.61] |
| — | *[`random`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/random/README.md)* | — | 1.1 | 16 | 0.002 [0.00–0.00] | 0.575 [0.57–0.58] | 0.099 [0.10–0.10] | 0.501 [0.50–0.51] | 0.001 [0.00–0.00] | 0.021 [0.02–0.02] | 0.645 [0.63–0.65] | 0.429 [0.42–0.44] |
| — | *[`reference_ceiling`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/reference_ceiling/README.md)* | — | 56.4 | 57 | 1.000 [1.00–1.00] | — | 0.252 [0.24–0.26] | 0.989 [0.99–0.99] | — | 0.008 [0.01–0.01] | 0.850 [0.84–0.86] | 0.761 [0.75–0.77] |
| | *chance level* | | 0 | | 0.002 | 0.61–0.84 | 0.085 | 0.500 | 0.000 | 0.021 | 0.021 | 0.021 |

| колонка | что проверяет (больше — лучше) |
|---|---|
| Neighbors@10 | у компьютеров с похожими векторами похожее поведение в сети |
| Links, AUC | по двум векторам можно сказать, будут ли компьютеры обмениваться трафиком |
| New partners, MRR | по похожим компьютерам угадываются их новые собеседники |
| Pairs, AUC | компьютеры с общими собеседниками получают близкие векторы |
| Identity, Top-1 | компьютер узнаётся по вектору, построенному по более позднему трафику |
| Tikhon LP, MRR* | собеседники компьютера в будущем — это компьютеры с близкими векторами (идея 1) |
| Tikhon transfer, MRR* | компьютер будет общаться с теми, с кем раньше общались 10 самых похожих на него (идея 3) |
| Tikhon soft transfer, MRR* | то же, но угадываются компьютеры, похожие на собеседников похожих (идея 3, мягкий вариант) |

\* Метрики Тихона (#40, PR #65, [`evaluation/tikhon.py`](../evaluation/tikhon.py)): показаны для сравнения, в score не входят.
Chance level для них — случайный порядок кандидатов. Без эмбеддинга «с кем общался раньше» даёт 0.974 — у Тихона это бейзлайн history.

Жирным — лучшее в колонке. В скобках — от худшего до лучшего значения по test-окнам (дням).
dim — длина вектора. Chance level — значение при случайном угадывании.
`random` (случайные векторы) и `reference_ceiling` (точный ответ для Neighbors) — ориентиры, в рейтинге не участвуют.

## Как считается score

Каждая колонка переводится в шкалу 0–100 (0 — случайное угадывание, 100 — идеал).
Колонки объединены в четыре задачи: Behavior (Neighbors), Future links (Links и New partners),
Shared partners (Pairs), Identity. Score — среднее геометрическое четырёх задач:
метод, который провалил хотя бы одну задачу, не поднимется высоко за счёт остальных. Для нескольких test-окон каждая колонка — среднее по окнам.

## По дням

Score на каждом test-окне отдельно. Хороший метод держится на всех днях, а не только на соседнем с обучением.

| method | день 02, 12:30–13:30 | день 03, 12:30–13:30 | день 04, 12:30–13:30 | **mean** |
|---|---|---|---|---|
| `temporal_encoder` | 42.0 | 38.5 | 36.9 | 39.1 |
| `temporal_encoder_tuned` | 42.4 | 38.3 | 36.0 | 38.9 |
| `temporal_cca_tuned` | 41.2 | 37.8 | 35.6 | 38.2 |
| `hybrid_tfidf` | 39.2 | 37.6 | 36.2 | 37.7 |
| `hybrid_services` | 35.4 | 34.7 | 32.3 | 34.1 |
| `hybrid_weighted` | 34.3 | 33.2 | 31.6 | 33.0 |
| `behavioral` | 32.7 | 32.5 | 30.6 | 31.9 |
| `hybrid_behavioral_node2vec` | 29.3 | 29.3 | 27.6 | 28.8 |
| `hybrid_behavioral_node2vec_spectral` | 29.3 | 29.1 | 27.3 | 28.6 |
| `hybrid_behavioral_deepwalk` | 28.5 | 29.2 | 27.1 | 28.3 |
| `behavioral_pca` | 27.1 | 27.5 | 26.3 | 27.0 |
| `behavioral_svd` | 27.1 | 27.5 | 26.2 | 26.9 |
| `node2vec_services` | 20.0 | 20.5 | 18.8 | 19.8 |
| `node2vec` | 16.7 | 16.2 | 15.7 | 16.2 |
| `deepwalk_services` | 16.1 | 17.0 | 15.3 | 16.1 |
| `deepwalk` | 13.8 | 13.2 | 14.2 | 13.7 |
| `graph_spectral` | 11.6 | 15.1 | 13.8 | 13.5 |
| `graphsage` | 11.9 | 12.1 | 11.3 | 11.7 |
| `rolx` | 11.8 | 11.2 | 11.7 | 11.6 |
| `struc2vec` | 4.2 | 4.3 | 4.0 | 4.2 |
| `random` | 1.0 | 1.1 | 1.1 | 1.1 |
| `reference_ceiling` | 57.1 | 56.8 | 55.1 | 56.4 |
