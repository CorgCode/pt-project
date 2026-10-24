# Таблица результатов

Каждый метод описывает компьютер в сети набором чисел — вектором. Таблица показывает, насколько эти векторы полезны.
Данные — LANL, корпоративная сеть, компьютеров: 2242. Вектор строится по часу трафика и проверяется на следующих 30 минутах.

**Score** — итог от 0 до 100: 0 — как у случайных векторов, 100 — идеал. Места — по нему.
Название метода — ссылка на его карточку: устройство, какие данные берёт, как обучается.

| # | method | submitted by | **score** | dim | Neighbors@10 | Links, AUC | New partners, MRR | Pairs, AUC | Identity, Top-1 | Triplets* | Cosine links, AUC* |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | [`hybrid_tfidf`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_tfidf/README.md) | @physcorgi | **45.3** | 313 | 0.238 [0.23–0.25] | 0.985 [0.98–0.99] | 0.225 [0.21–0.24] | 0.941 [0.93–0.95] | **0.525 [0.50–0.56]** | 0.959 [0.94–0.97] | 0.301 [0.26–0.35] |
| 2 | [`hybrid_services`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_services/README.md) | @physcorgi | 42.7 | 105 | 0.258 [0.25–0.27] | **0.994 [0.99–1.00]** | **0.235 [0.21–0.25]** | 0.953 [0.94–0.96] | 0.341 [0.31–0.37] | 0.956 [0.94–0.97] | 0.289 [0.26–0.33] |
| 3 | [`hybrid_weighted`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_weighted/README.md) | @physcorgi | 41.7 | 89 | 0.262 [0.25–0.27] | 0.993 [0.99–1.00] | 0.221 [0.20–0.24] | 0.956 [0.95–0.96] | 0.319 [0.29–0.34] | 0.957 [0.94–0.97] | 0.279 [0.25–0.32] |
| 4 | [`behavioral`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/behavioral/README.md) | @physcorgi | 40.4 | 57 | **0.269 [0.26–0.28]** | 0.983 [0.98–0.99] | 0.204 [0.19–0.22] | 0.954 [0.95–0.96] | 0.295 [0.27–0.32] | 0.954 [0.94–0.97] | 0.272 [0.24–0.31] |
| 5 | [`hybrid_behavioral_node2vec`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_behavioral_node2vec/README.md) | @physcorgi | 36.4 | 73 | 0.179 [0.17–0.19] | 0.993 [0.99–1.00] | 0.216 [0.20–0.23] | 0.959 [0.95–0.97] | 0.275 [0.25–0.30] | 0.973 [0.96–0.98] | 0.342 [0.30–0.41] |
| 6 | [`hybrid_behavioral_node2vec_spectral`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_behavioral_node2vec_spectral/README.md) | @physcorgi | 36.3 | 89 | 0.177 [0.17–0.19] | 0.993 [0.99–1.00] | 0.216 [0.20–0.23] | 0.963 [0.96–0.97] | 0.275 [0.25–0.30] | 0.975 [0.96–0.98] | 0.350 [0.32–0.41] |
| 7 | [`hybrid_behavioral_deepwalk`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/hybrid_behavioral_deepwalk/README.md) | @physcorgi | 36.2 | 73 | 0.191 [0.18–0.20] | 0.992 [0.99–0.99] | 0.220 [0.20–0.24] | 0.971 [0.97–0.98] | 0.244 [0.22–0.26] | 0.973 [0.96–0.98] | 0.314 [0.28–0.38] |
| 8 | [`behavioral_pca`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/behavioral_pca/README.md) | @physcorgi | 35.2 | 16 | 0.242 [0.23–0.25] | 0.973 [0.96–0.98] | 0.191 [0.18–0.21] | 0.956 [0.95–0.96] | 0.202 [0.18–0.22] | 0.958 [0.94–0.97] | 0.272 [0.24–0.31] |
| 9 | [`behavioral_svd`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/behavioral_svd/README.md) | @physcorgi | 35.2 | 16 | 0.242 [0.23–0.25] | 0.973 [0.96–0.98] | 0.190 [0.18–0.21] | 0.956 [0.95–0.96] | 0.201 [0.18–0.22] | 0.958 [0.94–0.97] | 0.272 [0.24–0.31] |
| 10 | [`node2vec_services`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/node2vec_services/README.md) | @physcorgi | 23.8 | 32 | 0.088 [0.08–0.09] | 0.822 [0.80–0.84] | 0.181 [0.17–0.20] | 0.906 [0.89–0.92] | 0.200 [0.18–0.22] | 0.911 [0.89–0.93] | **0.791 [0.76–0.82]** |
| 11 | [`deepwalk_services`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/deepwalk_services/README.md) | @physcorgi | 18.3 | 32 | 0.085 [0.08–0.09] | 0.794 [0.78–0.81] | 0.169 [0.16–0.19] | 0.906 [0.89–0.92] | 0.086 [0.07–0.10] | 0.906 [0.88–0.93] | 0.776 [0.74–0.80] |
| 12 | [`node2vec`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/node2vec/README.md) | @physcorgi | 17.6 | 16 | 0.073 [0.07–0.08] | 0.776 [0.75–0.81] | 0.215 [0.20–0.23] | 0.981 [0.98–0.99] | 0.060 [0.05–0.07] | 0.986 [0.98–0.99] | 0.722 [0.69–0.75] |
| 13 | [`deepwalk`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/deepwalk/README.md) | @physcorgi | 16.3 | 16 | 0.075 [0.07–0.08] | 0.801 [0.78–0.82] | 0.204 [0.19–0.22] | 0.977 [0.97–0.98] | 0.042 [0.03–0.05] | 0.981 [0.97–0.99] | 0.777 [0.75–0.80] |
| 14 | [`graph_spectral`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/graph_spectral/README.md) | @physcorgi | 15.0 | 16 | 0.077 [0.07–0.08] | 0.980 [0.97–0.99] | 0.196 [0.18–0.21] | **0.984 [0.98–0.99]** | 0.021 [0.01–0.03] | **0.991 [0.98–1.00]** | 0.064 [0.05–0.08] |
| 15 | [`graphsage`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/graphsage/README.md) | @physcorgi | 13.2 | 16 | 0.129 [0.12–0.14] | 0.917 [0.88–0.94] | 0.168 [0.15–0.18] | 0.835 [0.82–0.85] | 0.013 [0.01–0.02] | 0.863 [0.84–0.89] | 0.399 [0.32–0.53] |
| 16 | [`rolx`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/rolx/README.md) | @physcorgi | 12.4 | 16 | 0.059 [0.05–0.06] | 0.971 [0.96–0.98] | 0.168 [0.15–0.18] | 0.816 [0.79–0.83] | 0.022 [0.02–0.03] | 0.833 [0.81–0.86] | 0.326 [0.29–0.38] |
| 17 | [`struc2vec`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/struc2vec/README.md) | @physcorgi | 3.8 | 16 | 0.027 [0.02–0.03] | 0.842 [0.79–0.89] | 0.104 [0.09–0.11] | 0.530 [0.51–0.55] | 0.004 [0.00–0.01] | 0.534 [0.50–0.57] | 0.486 [0.47–0.50] |
| — | *[`random`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/random/README.md)* | — | 1.2 | 16 | 0.004 [0.00–0.01] | 0.598 [0.54–0.64] | 0.063 [0.06–0.07] | 0.511 [0.50–0.52] | 0.000 [0.00–0.00] | 0.512 [0.49–0.54] | 0.502 [0.49–0.51] |
| — | *[`reference_ceiling`](https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/reference_ceiling/README.md)* | — | 55.1 | 57 | 1.000 [1.00–1.00] | — | 0.230 [0.21–0.25] | 0.973 [0.97–0.98] | — | 0.965 [0.95–0.98] | 0.260 [0.23–0.30] |
| | *chance level* | | 0 | | 0.004 | 0.64–0.87 | 0.065 | 0.500 | 0.001 | 0.500 | 0.500 |

| колонка | что проверяет (больше — лучше) |
|---|---|
| Neighbors@10 | у компьютеров с похожими векторами похожее поведение в сети |
| Links, AUC | по двум векторам можно сказать, будут ли компьютеры обмениваться трафиком |
| New partners, MRR | по похожим компьютерам угадываются их новые собеседники |
| Pairs, AUC | компьютеры с общими собеседниками получают близкие векторы |
| Identity, Top-1 | компьютер узнаётся по вектору через полчаса |
| Triplets* | из двух компьютеров ближе к данному тот, у кого с ним больше общих собеседников |
| Cosine links, AUC* | будут ли компьютеры обмениваться трафиком, видно прямо по близости векторов, без обучения. Ниже 0.5 — близкие векторы, наоборот, реже связаны |

\* Метрики, предложенные Тихоном (#40, #22): показаны для сравнения, в score не входят.

Жирным — лучшее в колонке. В скобках — диапазон, в котором значение лежит с вероятностью 95%.
dim — длина вектора. Chance level — значение при случайном угадывании.
`random` (случайные векторы) и `reference_ceiling` (точный ответ для Neighbors) — ориентиры, в рейтинге не участвуют.

## Как считается score

Каждая колонка переводится в шкалу 0–100 (0 — случайное угадывание, 100 — идеал).
Колонки объединены в четыре задачи: Behavior (Neighbors), Future links (Links и New partners),
Shared partners (Pairs), Identity. Score — среднее геометрическое четырёх задач:
метод, который провалил хотя бы одну задачу, не поднимется высоко за счёт остальных.
