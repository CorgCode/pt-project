# Fresh temporal holdout: day02 15:00–15:30

Training: day02 11:00–12:00. Selection: 12:00–12:30. This late window was fixed before v6 selection. No model is fitted or retuned on it. It is a later window of the same day, not an independent day.

| # | method | submitted by | **score** | dim | Neighbors@10 | Links, AUC | New partners, MRR | Pairs, AUC | Identity, Top-1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `hybrid_adaptive_fast_v6` | — | **46.5** | 384 | 0.205 [0.19–0.22] | 0.986 [0.98–0.99] | **0.296 [0.28–0.31]** | 0.944 [0.94–0.95] | **0.567 [0.53–0.59]** |
| 2 | `hybrid_temporal_graph_v5` | — | 45.9 | 825 | 0.205 [0.19–0.22] | 0.986 [0.98–0.99] | 0.295 [0.28–0.31] | 0.944 [0.94–0.95] | 0.565 [0.53–0.59] |
| 3 | `hybrid_tfidf_reproduced` | — | 44.1 | 313 | **0.214 [0.20–0.23]** | **0.988 [0.98–0.99]** | 0.260 [0.25–0.28] | **0.948 [0.94–0.96]** | 0.470 [0.44–0.50] |
| | *chance level* | | 0 | | 0.004 | 0.87–0.92 | 0.074 | 0.500 | 0.001 |
