# Hybrid v4 vs existing leader

| # | method | submitted by | **score** | dim | Neighbors@10 | Links, AUC | New partners, MRR | Pairs, AUC | Identity, Top-1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `hybrid_tfidf_v4` | — | **45.4** | 633 | 0.235 [0.22–0.25] | **0.989 [0.98–0.99]** | **0.233 [0.22–0.25]** | 0.940 [0.93–0.95] | **0.528 [0.50–0.56]** |
| 2 | `hybrid_tfidf` | @physcorgi | 45.3 | 313 | **0.238 [0.23–0.25]** | 0.985 [0.98–0.99] | 0.225 [0.21–0.24] | **0.941 [0.93–0.95]** | 0.525 [0.50–0.56] |
| | *chance level* | | 0 | | 0.004 | 0.87–0.91 | 0.065 | 0.500 | 0.001 |

Selected recipe:
{
  "name": "candidate_25",
  "recipe": {
    "behavioral": 2.0,
    "peers256": 1.0,
    "services256": 1.0,
    "node2vec": 0.5
  },
  "validation_score": 35.82605999210196
}