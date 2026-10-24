# Temporal graph encoder v5

| # | method | submitted by | **score** | dim | Neighbors@10 | Links, AUC | New partners, MRR | Pairs, AUC | Identity, Top-1 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | `hybrid_temporal_graph_v5` | — | **46.9** | 825 | 0.226 [0.22–0.24] | 0.982 [0.98–0.99] | **0.260 [0.24–0.28]** | 0.937 [0.93–0.95] | **0.617 [0.59–0.65]** |
| 2 | `hybrid_tfidf` | @physcorgi | 45.3 | 313 | **0.238 [0.23–0.25]** | **0.985 [0.98–0.99]** | 0.225 [0.21–0.24] | **0.941 [0.93–0.95]** | 0.525 [0.50–0.56] |
| | *chance level* | | 0 | | 0.004 | 0.87–0.92 | 0.065 | 0.500 | 0.001 |

Selected recipe:
{
  "name": "v5_candidate_07",
  "recipe": {
    "behavioral": 2.0,
    "peers": 1.0,
    "services": 1.0,
    "diffusion": 0.5,
    "neural": 1.0
  },
  "checkpoint": 20,
  "validation_score": 38.88684825195815
}