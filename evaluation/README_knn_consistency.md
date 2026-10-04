# k-NN consistency (neighborhood preservation)

Metric for embedding quality. For each host:

1. take the top-k neighbours under an **independent reference similarity**
   built from raw behaviour (`build_reference_features`);
2. take the top-k neighbours in the **embedding space**;
3. compare the two sets (the host itself is always excluded).

| metric | definition |
|---|---|
| `Consistency@k` | `\|R ∩ E\| / k` |
| `Jaccard@k` | `\|R ∩ E\| / \|R ∪ E\|` |
| `nDCG@k` (rank-aware) | embedding list scored with relevance `k − ref_rank`, normalised by the ideal ordering |

Reported per host, and as mean / median / std for `k = 5, 10, 20`.

## Usage

```python
from knn_consistency import build_reference_features, evaluate_embedding

ref = build_reference_features(flows, hosts=emb.index)   # flows: src,dst,dst_port,proto,bytes,packets,ts
summary = evaluate_embedding(emb, ref, ks=(5, 10, 20))   # emb: DataFrame indexed by host
summary, per_host = evaluate_embedding(emb, ref, return_per_host=True)
```

Any reference matrix (hosts × features) can be passed instead of the default one.
Run the synthetic check with `python demo_knn_consistency.py`, tests with `pytest`.

## Reference similarity

Cosine on a blockwise-standardised profile: volume (flows/bytes/packets, in/out),
peer degree, out/in ratio, top-port and protocol distributions, 24-bin hourly activity.

## Leakage

- **Forbidden:** using the evaluated embedding representation itself (or a copy / linear
  transform of it) as the reference.
- **Allowed with care:** behavioural features may be used even if the evaluated method was
  trained on related features, but the overlap inflates scores and must be stated.
- **Preferred:**
  - held-out feature groups (e.g. embed from ports/peers, reference from volume/temporal);
  - a different time window for the reference than for training;
  - an independent similarity view (e.g. host-event / WLS data vs. netflow).
- Peer identity is not in the default reference, since graph embeddings trained on the same
  graph would reproduce it.

## Limitations

- Chance level is about `k / (n − 1)` and grows with k; compare at equal k and against a random baseline.
- The reference is a proxy, not ground truth; results depend on its features and block weights.
- Ties/duplicate hosts give arbitrary neighbour order (affects nDCG most).
- Brute-force search is O(n²); sample or use ANN for very large host sets.
