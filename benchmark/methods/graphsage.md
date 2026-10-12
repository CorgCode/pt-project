# GraphSAGE

Hamilton, Ying, Leskovec, *Inductive Representation Learning on Large Graphs*, NeurIPS 2017. Implementation: `graphsage.py`.

Unsupervised, mean aggregator, two layers (module docstring has the equations). Hand-written backward pass verified
against finite differences.

* **Inductive:** weights are trained once on the train window; later windows are embedded by deterministic
  full-neighbourhood inference with those weights (~10 ms for 8,000 nodes), so embeddings of different windows share ONE
  coordinate system (`shared_coordinate_system: True`; the raw-coordinate identity score is meaningful, though the inputs
  drift between windows).
* **Features** (`features`): `behavioral` (default; train-window behavioural profile, standardised with TRAIN statistics; the
  same feature definitions as the behavioural reference, a documented overlap), `structural`, or `degree`. With degree or
  structural features it is much weaker on this benchmark's references (fixture: pair AUC 0.62 vs 0.88).
* **Loss floor:** embeddings are L2-normalised, so negatives cannot score below about 0 on average; with Q = 20 the per-pair loss
  cannot fall under ~14.1 (it starts near 15), so a small absolute decrease is expected.
* **Scale** (measured, one core, paper sampling sizes 25/10, hidden 64): ~25k training pairs/s; default settings (10 walks of
  length 5, window 2, 2 epochs) are ~4 min at 20,000 hosts and ~19 min at 100,000 (extrapolated).
