# RolX

Henderson et al., *RolX: Structural Role Extraction & Mining in Large Graphs*, KDD 2012 (ReFeX: KDD 2011).
Implementation: `rolx.py`.

ReFeX recursive structural features (degree, strength, egonet edges inside/outside; sum and mean over neighbours;
logarithmic binning with similarity pruning) followed by KL-divergence NMF; the L1-normalised role memberships are the
embedding. `n_roles` defaults to the embedding dimension (`--dim`); `n_roles: "mdl"` selects it by minimum description
length.

* **What it is for:** structural roles are recovered across disconnected components (on 8 copies of a
  hub/leaf/bridge/clique component the nearest other-component node has the same role for >= 95% of nodes of every role);
  proximity methods such as DeepWalk have no mechanism for that.
* **What it is not for:** the references of this benchmark reward shared peers and behaviour, so RolX is expected to align
  only weakly with them (declared overlap: weak). On the synthetic fixture: pair AUC 0.59, kNN@10 0.07.
* **MDL caveat:** on clean role graphs the model cost dominates and MDL picks the minimum number of roles (2); this is a
  property of the criterion at these feature counts, not evidence about the true number of roles.
* **Cost:** negligible (0.1 s for 300 hosts); triangles are counted row-chunked so hubs do not exhaust memory.
