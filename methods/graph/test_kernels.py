import numpy as np
import pytest

from methods.graph.sgns import SkipGram
from methods.graph.struc2vec import EMPTY_DEGREE, dtw
from methods.graph import kernels


@pytest.mark.parametrize("objective", ["neg", "hs"])
def test_numba_skipgram_matches_numpy(objective):
    rng = np.random.default_rng(0)
    walks = rng.integers(0, 200, (300, 30))
    freq = np.bincount(walks.ravel(), minlength=200).astype(float)
    a, b = (SkipGram(200, 8, objective=objective, engine=e, seed=1, node_freq=freq, max_chunk_pairs=20000).fit(walks)
            for e in ("numpy", "numba"))
    assert np.abs(a.W_in - b.W_in).max() < 1e-10
    assert np.allclose(a.loss_history, b.loss_history)


def test_compiled_dtw_matches_reference():
    rng = np.random.default_rng(2)
    for _ in range(200):
        a = np.unique(rng.integers(1, 40, rng.integers(0, 10))).astype(float)
        b = np.unique(rng.integers(1, 40, rng.integers(0, 10))).astype(float)
        ca, cb = rng.integers(1, 5, len(a)).astype(float), rng.integers(1, 5, len(b)).astype(float)
        assert kernels.dtw(a, ca, b, cb, EMPTY_DEGREE) == dtw(a, ca, b, cb)
