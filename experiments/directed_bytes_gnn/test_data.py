import numpy as np
import pandas as pd
from scipy import sparse

from .data import add_flow_chunk, features


def test_bytes_count_forward_and_reverse_for_each_host():
    hosts = pd.Index(['A', 'B'])
    data = pd.DataFrame([
        ['A', 'B', 100, 40, 10, 4],
        ['B', 'A', 30, 20, 3, 2],
        ['A', 'external', 7, 5, 1, 2],
        ['A', 'A', 999, 999, 999, 999],
    ], columns=['src', 'dst', 'bytes', 'resp_bytes', 'packets', 'resp_packets'])
    totals, count, byte = add_flow_chunk(np.zeros((2, 6)), sparse.csr_matrix((2, 2)),
                                        sparse.csr_matrix((2, 2)), data, hosts)
    np.testing.assert_array_equal(totals, [[127, 75, 13, 9, 2, 1], [70, 120, 7, 12, 1, 1]])
    np.testing.assert_array_equal(byte.toarray(), [[0, 120], [70, 0]])
    np.testing.assert_array_equal(count.toarray(), [[0, 1], [1, 0]])
    f, g, active = features(totals, count, byte, include_bytes=False)
    assert (f[:, :2] == 0).all() and g[2].nnz == g[3].nnz == 0
    np.testing.assert_array_equal(active, [3, 2])
