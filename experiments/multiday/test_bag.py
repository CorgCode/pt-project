import cloudpickle
import numpy as np
import scipy.sparse as sp

from experiments.multiday.bag import BagConfig, BagHead, hash_sparse, train_head


def test_hash_collisions_are_summed_and_normalized():
    x = sp.csr_matrix([[1., 2., 3.]])
    actual = hash_sparse(x, np.array([0, 0, 1]), 2).toarray()
    np.testing.assert_allclose(actual, [[2**-0.5, 2**-0.5]], atol=1e-6)


def test_bag_predictions_follow_features_and_empty_rows_are_zero():
    config = BagConfig(width=8, dim=4)
    head = BagHead(config, 3)
    x = sp.csr_matrix(np.eye(3, 8, dtype=np.float32))
    for mode in ['neural', 'posterior']:
        expected = head.encode(x, mode)
        np.testing.assert_allclose(head.encode(x[[2, 0, 1]], mode), expected[[2, 0, 1]], atol=1e-6)
        assert not head.encode(sp.csr_matrix((1, 8), dtype=np.float32), mode).any()


def test_small_training_checkpoint_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setenv('PT_TORCH_THREADS', '1')
    x = sp.csr_matrix(np.tile(np.eye(3, 8, dtype=np.float32), (2, 1)))
    train_head(x, ['a', 'b', 'c'], np.eye(3), BagConfig(width=8, dim=4), tmp_path, checkpoints=(2,))
    with (tmp_path / 'epoch2.pkl').open('rb') as stream:
        head = cloudpickle.load(stream)
    assert np.isfinite(head.encode(x)).all()
    assert (tmp_path / 'complete.json').exists()
