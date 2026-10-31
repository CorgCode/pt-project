import cloudpickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

from experiments.fingerprint_quick import Projection


def test_projection_matches_frozen_pca_on_unseen_data():
    rng=np.random.default_rng(3)
    train=rng.normal(size=(80,12)).astype(np.float32)
    future=rng.normal(size=(20,12)).astype(np.float32)
    pca=PCA(n_components=6).fit(normalize(train))
    projection=Projection(pca,6,0)
    np.testing.assert_allclose(projection.transform(future),normalize(pca.transform(normalize(future))),atol=1e-5)
    mean=projection.mean.copy()
    before=projection.transform(train)
    projection.transform(future*1000)
    np.testing.assert_array_equal(mean,projection.mean)
    np.testing.assert_array_equal(before,projection.transform(train))


def test_partial_whitening_is_finite_permutation_equivariant_and_serializable():
    rng=np.random.default_rng(4)
    train=rng.normal(size=(60,10)).astype(np.float32)
    pca=PCA(n_components=6).fit(normalize(train))
    model=Projection(pca,4,0.5)
    restored=cloudpickle.loads(cloudpickle.dumps(model))
    expected=model.transform(train)
    assert expected.shape==(60,4) and np.isfinite(expected).all()
    np.testing.assert_allclose(restored.transform(train[::-1]),expected[::-1],atol=1e-6)
