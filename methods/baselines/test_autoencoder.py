"""Smoke tests for the train-only neural baseline; uses the repository's `processed` fixture."""
import numpy as np
import pytest

pytest.importorskip("torch")

from methods.baselines.autoencoder import BehavioralDAE
from pipeline.data import ProcessedData


class TrainOnlyGuard(ProcessedData):
    """Raise if fit tries to read validation/test flows."""
    fitting = False

    def flows(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() tried to read {window!r} window")
        return super().flows(window)


@pytest.fixture(scope="module")
def guarded_data(processed):
    return TrainOnlyGuard(processed)


def test_fit_reads_only_train(guarded_data):
    guarded_data.fitting = True
    try:
        BehavioralDAE(dim=6, hidden_dim=24, epochs=2, batch_size=128).fit(guarded_data)
    finally:
        guarded_data.fitting = False


def test_embedding_contract_and_shared_coordinates(guarded_data):
    model = BehavioralDAE(dim=6, hidden_dim=24, epochs=3, batch_size=128).fit(guarded_data)
    train_hosts = list(guarded_data.cohort)
    train = model.embed(guarded_data, "train", train_hosts)
    assert train.shape == (len(train_hosts), 6)
    assert train.index.name == "host_id"
    assert all(c.startswith("emb_") for c in train.columns)
    assert np.isfinite(train.to_numpy()).all()
    assert model.shared_coordinate_system

    if "test" in guarded_data.windows:
        test_hosts = guarded_data.evaluable("test")
        test = model.embed(guarded_data, "test", test_hosts)
        assert test.shape == (len(test_hosts), 6)
        assert np.isfinite(test.to_numpy()).all()


def test_fit_is_reproducible(guarded_data):
    a = BehavioralDAE(dim=6, hidden_dim=24, epochs=3, batch_size=128, seed=21).fit(guarded_data)
    b = BehavioralDAE(dim=6, hidden_dim=24, epochs=3, batch_size=128, seed=21).fit(guarded_data)
    za = a.embed(guarded_data, "train").to_numpy()
    zb = b.embed(guarded_data, "train").to_numpy()
    np.testing.assert_allclose(za, zb, rtol=1e-5, atol=1e-6)


def test_denoising_loss_usually_decreases(guarded_data):
    model = BehavioralDAE(dim=6, hidden_dim=24, epochs=8, batch_size=128, seed=23).fit(guarded_data)
    assert len(model.loss_history_) == 8
    assert np.isfinite(model.loss_history_).all()
    assert model.loss_history_[-1] < model.loss_history_[0]
