import numpy as np
import pandas as pd
import pytest

from methods.baselines.temporal import (BehavioralBaseline, BehavioralPCA, BehavioralSVD,
                                          GraphSpectral, RandomBaseline, default_baselines)
from pipeline.data import ProcessedData


class TrainOnlyGuard(ProcessedData):
    """Громко падает, если обучение трогает что-то, кроме train-окна."""
    fitting = False

    def flows(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} window")
        return super().flows(window)

    def graph(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} graph")
        return super().graph(window)


@pytest.fixture(scope="module")
def data(processed):
    return TrainOnlyGuard(processed)


@pytest.mark.parametrize("make", [RandomBaseline, BehavioralBaseline, BehavioralPCA, BehavioralSVD, GraphSpectral])
def test_fit_reads_only_the_train_window(make, data):
    data.fitting = True
    try:
        make().fit(data)
    finally:
        data.fitting = False


@pytest.mark.parametrize("b", default_baselines(dim=8), ids=lambda b: b.name)
def test_output_contract(b, data):
    e = b.fit(data).embed(data, "train")
    assert list(e.index) == data.cohort and e.index.name == "host_id"
    assert all(c.startswith("emb_") for c in e.columns)
    assert np.isfinite(e.to_numpy()).all() and (np.linalg.norm(e.to_numpy(), axis=1) > 0).all()
    assert e.shape[1] <= 56


def test_dimension_is_configurable_and_deterministic(data):
    a = BehavioralPCA(dim=5, seed=1).fit(data).embed(data, "train")
    b = BehavioralPCA(dim=5, seed=1).fit(data).embed(data, "train")
    assert a.shape[1] == 5 and np.array_equal(a.to_numpy(), b.to_numpy())
    g1 = GraphSpectral(dim=6, seed=2).embed(data, "train")
    g2 = GraphSpectral(dim=6, seed=2).embed(data, "train")
    np.testing.assert_allclose(g1.to_numpy(), g2.to_numpy())


def test_random_seed_changes_output_and_windows_are_independent(data):
    r0, r1 = RandomBaseline(4, 0), RandomBaseline(4, 1)
    assert not np.array_equal(r0.embed(data, "train"), r1.embed(data, "train"))
    assert not np.array_equal(r0.embed(data, "train"), r0.embed(data, "test"))


def test_shared_coordinates_flag_matches_behaviour(data):
    b = BehavioralPCA(dim=6).fit(data)
    assert b.shared_coordinate_system and not GraphSpectral(6).shared_coordinate_system
    # фиксированное преобразование с train даёт ТЕ ЖЕ координаты для тех же потоков
    tr = b.embed(data, "train")
    again = b._model.transform(b._features(data, "train").to_numpy())
    np.testing.assert_allclose(tr.to_numpy(), again)


def test_other_windows_keep_train_columns_and_scale(data):
    b = BehavioralBaseline().fit(data)
    tr, te = b.embed(data, "train"), b.embed(data, "test")
    assert list(tr.columns) == list(te.columns)
    live = te.std() > 1e-12                      # колонка, постоянная в окне, остаётся нулём
    np.testing.assert_allclose(te.std()[live].to_numpy(), tr.std()[live].to_numpy(), rtol=1e-6)


class Doubled(ProcessedData):
    """Test-окно, где каждый поток повторён дважды: как окно другой длины с тем же поведением."""

    def flows(self, window):
        f = super().flows(window)
        return pd.concat([f, f], ignore_index=True) if window == "test" else f


def test_window_length_does_not_shift_behavioral(processed):
    plain, doubled = ProcessedData(processed), Doubled(processed)
    b = BehavioralBaseline().fit(plain)
    diff = np.abs(b.embed(doubled, "test").to_numpy() - b.embed(plain, "test").to_numpy()).max()
    assert diff < 0.05                           # log1p(2x) — почти сдвиг; без пересчёта масштаба разница ~0.2


def test_baselines_score_above_random_on_their_matching_reference(data, reference, tmp_path):
    from evaluation.evaluate import run
    scores = {}
    for b in [RandomBaseline(8), BehavioralPCA(8)]:
        b.fit(data)
        path = tmp_path / f"{b.name}.csv"
        b.embed(data, "train").to_csv(path)
        scores[b.name] = run(path, reference / "test", b.name)["metrics"]["consistency@10_mean"]
    assert scores["behavioral_pca"] > scores["random"] + 0.05
