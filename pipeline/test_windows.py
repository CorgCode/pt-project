import json

import numpy as np
import pytest

from pipeline.windows import PipelineConfig, window_mask


def cfg(**windows):
    return PipelineConfig(windows=windows)


def test_valid_config_roundtrip(tmp_path):
    c = cfg(train=(0, 10), validation=(10, 20), test=(25, 30))
    p = tmp_path / "c.json"
    p.write_text(json.dumps(c.to_dict()))
    assert PipelineConfig.from_json(p) == c
    assert c.ordered == ["train", "validation", "test"]


@pytest.mark.parametrize("windows", [
    dict(train=(0, 10)),                                   # test missing
    dict(train=(0, 10), test=(5, 20)),                     # overlap
    dict(train=(10, 20), test=(0, 5)),                     # test before train
    dict(train=(0, 10), validation=(10, 20), test=(15, 30)),  # validation/test overlap
    dict(train=(5, 5), test=(6, 9)),                       # empty window
    dict(train=(0, 10), test=(10, 20), holdout=(30, 40)),  # unknown name
    dict(train=(0, float("inf")), test=(1, 2)),
])
def test_invalid_windows_rejected(windows):
    with pytest.raises(ValueError):
        PipelineConfig(windows=windows)


def test_touching_windows_are_allowed_half_open():
    cfg(train=(0, 10), test=(10, 20))


def test_small_cohort_threshold_rejected():
    with pytest.raises(ValueError):
        PipelineConfig(windows={"train": (0, 1), "test": (2, 3)}, min_cohort_size=10)


def test_window_mask_half_open_and_straddling():
    t = np.array([0.0, 5.0, 9.0, 10.0])
    d = np.array([0.0, 1.0, 2.0, 0.0])
    assert window_mask(t, d, (0, 10), drop_straddling=False).tolist() == [True, True, True, False]
    assert window_mask(t, d, (0, 10), drop_straddling=True).tolist() == [True, True, False, False]
