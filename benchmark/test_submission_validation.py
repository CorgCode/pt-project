"""Boundary checks for canonical submission validation."""
from __future__ import annotations

import pandas as pd
import pytest

from benchmark.validate_submission import validate_submission
from evaluation.evaluate import _load_matrix


def test_validates_and_preserves_lexical_host_ids(tmp_path):
    file = tmp_path / "embeddings.csv"
    pd.DataFrame({
        "host_id": ["001", "1", "02"],
        "emb_0": [1.0, 2.0, 3.0],
        "emb_1": [0.0, 1.0, 1.0],
    }).to_csv(file, index=False)
    validate_submission(str(file))
    mat = _load_matrix(file, "host_id", "emb_")
    assert list(mat.index) == ["001", "1", "02"]


@pytest.mark.parametrize("invalid_id", ["", "  "])
def test_rejects_empty_or_blank_host_id(tmp_path, invalid_id):
    file = tmp_path / "embedding.csv"
    pd.DataFrame({"host_id": ["ok", invalid_id],
                  "emb_0": [1.0, 2.0]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="host_id"):
        validate_submission(str(file))


@pytest.mark.parametrize("bad_value", [float("inf"), -float("inf"), float("nan")])
def test_rejects_nonfinite_coordinates(tmp_path, bad_value):
    file = tmp_path / "embedding.csv"
    pd.DataFrame({"host_id": ["a", "b"],
                  "emb_0": [1.0, bad_value]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="NaN|finite"):
        validate_submission(str(file))

@pytest.mark.parametrize("value", ["inf", "-inf", "garbage"])
def test_common_evaluator_refuses_invalid_numbers(tmp_path, value):
    file = tmp_path / "embeddings.csv"
    pd.DataFrame({"host_id": ["a", "b"], "emb_0": ["1.2", value]}).to_csv(file, index=False)
    with pytest.raises(ValueError, match="finite|numeric"):
        _load_matrix(file, "host_id", "emb_")
