import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scoreboard"))
import combine  # noqa: E402


def _emb(path, seed):
    hosts = pd.Index([f"h{i}" for i in range(5)], name="host_id")
    pd.DataFrame(np.random.default_rng(seed).normal(size=(5, 3)), index=hosts).add_prefix("emb_").to_csv(path)


def test_rebuild_drops_stale_outputs_and_reports_gaps(tmp_path, monkeypatch, capsys):
    subs = tmp_path / "subs"
    for i, p in enumerate(("a", "b")):
        (subs / p).mkdir(parents=True)
        _emb(subs / p / "embeddings.csv", i)
    _emb(subs / "a" / "test.csv", 5)                               # test есть только у одной части
    out = subs / "ab"
    out.mkdir()
    _emb(out / "test.csv", 9)                                      # остался от прошлой сборки
    monkeypatch.setattr(combine.submit, "SUBMISSIONS", subs)
    combine.combine(["a", "b"], "ab")
    assert (out / "embeddings.csv").exists() and not (out / "test.csv").exists()
    assert "test.csv нет у ['b']" in capsys.readouterr().err
    (subs / "b" / "embeddings.csv").unlink()
    with pytest.raises(FileNotFoundError):
        combine.combine(["a", "b"], "ab")
