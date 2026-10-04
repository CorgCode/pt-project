import pandas as pd
import pytest

from app import core


@pytest.fixture(scope="module")
def reference_dir(tmp_path_factory):
    return core.make_demo_reference(tmp_path_factory.mktemp("ref"), n_hosts=100)


@pytest.fixture(scope="module")
def sample(reference_dir, tmp_path_factory):
    return core.sample_embedding(reference_dir, tmp_path_factory.mktemp("s"), dim=6)


def test_full_flow_on_valid_upload(sample, reference_dir):
    out = core.evaluate_upload(sample, reference_dir)
    board = out["board"]
    assert set(board.index) == {"uploaded", "random", "pca", "svd", "reference ceiling"}
    assert {f"{m}@{k}" for m in core.METRICS for k in core.KS} == set(board.columns)
    assert board.loc["reference ceiling", "consistency@10"] >= board.loc["random", "consistency@10"]
    assert board.loc["uploaded", "consistency@10"] > board.loc["random", "consistency@10"]
    assert out["projection"] is not None and {"host_id", "PC1", "PC2"} <= set(out["projection"].columns)
    assert any(u.startswith("triplet_accuracy") for u in out["unavailable"])


def test_metric_long_shape(sample, reference_dir):
    long = core.metric_long(core.evaluate_upload(sample, reference_dir)["board"], "jaccard")
    assert list(long.columns) == ["method", "k", "Jaccard"] and len(long) == 5 * 3


@pytest.mark.parametrize("data,msg", [
    (b"", "empty"),
    (b"a,b\n1,2\n", "host_id"),
    (b"host_id,x\nh1,1\n", "No embedding columns"),
    (b"host_id,emb_0\nh1,1\nh1,2\n", "Duplicate host_id"),
    (b"host_id,emb_0\nh1,\nh2,2\n", "NaN"),
    (b"host_id,emb_0\nh1,abc\nh2,2\n", "numeric"),
])
def test_malformed_files_give_clear_errors(data, msg, reference_dir, tmp_path):
    with pytest.raises(core.UploadError, match=msg):
        core.validate_upload(data, tmp_path)


def test_binary_garbage_is_an_upload_error(tmp_path):
    with pytest.raises(core.UploadError):
        core.validate_upload(b"\xff\xfe\x00\x01\x02", tmp_path)


def test_wrong_host_ids_rejected(sample, reference_dir):
    df = pd.read_csv(__import__("io").BytesIO(sample)).iloc[:-5]
    with pytest.raises(core.UploadError, match="do not match the demo reference"):
        core.evaluate_upload(df.to_csv(index=False).encode(), reference_dir)


def test_single_dimension_has_no_projection(sample, reference_dir):
    df = pd.read_csv(__import__("io").BytesIO(sample))[["host_id", "emb_0"]]
    out = core.evaluate_upload(df.to_csv(index=False).encode(), reference_dir)
    assert out["projection"] is None


def test_streamlit_app_smoke():
    pytest.importorskip("streamlit")
    pytest.importorskip("plotly")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file("streamlit_app.py", default_timeout=60).run()
    assert not at.exception
    assert at.title[0].value == "Host embedding benchmark"
