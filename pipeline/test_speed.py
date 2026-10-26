from pipeline import speed
from pipeline.data import ProcessedData


def _square(x):
    return x * x


def test_run_parallel_returns_every_item_in_processes():
    got = {it: res for it, res, _ in speed.run_parallel(_square, [1, 2, 3, 4], jobs=2, desc="тест")}
    assert got == {1: 1, 2: 4, 3: 9, 4: 16}


def test_jobs_respect_env(monkeypatch):
    monkeypatch.setenv("PT_JOBS", "3")
    assert speed.default_jobs(gb_per_job=100) == 3
    monkeypatch.delenv("PT_JOBS")
    assert speed.default_jobs(gb_per_job=1e6) == 1                     # памяти не хватит — один процесс


def test_flows_cache_is_used_and_refreshed(processed, tmp_path):
    import shutil
    root = tmp_path / "p"
    shutil.copytree(processed, root)
    d = ProcessedData(root)
    first = d.flows("train")
    assert (root / "flows" / ".cache" / "train.pkl").exists()
    assert d.flows("train").equals(first)
    csv = root / "flows" / "train.csv"
    csv.write_text(csv.read_text().splitlines()[0] + "\n" + "\n".join(csv.read_text().splitlines()[1:6]) + "\n")
    assert len(d.flows("train")) == 5                                   # CSV поменялся — копия пересоздана
