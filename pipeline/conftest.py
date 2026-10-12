import json
from pathlib import Path

import pytest

from pipeline import fixtures
from pipeline.prepare import prepare
from pipeline.reference import build_reference
from pipeline.windows import PipelineConfig

CONFIG = PipelineConfig(windows={"train": (0, 21600), "validation": (21600, 32400),
                                 "test": (32400, 43200)})


@pytest.fixture(scope="session")
def raw_fixture(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("raw")
    fixtures.generate(d, n_hosts=80, seed=3)
    return d


@pytest.fixture(scope="session")
def processed(raw_fixture, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("processed")
    prepare(sorted(raw_fixture.glob("netflow_day-*.bz2")), CONFIG, out)
    return out


@pytest.fixture(scope="session")
def reference(processed, tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("reference")
    build_reference(processed, out)
    return out
