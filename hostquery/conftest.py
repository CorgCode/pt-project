"""Общие данные тестов: закоммиченный синтетический прогон."""
from pathlib import Path

import pytest

from hostquery.store import Dataset, RunStore

DEMO = Path(__file__).resolve().parents[1] / "app" / "testdata" / "synthetic_run"


@pytest.fixture(scope="session")
def demo_path() -> Path:
    return DEMO


@pytest.fixture(scope="session")
def demo() -> Dataset:
    return RunStore(bundle=DEMO).dataset("behavioral")
