"""Общие фикстуры тестов MD-TGR: срезы, собранные из синтетического ``processed``."""
from __future__ import annotations

import pytest

from methods.mdtgr.dataset import ShardSet
from methods.mdtgr.shards import build_shards


@pytest.fixture(scope="session")
def mdtgr_shards_dir(processed, tmp_path_factory):
    out = tmp_path_factory.mktemp("mdtgr_shards")
    build_shards(processed, out)
    return out


@pytest.fixture(scope="session")
def mdtgr_shards(mdtgr_shards_dir):
    return ShardSet(mdtgr_shards_dir)
