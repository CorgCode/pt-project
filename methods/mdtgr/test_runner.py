"""Раннер scoreboard: выбор эпизодов по времени окна, а не по имени."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import pytest

from methods.mdtgr.train import OFFICIAL_TEST_START
from scoreboard.mdtgr import plan_windows, total_weights


@dataclass
class FakeShards:
    root: str = "fake"
    windows: list[str] = field(default_factory=list)
    manifest: dict = field(default_factory=dict)


def _times(**hours: float) -> dict:
    """Окна по времени: имя -> [start, start+3600]."""
    return {name: [value, value + 3600.0] for name, value in hours.items()}


def test_plan_splits_days_dev_and_official_test():
    shards = FakeShards(windows=["train", "validation", "test_day02", "test_day03",
                                 "test_day04", "test", "test_day15", "test_day16"],
                        manifest={"window_times": _times(
                            train=126000, validation=129600, test_day02=131400,
                            test_day03=217800, test_day04=304200,
                            test=OFFICIAL_TEST_START,
                            test_day15=OFFICIAL_TEST_START + 86400,
                            test_day16=OFFICIAL_TEST_START + 2 * 86400)})
    episodes, dev, late = plan_windows(shards)
    assert episodes == ["test_day02", "test_day03"]
    assert dev == "test_day04"
    assert late == ["test", "test_day15", "test_day16"]


def test_plan_never_learns_on_official_test_even_if_named_like_a_day():
    """Имя не решает: окно с временем официального test в эпизоды не попадает, а dev — последний день."""
    shards = FakeShards(windows=["train", "validation", "test_day12", "test_day13", "test"],
                        manifest={"window_times": _times(
                            train=126000, validation=129600,
                            test_day12=OFFICIAL_TEST_START - 2 * 86400,
                            test_day13=OFFICIAL_TEST_START - 86400,
                            test=OFFICIAL_TEST_START)})
    episodes, dev, late = plan_windows(shards)
    assert episodes == ["test_day12"]
    assert dev == "test_day13"
    assert late == ["test"]

    with pytest.raises(ValueError, match="хотя бы два дня"):
        plan_windows(FakeShards(windows=["train", "validation", "test"],
                                manifest={"window_times": _times(
                                    train=126000, validation=129600, test=OFFICIAL_TEST_START)}))


def test_plan_requires_window_times():
    with pytest.raises(ValueError, match="window_times"):
        plan_windows(FakeShards(windows=["train", "test"], manifest={}))


def test_total_weights_reads_tuned_values(tmp_path):
    assert total_weights(tmp_path / "missing.json") == (1.0, 1.0, 1.0)
    path = tmp_path / "weights.json"
    path.write_text(json.dumps({"best": {"weights": [0.5, 2.0, 1.5], "score": 42.0}}),
                    encoding="utf-8")
    assert total_weights(path) == (0.5, 2.0, 1.5)
