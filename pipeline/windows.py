"""Temporal windows and the leakage rules they enforce.

Windows are half-open intervals [start, end) on the dataset's own Time axis (same units
as the `Time` column; the LANL page does not define a calendar origin). Required:

    train < validation (optional) < test        -- strictly ordered, non-overlapping

Cohort policy (which hosts are evaluated):
- train_only (default): the cohort is every host active in the TRAIN window; a host is
  *evaluable* in a later window only if it is also active there. Metrics are computed on the
  evaluable hosts and the coverage is reported, so no future information decides who is in.
- all_windows: the cohort is the hosts active in EVERY window. Future activity influences
  inclusion (a transductive "stable-host" benchmark): scores condition on host survival and
  overstate persistence/identity/link-persistence for prospective use. Kept for comparison only.

Rules:
- train is the ONLY window models may fit on; validation is for tuning choices;
  test is for final reporting. Reference artifacts are built from later windows.
- A flow is assigned by START time, and (drop_straddling=True) dropped if it is still
  running at the window end (Time + Duration >= end), so no flow carries information
  across a boundary.
- Windows must be declared BEFORE looking at evaluation scores (see docs).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ORDER = ("train", "validation", "test")
COHORT_POLICIES = ("train_only", "all_windows")


@dataclass(frozen=True)
class PipelineConfig:
    windows: dict[str, tuple[float, float]]
    min_events_per_window: int = 5
    min_cohort_size: int = 25
    drop_straddling: bool = True
    seed: int = 0
    cohort_policy: str = "train_only"

    def __post_init__(self) -> None:
        names = set(self.windows)
        if not names <= set(ORDER):
            raise ValueError(f"Unknown window names {sorted(names - set(ORDER))}; use {ORDER}")
        if not {"train", "test"} <= names:
            raise ValueError("Windows 'train' and 'test' are required")
        previous_end = None
        for name in (n for n in ORDER if n in names):
            start, end = self.windows[name]
            if not (np.isfinite(start) and np.isfinite(end) and start < end):
                raise ValueError(f"Window {name}: need finite start < end, got {start}, {end}")
            if previous_end is not None and start < previous_end:
                raise ValueError(f"Window {name} overlaps or precedes the previous window")
            previous_end = end
        if self.cohort_policy not in COHORT_POLICIES:
            raise ValueError(f"cohort_policy must be one of {COHORT_POLICIES}")
        if self.min_events_per_window < 1 or self.min_cohort_size < 22:
            raise ValueError("min_events_per_window >= 1 and min_cohort_size >= 22 "
                             "(k=20 needs more than 21 hosts)")

    @property
    def ordered(self) -> list[str]:
        return [n for n in ORDER if n in self.windows]

    def to_dict(self) -> dict:
        return {"windows": {k: list(v) for k, v in self.windows.items()},
                "min_events_per_window": self.min_events_per_window,
                "min_cohort_size": self.min_cohort_size,
                "drop_straddling": self.drop_straddling, "seed": self.seed,
                "cohort_policy": self.cohort_policy}

    @classmethod
    def from_json(cls, path: Path) -> "PipelineConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        try:
            windows = {k: (float(v[0]), float(v[1])) for k, v in raw["windows"].items()}
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise ValueError('config needs {"windows": {"train": [start, end], ...}}') from exc
        extra = {k: raw[k] for k in ("min_events_per_window", "min_cohort_size",
                                     "drop_straddling", "seed", "cohort_policy") if k in raw}
        return cls(windows=windows, **extra)


def window_mask(time: np.ndarray, duration: np.ndarray, window: tuple[float, float],
                drop_straddling: bool) -> np.ndarray:
    start, end = window
    mask = (time >= start) & (time < end)
    if drop_straddling:
        mask &= (time + duration) < end
    return mask
