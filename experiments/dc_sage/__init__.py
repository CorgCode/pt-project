"""Сохраняет импорт experiments.dc_sage после переноса кода в DC-SAGE."""
from pathlib import Path

__path__.append(str(Path(__file__).resolve().parents[1] / "DC-SAGE" / "dc_sage"))
