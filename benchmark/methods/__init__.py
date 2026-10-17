"""Графовые методы, находятся автоматически.

Любой модуль пакета может объявить METHODS = {"имя": Класс} — центральный список править не нужно.
Методы следуют временному протоколу (обучение только на train) и выбираются флагом --methods.
"""
from __future__ import annotations

import importlib
import pkgutil

_SKIP = {"base", "graph_input", "walks", "sgns"}


def available() -> dict[str, type]:
    found: dict[str, type] = {}
    for mod in pkgutil.iter_modules(__path__):
        if mod.name in _SKIP or mod.name.startswith("test_") or mod.name.startswith("_"):
            continue
        found.update(getattr(importlib.import_module(f"{__name__}.{mod.name}"), "METHODS", {}))
    return dict(sorted(found.items()))


def create(names: list[str], dim: int = 16, seed: int = 0, config: dict | None = None, fast: bool = False) -> list:
    """Создать методы по именам ("all" — все зарегистрированные).

    config: {метод: переопределения гиперпараметров}. fast=True сначала ставит крошечные fast_params
    (только для CI — числа бессмысленны), затем переопределения.
    """
    reg, config = available(), config or {}
    names = list(reg) if list(names) == ["all"] else list(names)
    unknown = [n for n in names if n not in reg]
    if unknown:
        raise ValueError(f"неизвестные методы {unknown}; есть: {sorted(reg)}")
    stray = set(config) - set(names)
    if stray:
        raise ValueError(f"параметры заданы для невыбранных методов: {sorted(stray)}")
    return [reg[n](dim=dim, seed=seed, **{**(reg[n].fast_params if fast else {}), **config.get(n, {})}) for n in names]
