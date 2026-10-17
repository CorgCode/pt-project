"""Перенести результаты прогона на LANL в репозиторий (только итоги, без данных и эмбеддингов).

    python experiments/lanl/publish.py /путь/к/lanl/run results/lanl_day02

Копируются таблицы результатов, отчёт, графики, аудит, манифест и JSON по методам.
Сырые логи, потоки и эмбеддинги в git не попадают: их можно получить заново по README.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

KEEP = ["report.md", "scoreboard_test.md", "scoreboard_test.csv", "scoreboard_validation.md",
        "scoreboard_validation.csv", "audit.json", "manifest.json"]


def publish(run: Path, out: Path) -> None:
    run, out = Path(run), Path(out)
    if not (run / "manifest.json").exists():
        raise SystemExit(f"{run}: нет manifest.json — прогон не закончен")
    out.mkdir(parents=True, exist_ok=True)
    for name in KEEP:
        if (run / name).exists():
            shutil.copy(run / name, out / name)
    for sub in ("figures", "results", "identity"):
        if (run / sub).exists():
            shutil.copytree(run / sub, out / sub, dirs_exist_ok=True)
    for p in out.rglob("._*"):                       # служебные файлы macOS на exFAT
        p.unlink()
    for p in out.rglob("*"):
        if p.suffix in {".json", ".md", ".csv"}:
            p.write_text(p.read_text(encoding="utf-8").replace(str(run.parent), "<workdir>"), encoding="utf-8")
    print(f"готово: {out} ({sum(1 for _ in out.rglob('*') if _.is_file())} файлов)")


if __name__ == "__main__":
    publish(Path(sys.argv[1]), Path(sys.argv[2]))
