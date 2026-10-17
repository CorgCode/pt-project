"""Перенести итоги прогона на LANL в репозиторий — без данных и эмбеддингов.

    python experiments/lanl/publish.py /путь/к/lanl results/lanl_day02            # итоги прогона
    python experiments/lanl/publish.py /путь/к/lanl --scoreboard                  # общая таблица на этих данных

/путь/к/lanl — папка с processed/, reference/ и run/. Результат открывается дашбордом
(results/<имя> находится сам) и читается таблицей: run/, reference/<окно>/manifest.json, processed_manifest.json.
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

RUN_FILES = ["report.md", "scoreboard_test.md", "scoreboard_test.csv", "scoreboard_validation.md",
             "scoreboard_validation.csv", "audit.json", "manifest.json"]
RUN_DIRS = ["figures", "results", "identity"]


def publish(src: Path, out: Path) -> None:
    src, out = Path(src), Path(out)
    run = src / "run"
    if not (run / "manifest.json").exists():
        raise SystemExit(f"{run}: нет manifest.json — прогон не закончен")
    if out.exists():
        shutil.rmtree(out)
    (out / "run").mkdir(parents=True)
    for name in RUN_FILES:
        if (run / name).exists():
            shutil.copy(run / name, out / "run" / name)
    for sub in RUN_DIRS:
        if (run / sub).exists():
            shutil.copytree(run / sub, out / "run" / sub)
    for t in ("validation", "test"):
        ref = src / "reference" / t
        if ref.exists():
            (out / "reference" / t).mkdir(parents=True)
            for name in ("manifest.json", "eval_hosts.csv"):
                shutil.copy(ref / name, out / "reference" / t / name)
    shutil.copy(src / "processed" / "manifest.json", out / "processed_manifest.json")
    for p in list(out.rglob("._*")):                     # служебные файлы macOS на exFAT
        p.unlink()
    for p in out.rglob("*"):
        if p.suffix in {".json", ".md", ".csv"}:
            p.write_text(p.read_text(encoding="utf-8").replace(str(src), "<workdir>"), encoding="utf-8")
    print(f"готово: {out} ({sum(1 for f in out.rglob('*') if f.is_file())} файлов)")


def to_scoreboard(src: Path, board: Path = Path("scoreboard")) -> None:
    """Перевести общую таблицу на данные прогона: эталоны, манифест и когорта, эмбеддинги всех методов."""
    import json

    import pandas as pd

    src, board = Path(src), Path(board)
    data = board / "data"
    if data.exists():
        shutil.rmtree(data)
    (data / "processed").mkdir(parents=True)
    for name in ("manifest.json", "cohort.csv"):                      # потоки не копируем: они в PT_DATA
        shutil.copy(src / "processed" / name, data / "processed" / name)
    for t in ("validation", "test"):
        shutil.copytree(src / "reference" / t, data / "reference" / t, ignore=shutil.ignore_patterns("._*"))
    subs = board / "submissions"
    if subs.exists():
        shutil.rmtree(subs)
    for d in sorted(p for p in (src / "run" / "embeddings").iterdir() if p.is_dir() and not p.name.startswith(".")):
        out = subs / d.name
        out.mkdir(parents=True)
        for window, name in (("train", "embeddings.csv"), ("test", "test.csv")):
            if (d / f"{window}.csv").exists():
                pd.read_csv(d / f"{window}.csv", dtype={"host_id": str}).to_csv(out / name, index=False, float_format="%.7g")
        info = d / "info.json"
        if info.exists() and json.loads(info.read_text(encoding="utf-8")).get("shared_coordinate_system"):
            (out / "SHARED_COORDINATES").touch()
    f = pd.read_csv(data / "reference" / "test" / "features.csv", dtype={"host_id": str}).set_index("host_id")
    f.columns = [c.replace("ref_", "emb_") for c in f.columns]
    (subs / "reference_ceiling").mkdir(exist_ok=True)
    f.to_csv(subs / "reference_ceiling" / "embeddings.csv", float_format="%.7g")
    print(f"scoreboard/: данные из {src}; отправок: {sum(1 for _ in subs.iterdir())}. "
          "Дальше: python -m evaluation.submit --all --force")


if __name__ == "__main__":
    if sys.argv[2] == "--scoreboard":
        to_scoreboard(Path(sys.argv[1]))
    else:
        publish(Path(sys.argv[1]), Path(sys.argv[2]))
