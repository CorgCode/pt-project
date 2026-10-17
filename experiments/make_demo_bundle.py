"""Пересобрать demo/synthetic_run — маленький СИНТЕТИЧЕСКИЙ пример прогона для дашборда и сайта.

    python -m experiments.make_demo_bundle --output demo/synthetic_run

Прогоняет синтетику целиком (seed фиксирован) и копирует отчёт, графики, результаты, таблицы, аудит,
манифест, эмбеддинги и эталонных соседей test-окна, убрав абсолютные пути. Всё помечено `synthetic: true`:
это НЕ данные LANL и не результат бенчмарка.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from experiments.end_to_end import end_to_end
from experiments.feature_ablation import run_ablation

BANNER = """# Синтетический пример — НЕ данные LANL

Собран командой `python -m experiments.make_demo_bundle` из синтетики в формате LANL NetFlow
(`pipeline/fixtures.py`, seed 0, 300 хостов). Нужен, чтобы дашборд и сайт показывали полный прогон;
каждый файл помечен `synthetic: true`. Числа ничего не говорят о реальной сети, цитировать их как
результат бенчмарка нельзя.
"""


def _clean(obj, prefix: str):
    if isinstance(obj, dict):
        return {k: _clean(v, prefix) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v, prefix) for v in obj]
    if isinstance(obj, str) and prefix in obj:
        return obj.replace(prefix, "<workdir>")
    return obj


def build(output: Path, *, n_hosts: int = 300, seed: int = 0, n_boot: int = 200,
          methods: list[str] | None = None, fast_methods: bool = False) -> Path:
    output = Path(output)
    with tempfile.TemporaryDirectory() as tmp:
        work = Path(tmp).resolve()
        end_to_end(work, fixture=True, n_hosts=n_hosts, seed=seed, n_boot=n_boot, pairs_per_anchor=10,
                   methods=methods, fast_methods=fast_methods)
        run_ablation(work / "processed", work / "reference", work / "ablation", n_boot=n_boot, seed=seed)
        if output.exists():
            shutil.rmtree(output)
        run, ref = output / "run", output / "reference"
        shutil.copytree(work / "run", run)
        for t in ("validation", "test"):
            (ref / t).mkdir(parents=True)
            shutil.copy(work / "reference" / t / "manifest.json", ref / t / "manifest.json")
        shutil.copy(work / "reference" / "test" / "neighbors.csv", ref / "test" / "neighbors.csv")
        for t in ("validation", "test"):
            shutil.copy(work / "reference" / t / "eval_hosts.csv", ref / t / "eval_hosts.csv")
        shutil.copy(work / "processed" / "manifest.json", output / "processed_manifest.json")
        (output / "ablation").mkdir()
        for name in ("ablation.json", "ablation.md"):                       # абляция без портов (#43)
            shutil.copy(work / "ablation" / name, output / "ablation" / name)
        for path in output.rglob("[!.]*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps(_clean(data, str(work)), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        for path in output.rglob("*.md"):
            path.write_text(path.read_text(encoding="utf-8").replace(str(work), "<workdir>"), encoding="utf-8")
        (output / "README.md").write_text(BANNER, encoding="utf-8")
    return output


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=Path("demo/synthetic_run"))
    ap.add_argument("--n-hosts", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", nargs="+", default=["all"], help="графовые методы (по умолчанию все пять)")
    a = ap.parse_args()
    print(build(a.output, n_hosts=a.n_hosts, seed=a.seed, methods=a.methods))


if __name__ == "__main__":
    main()
