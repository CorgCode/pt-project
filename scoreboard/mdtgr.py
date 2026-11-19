"""MD-TGR на scoreboard: обучение на днях 02–12, dev — день 13, оценка — дни 14–16.

    PT_DATA=/путь/к/processed python scoreboard/mdtgr.py

``PT_DATA`` — каталог ``pipeline.prepare`` с многосуточными окнами: ``train`` (день 02),
``validation`` (день 02, позднее окно), окна дней 03–13 и официальные test-дни 14–16.
Срезы окон кэшируются в ``submissions/mdtgr/.build/shards``; модель учится один раз и
применяется ко всем окнам, поэтому ``SHARED_COORDINATES`` выставлен.

Дни 14–16 не участвуют ни в обучении, ни в подборе весов: эпизоды и dev выбираются по
времени окна, а не по имени (``OFFICIAL_TEST_START``).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from evaluation import submit  # noqa: E402
from methods.mdtgr.dataset import ShardSet  # noqa: E402
from methods.mdtgr.infer import export, load_checkpoint, tune_weights  # noqa: E402
from methods.mdtgr.shards import build_shards  # noqa: E402
from methods.mdtgr.train import OFFICIAL_TEST_START, TrainConfig, train  # noqa: E402

NAME = "mdtgr"
AUTHOR = "kalivixZ"
DIM = 96
CODE_DIM = 64
EPOCHS = 40
ANCHORS = 1024          # локально проверено на 1024; на слабой машине можно снизить флагом
SEED = 0


def plan_windows(shards: ShardSet) -> tuple[list[str], str | None, list[str]]:
    """Разделить окна по времени: эпизоды, dev и официальные test-дни.

    Эпизоды — все дни до официального test, кроме ``train`` (источник M) и
    ``validation`` (на нём подбираются веса). Последний такой день — dev: на нём
    выбирается чекпойнт, в обучение он не идёт.
    """
    times = shards.manifest.get("window_times", {})
    if not times:
        raise ValueError(f"{shards.root}: в манифесте нет window_times — по именам окна не различить")

    late = [w for w in shards.windows if times[w][0] >= OFFICIAL_TEST_START]
    days = [w for w in shards.windows
            if w not in ("train", "validation") and times[w][0] < OFFICIAL_TEST_START]
    if len(days) < 2:
        raise ValueError(f"нужны хотя бы два дня до официального test (обучение и dev), есть {days}")
    dev = days[-1]                                   # последний день — контроль, не обучение
    episodes = days[:-1]
    return episodes, dev, late


def total_weights(path: Path) -> tuple[float, float, float]:
    """Веса блоков, подобранные на validation (если подбор не делали — равные)."""
    if not path.exists():
        return (1.0, 1.0, 1.0)
    best = json.loads(path.read_text(encoding="utf-8"))["best"]["weights"]
    return tuple(float(x) for x in best)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="MD-TGR: обучение и отправка в scoreboard")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--anchors", type=int, default=ANCHORS, help="якорей на шаг (память/скорость)")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--rebuild-shards", action="store_true")
    ap.add_argument("--no-tune", action="store_true", help="не перебирать веса блоков на validation")
    ap.add_argument("--no-score", action="store_true", help="только обучить, не оценивать")
    args = ap.parse_args(argv)

    processed = Path(os.environ.get("PT_DATA", ROOT / "scoreboard" / "data" / "processed"))
    out = ROOT / "scoreboard" / "submissions" / NAME
    out.mkdir(parents=True, exist_ok=True)
    shard_dir = out / ".build" / "shards"

    if args.rebuild_shards or not (shard_dir / "manifest.json").exists():
        print(f"срезы: {processed} -> {shard_dir}", flush=True)
        build_shards(processed, shard_dir)
    shards = ShardSet(shard_dir)
    episodes, dev, late = plan_windows(shards)
    print(f"эпизоды: {episodes}\ndev: {dev}\nофициальные test: {late}", flush=True)

    cfg = TrainConfig(episodes=episodes, dev=dev, epochs=args.epochs, seed=args.seed,
                      dim=DIM, code_dim=CODE_DIM, anchors=args.anchors, lr=1e-3, link_weight=0.05,
                      random_weights=True, eval_every=5)
    result = train(shards, cfg, out / ".build" / "run")
    print(f"лучший dev: {result['best']}", flush=True)

    weights = (1.0, 1.0, 1.0)
    if not args.no_tune and "validation" in shards.windows:
        tuned = tune_weights(shards, load_checkpoint(out / ".build" / "run" / "best.pt"), "validation",
                             work_dir=out / ".build" / "tune")
        weights = tuple(tuned["best"]["weights"])
        print(f"веса блоков по validation: {weights} (score {tuned['best']['score']:.2f})", flush=True)

    model = load_checkpoint(out / ".build" / "run" / "best.pt")
    official = [w for w in late if w in submit.test_windows()]
    if not official:
        raise SystemExit("в PT_DATA нет официальных test-окон (дни 14–16): оценивать нечего")
    missing = sorted(set(submit.test_windows()) - set(official))
    if missing:
        print(f"внимание: нет окон {missing} — отправка будет оценена не на всех test-днях", flush=True)
    export(shards, model, out, weights, windows=official + [w for w in ("validation",) if w in shards])
    submit.keep_vectors_local(out)
    (out / ".gitignore").write_text("*.csv\n.build/\n", encoding="utf-8")
    (out / "SHARED_COORDINATES").touch()
    (out / "AUTHOR").write_text(AUTHOR + "\n", encoding="utf-8")
    (out / "params.json").write_text(json.dumps({
        "dim": DIM, "code_dim": CODE_DIM, "epochs": args.epochs, "seed": args.seed,
        "episodes": episodes, "dev": dev, "block_weights": list(weights),
        "trained_through_day": 12, "selected_on": "day 13",
    }, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    card = out / "README.md"
    if not card.exists():
        raise SystemExit(f"эмбеддинги сохранены; заполни карточку {card} и запусти ещё раз")
    if args.no_score:
        return
    r = submit.score(out)
    submit.render()
    print(f"\n{NAME}: " + ", ".join(f"{s['title']} = {s['value']:.3f}"
                                    for s in r["summary"] if s["value"] is not None))


if __name__ == "__main__":
    main()
