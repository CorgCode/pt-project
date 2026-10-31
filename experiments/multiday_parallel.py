"""Два обучения с общими train-признаками и ограничением памяти Mac."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil

EXPECTED = {"gain": 16, "roles": 6, "posterior": 12, "dc": 12}
GIB = 2**30


def trial_count(directory):
    path = directory / "trials.jsonl"
    names = set()
    if path.exists():
        for line in path.read_text().splitlines():
            try:
                names.add(json.loads(line)["name"])
            except (ValueError, KeyError):
                pass
    return len(names)


def snapshot_features(source: Path, destination: Path):
    """Копия завершённого checkpoint; изменение источника во время копирования — повтор позже."""
    if destination.exists():
        return destination
    metadata = source.parent / "best.json"
    if not source.exists() or not metadata.exists():
        return None
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_suffix(".tmp")
    before = source.stat()
    if not before.st_size or metadata.stat().st_mtime_ns < before.st_mtime_ns:
        return None
    record = metadata.read_bytes()
    shutil.copy2(source, temp)
    after = source.stat()
    if (before.st_size, before.st_mtime_ns, record) != (after.st_size, after.st_mtime_ns, metadata.read_bytes()):
        temp.unlink(missing_ok=True)
        return None
    temp.replace(destination)
    return destination


def admission(available, active_rss, shared_features):
    reserve = sum(max(0, 3.5 * GIB - rss) for rss in active_rss)
    expected = 3 * GIB if shared_features else 5 * GIB
    return available - reserve >= expected + 2.5 * GIB


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--adopt-family", choices=list(EXPECTED))
    ap.add_argument("--adopt-pid", type=int)
    args = ap.parse_args()
    work, root = args.work.resolve(), Path(__file__).resolve().parents[1]
    experiment = work / "extended"
    import fcntl
    lock = (work / "extended.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    active, pending, completed = {}, [], set()
    env = dict(os.environ, PT_COMPACT="1", PYTHONUNBUFFERED="1", PT_VIEW_CACHE_DIR=str(work / "view_cache"))
    base = work / "paired_validation"
    extras = [Path("/Volumes/KINGSTON/PT_PROJECT_ML/multiday2d/day03"),
              *[work / f"day{day:02d}" for day in [4, 5, 6, 7]]]
    limit = 2
    resources_path = work / "runtime_resources.json"
    resources = json.loads(resources_path.read_text()) if resources_path.exists() else {}
    resources.update(training_workers=2, total_cpu_budget=8, threads_per_new_job=4,
                     threads_alongside_adopted_job=2, memory_headroom_gb=2.5,
                     mode="parallel_training_with_shared_train_features")
    resources_path.write_text(json.dumps(resources, ensure_ascii=False, indent=2) + "\n")

    def state(stage, **extra):
        row = {"stage": stage, "at": datetime.now().astimezone().isoformat(), **extra}
        (work / "extended_state.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def terminate(record):
        try:
            os.killpg(record["pid"], signal.SIGTERM)
        except ProcessLookupError:
            pass
        if "process" in record:
            record["process"].wait()
        if "log" in record:
            record["log"].close()

    def stop(signum, frame):
        for record in active.values():
            terminate(record)
        state("stopped", signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for family in ["gain", "posterior", "roles", "dc"]:
        marker = work / f"extended_{family}.exit_code"
        if marker.exists() and marker.read_text().strip() == "0" and trial_count(experiment / family) >= EXPECTED[family]:
            completed.add(family)
        elif family == args.adopt_family:
            active[family] = {"pid": args.adopt_pid, "adopted": True, "order": 0,
                              "handle": psutil.Process(args.adopt_pid)}
        else:
            pending.append(family)
    last_state = None
    try:
        while pending or active:
            for family, record in list(active.items()):
                if record.get("adopted"):
                    try:
                        running = record["handle"].is_running() and record["handle"].status() != psutil.STATUS_ZOMBIE
                    except psutil.NoSuchProcess:
                        running = False
                    code = None if running else (0 if trial_count(experiment / family) >= EXPECTED[family] else 1)
                else:
                    code = record["process"].poll()
                if code is not None:
                    if "log" in record:
                        record["log"].close()
                    del active[family]
                    (work / f"extended_{family}.exit_code").write_text(str(code) + "\n")
                    if code:
                        raise RuntimeError(f"Модель {family} завершилась с кодом {code}")
                    completed.add(family)
            if len(active) > 1 and psutil.virtual_memory().available < 2 * GIB:
                family = max(active, key=lambda key: active[key]["order"])
                terminate(active.pop(family))
                retry = experiment / "retries" / f"{family}_{time.time_ns()}"
                retry.parent.mkdir(exist_ok=True)
                (experiment / family).rename(retry)
                pending.insert(0, family)
                limit = 1
                state("training_memory_fallback", deferred_family=family)
            for family in list(pending):
                if len(active) >= limit:
                    break
                shared = None
                if family == "posterior":
                    shared = snapshot_features(experiment / "gain/best.pkl", work / "shared_features/gain.pkl")
                    if shared is None:
                        continue
                elif family == "dc":
                    meta = experiment / "roles/best.json"
                    if not meta.exists() or not json.loads(meta.read_text()).get("has_training_graphs"):
                        continue
                    shared = snapshot_features(experiment / "roles/best.pkl", work / "shared_features/roles.pkl")
                    if shared is None:
                        continue
                rss = []
                for record in active.values():
                    try:
                        rss.append(psutil.Process(record["pid"]).memory_info().rss)
                    except psutil.NoSuchProcess:
                        pass
                if not admission(psutil.virtual_memory().available, rss, shared is not None):
                    continue
                threads = 2 if any(r.get("adopted") for r in active.values()) else 4
                if limit == 1:
                    threads = 4
                worker_env = dict(env, OMP_NUM_THREADS=str(threads), OPENBLAS_NUM_THREADS=str(threads),
                                  MKL_NUM_THREADS=str(threads), VECLIB_MAXIMUM_THREADS=str(threads),
                                  PT_TORCH_THREADS=str(threads))
                output = experiment / family
                command = ["caffeinate", "-i", "nice", "-n", "5", sys.executable, "-m", f"experiments.multiday_{family}",
                           "--base", str(base), "--extra", *map(str, extras), "--out", str(output),
                           "--validation-window", "validation", "--reference-dir", str(work / "reference_validation/validation")]
                if shared:
                    command += ["--features-from", str(shared)]
                log = (work / f"extended_{family}.log").open("a")
                process = subprocess.Popen(command, cwd=root, env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                                           stdin=subprocess.DEVNULL, start_new_session=True)
                active[family] = {"pid": process.pid, "process": process, "log": log,
                                  "order": time.monotonic(), "threads": threads}
                pending.remove(family)
            signature = (tuple(active), tuple(pending), tuple(sorted(completed)))
            if signature != last_state:
                state("parallel_training", active={f: r["pid"] for f, r in active.items()},
                      pending=pending, completed=sorted(completed), max_workers=limit)
                last_state = signature
            time.sleep(2)
    except BaseException:
        for record in active.values():
            terminate(record)
        raise
    from experiments.multiday_finish import freeze_selection
    candidates = [experiment / name for name in EXPECTED]
    final = experiment / "final"
    winner = freeze_selection(candidates, final)
    state("winner_frozen", winner=winner["winner"]["name"], validation=winner["winner"]["score"])
    final_env = dict(env, OMP_NUM_THREADS="6", OPENBLAS_NUM_THREADS="6", MKL_NUM_THREADS="6",
                     VECLIB_MAXIMUM_THREADS="6", PT_TORCH_THREADS="4")
    stages = [
        ("prepare_day09", "experiments.multiday_prepare", ["--raw", "/Volumes/KINGSTON/PT_PROJECT_ML/multiday/raw/netflow_day-09.bz2",
          "--base", "/Volumes/KINGSTON/PT_PROJECT_ML/multiday/processed", "--out", str(work / "day09"), "--day", "9"]),
        ("reference_test", "experiments.multiday_reference", ["--base", "/Volumes/KINGSTON/PT_PROJECT_ML/multiday/processed",
          "--target", str(work / "day09"), "--out", str(work / "paired_test"),
          "--reference", str(work / "reference_test"), "--window", "test"]),
        ("final", "experiments.multiday_finish", ["--candidates", *map(str, candidates), "--data", str(work / "paired_test"),
          "--reference", str(work / "reference_test/test"), "--out", str(final)]),
    ]
    for stage, module, flags in stages:
        with (work / f"extended_{stage}.log").open("a") as log:
            process = subprocess.Popen(["caffeinate", "-i", "nice", "-n", "5", sys.executable, "-m", module, *flags],
                                       cwd=root, env=final_env, stdout=log, stderr=subprocess.STDOUT,
                                       stdin=subprocess.DEVNULL, start_new_session=True)
            active[stage] = {"pid": process.pid, "process": process}
            state(stage, pid=process.pid)
            code = process.wait()
            active.clear()
        if code:
            raise RuntimeError(f"Стадия {stage} завершилась с кодом {code}")
    state("complete", report=str(final / "REPORT.md"), **json.loads((final / "summary.json").read_text()))
    (work / "extended.exit_code").write_text("0\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--work" in sys.argv:
            location = Path(sys.argv[sys.argv.index("--work") + 1])
            (location / "extended_state.json").write_text(json.dumps({
                "stage": "failed", "at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }, ensure_ascii=False, indent=2) + "\n")
        raise
