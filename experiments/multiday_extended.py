"""Последовательный большой прогон: train02–07 → validation08 → test09."""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import psutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--work", type=Path, required=True)
    args = ap.parse_args()
    work = args.work.resolve()
    work.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    source = Path("/Volumes/KINGSTON/PT_PROJECT_ML")
    base = source / "multiday/processed"
    experiment = work / "extended"
    experiment.mkdir(exist_ok=True)
    import fcntl
    lock = (work / "extended.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    child = None
    preparations = {}

    def state(stage, **extra):
        row = {"stage": stage, "at": datetime.now().astimezone().isoformat(), **extra}
        (work / "extended_state.json").write_text(json.dumps(row, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(row, ensure_ascii=False), flush=True)

    def stop(signum, frame):
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signum)
        for process, log in preparations.values():
            if process.poll() is None:
                os.killpg(process.pid, signum)
        state("stopped", signal=signum)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    threads = str(min(6, psutil.cpu_count() or 1))
    env = dict(os.environ, PT_COMPACT="1", PYTHONUNBUFFERED="1", OMP_NUM_THREADS=threads,
               OPENBLAS_NUM_THREADS=threads, MKL_NUM_THREADS=threads, VECLIB_MAXIMUM_THREADS=threads,
               PT_TORCH_THREADS="4",
               PT_VIEW_CACHE_DIR=str(work / "view_cache"))
    resources = {"cpu_threads": int(threads), "torch_threads": 4, "preparation_workers": 2,
                 "requested_by_user": True, "at": datetime.now().astimezone().isoformat()}
    (work / "runtime_resources.json").write_text(json.dumps(resources, indent=2) + "\n")

    def run_stage(name, module, *flags):
        nonlocal child
        if psutil.virtual_memory().available < 3 * 2**30:
            raise RuntimeError(f"Перед {name} свободно меньше 3 ГБ памяти")
        if psutil.disk_usage(work).free < 20 * 2**30:
            raise RuntimeError("На SSD осталось меньше 20 ГБ")
        command = ["caffeinate", "-i", "nice", "-n", "5", sys.executable, "-m", module, *map(str, flags)]
        with (work / f"extended_{name}.log").open("a") as log:
            child = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT,
                                     stdin=subprocess.DEVNULL, start_new_session=True)
            state(name, pid=child.pid)
            while True:
                try:
                    code = child.wait(timeout=25 * 60)
                    break
                except subprocess.TimeoutExpired:
                    state(name, pid=child.pid, memory_available_gb=round(psutil.virtual_memory().available / 2**30, 2))
        (work / f"extended_{name}.exit_code").write_text(str(code) + "\n")
        if code:
            state("stage_failed", failed_stage=name, exit_code=code)
            raise RuntimeError(f"Стадия {name}: exit code {code}")

    def pilot_ready():
        previous = json.loads((work / "queue_state.json").read_text())
        if previous["stage"] == "ready_for_frozen_selection":
            return True
        if previous["stage"] in ("pilot_failed", "stage_failed", "low_memory_before_stage"):
            raise RuntimeError("Предыдущая очередь остановилась: " + json.dumps(previous))
        pid = int((work / "queue.pid").read_text())
        if not psutil.pid_exists(pid):
            raise RuntimeError("Предыдущая очередь исчезла без итогового состояния")
        return False
    state("waiting_for_downloads")
    while not (work / "download_extended.exit_code").exists():
        if not psutil.pid_exists(int((work / "download_extended.pid").read_text())):
            raise RuntimeError("Загрузка новых дней остановилась")
        time.sleep(30)
    # День05 готовит пилотная очередь; остальные можно обрабатывать уже сейчас.
    pending = [day for day in [4, 6, 7, 8] if not (work / f"day{day:02d}/manifest.json").exists()]
    last_active = None
    parallel_disabled = False
    try:
        while pending or preparations:
            if len(preparations) > 1 and psutil.virtual_memory().available < 1.5 * 2**30:
                day = max(preparations)
                process, log = preparations.pop(day)
                if process.poll() is None:
                    try:
                        os.killpg(process.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    process.wait()
                log.close()
                pending.insert(0, day)
                parallel_disabled = True
                state("prepare_days_memory_fallback", deferred_day=day)
            worker_limit = 1 if parallel_disabled or not pilot_ready() else 2
            while pending and len(preparations) < worker_limit:
                reserve = 0
                for process, _ in preparations.values():
                    if process.poll() is None:
                        try:
                            reserve += max(0, 2.5 * 2**30 - psutil.Process(process.pid).memory_info().rss)
                        except psutil.NoSuchProcess:
                            pass
                available = psutil.virtual_memory().available - reserve
                minimum = (4.5 if preparations else 3) * 2**30
                if available < minimum:
                    if not preparations:
                        raise RuntimeError("Для подготовки данных свободно меньше 3 ГБ памяти")
                    break
                day = pending.pop(0)
                log = (work / f"extended_prepare_day{day:02d}.log").open("a")
                worker_env = dict(env, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1",
                                  MKL_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
                process = subprocess.Popen([
                    "caffeinate", "-i", "nice", "-n", "5", sys.executable, "-m", "experiments.multiday_prepare",
                    "--raw", str(source / f"multiday/raw/netflow_day-{day:02d}.bz2"), "--base", str(base),
                    "--out", str(work / f"day{day:02d}"), "--day", str(day),
                ], cwd=root, env=worker_env, stdout=log, stderr=subprocess.STDOUT,
                   stdin=subprocess.DEVNULL, start_new_session=True)
                preparations[day] = (process, log)
            active = sorted(preparations)
            if active != last_active:
                state("prepare_days", active_days=active, pending_days=pending,
                      workers={str(d): p.pid for d, (p, _) in preparations.items()})
                last_active = active
            for day, (process, log) in list(preparations.items()):
                code = process.poll()
                if code is not None:
                    log.close()
                    del preparations[day]
                    (work / f"extended_prepare_day{day:02d}.exit_code").write_text(str(code) + "\n")
                    if code:
                        raise RuntimeError(f"Подготовка дня {day}: exit code {code}")
            time.sleep(2)
    finally:
        for process, log in preparations.values():
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGTERM)
                process.wait()
            log.close()
    if not pilot_ready():
        state("waiting_for_pilot_queue")
        while not pilot_ready():
            time.sleep(30)
    if not (work / "day05/manifest.json").exists():
        run_stage("prepare_day05", "experiments.multiday_prepare", "--raw",
                  source / "multiday/raw/netflow_day-05.bz2", "--base", base,
                  "--out", work / "day05", "--day", 5)
    paired_val, refs_val = work / "paired_validation", work / "reference_validation"
    run_stage("reference_validation", "experiments.multiday_reference", "--base", base,
              "--target", work / "day08", "--out", paired_val, "--reference", refs_val, "--window", "validation")
    extra = [source / "multiday2d/day03", *[work / f"day{day:02d}" for day in [4, 5, 6, 7]]]
    candidates = []
    for family in ["gain", "roles", "posterior", "dc"]:
        output = experiment / family
        candidates.append(output)
        run_stage(family, f"experiments.multiday_{family}", "--base", paired_val, "--extra", *extra,
                  "--out", output, "--validation-window", "validation", "--reference-dir", refs_val / "validation")
    # Победитель фиксируется ДО чтения финального дня.
    from experiments.multiday_finish import freeze_selection
    final = experiment / "final"
    frozen = freeze_selection(candidates, final)
    state("winner_frozen", winner=frozen["winner"]["name"], validation=frozen["winner"]["score"])
    run_stage("prepare_day09", "experiments.multiday_prepare", "--raw",
              source / "multiday/raw/netflow_day-09.bz2", "--base", base, "--out", work / "day09", "--day", 9)
    paired_test, refs_test = work / "paired_test", work / "reference_test"
    run_stage("reference_test", "experiments.multiday_reference", "--base", base,
              "--target", work / "day09", "--out", paired_test, "--reference", refs_test, "--window", "test")
    run_stage("final", "experiments.multiday_finish", "--candidates", *candidates,
              "--data", paired_test, "--reference", refs_test / "test", "--out", final)
    summary = json.loads((final / "summary.json").read_text())
    state("complete", report=str(final / "REPORT.md"), **summary)
    (work / "extended.exit_code").write_text("0\n")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        if "--work" in sys.argv:
            location = Path(sys.argv[sys.argv.index("--work") + 1])
            location.mkdir(parents=True, exist_ok=True)
            (location / "extended_state.json").write_text(json.dumps({
                "stage": "failed", "at": datetime.now().astimezone().isoformat(),
                "error": f"{type(error).__name__}: {error}",
            }, ensure_ascii=False, indent=2) + "\n")
        raise
