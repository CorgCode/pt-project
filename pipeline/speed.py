"""Скорость: сколько процессов запускать, сколько потоков им дать, прогресс-бары.

На Apple M-процессорах numpy считает через Accelerate, циклы методов скомпилированы numba.
Процессы запускаются через spawn; каждому достаётся своя доля ядер (NUMBA_NUM_THREADS и потоки BLAS),
чтобы четыре процесса по восемь потоков не дрались за восемь ядер.

    PT_JOBS=2          число процессов вместо автоматического
    PT_PROGRESS=0/1    выключить / включить прогресс-бары (по умолчанию — только в терминале)
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context

THREAD_VARS = ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS",
               "MKL_NUM_THREADS")


def _sysctl(name: str) -> int | None:
    try:
        return int(subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, timeout=2).stdout)
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def machine() -> dict:
    """Ядра (все и производительные — у M-процессоров есть ещё энергоэффективные) и память в ГБ."""
    cores = os.cpu_count() or 1
    perf = _sysctl("hw.perflevel0.physicalcpu") if sys.platform == "darwin" else None
    mem = _sysctl("hw.memsize") if sys.platform == "darwin" else None
    if mem is None:
        try:
            mem = os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
        except (ValueError, OSError, AttributeError):
            mem = 8 * 2 ** 30
    return {"cores": cores, "perf_cores": perf or cores, "mem_gb": mem / 2 ** 30}


def default_jobs(gb_per_job: float) -> int:
    """Процессов столько, сколько хватит производительных ядер и памяти (gb_per_job — оценка на процесс)."""
    if os.environ.get("PT_JOBS"):
        return max(1, int(os.environ["PT_JOBS"]))
    m = machine()
    return max(1, min(m["perf_cores"], int((m["mem_gb"] - 2) // gb_per_job)))


def progress_enabled() -> bool:
    flag = os.environ.get("PT_PROGRESS")
    return flag == "1" if flag in ("0", "1") else sys.stderr.isatty()


def progress(iterable=None, **kw):
    """tqdm, если прогресс включён; иначе тихая обёртка с тем же интерфейсом (update, set_postfix_str)."""
    if progress_enabled():
        from tqdm.auto import tqdm
        return tqdm(iterable, dynamic_ncols=True, **kw)
    return _Quiet(iterable)


class _Quiet:
    def __init__(self, iterable):
        self.iterable = iterable

    def __iter__(self):
        return iter(self.iterable)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def update(self, n=1):
        pass

    def set_postfix_str(self, s):
        pass

    def close(self):
        pass


def run_parallel(fn, items: list, jobs: int, desc: str):
    """fn(item) для каждого item в jobs процессах; отдаёт (item, результат, секунды) по мере готовности.

    Ошибка в одном item останавливает всё и пробрасывается с его именем. jobs = 1 — без процессов.
    """
    if jobs <= 1 or len(items) <= 1:
        bar = progress(total=len(items), desc=desc)
        for it in items:
            t = time.perf_counter()
            res = fn(it)
            bar.update()
            bar.set_postfix_str(str(it))
            yield it, res, time.perf_counter() - t
        bar.close()
        return
    threads = max(1, machine()["cores"] // jobs)
    saved = {k: os.environ.get(k) for k in THREAD_VARS + ("PT_PROGRESS",)}
    os.environ.update({k: str(threads) for k in THREAD_VARS}, PT_PROGRESS="0")   # дети наследуют при запуске
    try:
        with ProcessPoolExecutor(jobs, mp_context=get_context("spawn")) as pool:
            start = {}
            futs = {}
            for it in items:
                futs[pool.submit(_timed, fn, it)] = it
                start[it] = time.perf_counter()
            bar = progress(total=len(items), desc=f"{desc} [процессов {jobs}, потоков на каждый {threads}]")
            for f in as_completed(futs):
                it = futs[f]
                try:
                    res, secs = f.result()
                except Exception as e:
                    for other in futs:
                        other.cancel()
                    raise RuntimeError(f"{it}: {e}") from e
                bar.update()
                bar.set_postfix_str(f"{it} {secs:.0f} с")
                yield it, res, secs
            bar.close()
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _timed(fn, it):
    t = time.perf_counter()
    return fn(it), time.perf_counter() - t
