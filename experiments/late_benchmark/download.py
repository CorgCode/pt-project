"""Resumable, checked HTTP-range download of the requested LANL NetFlow days."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import fcntl
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from urllib.parse import urlsplit


class DownloadError(RuntimeError):
    pass


def request(url, *, method="GET", start=None, end=None):
    headers = {"User-Agent": "PT-LANL-research/1", "Accept-Encoding": "identity"}
    if start is not None:
        headers["Range"] = f"bytes={start}-{end}"
    return urllib.request.urlopen(urllib.request.Request(url, method=method, headers=headers), timeout=90)


def metadata(url):
    try:
        with request(url, method="HEAD") as response:
            size = int(response.headers["Content-Length"])
            if size <= 0:
                raise DownloadError("empty remote archive")
            return {"bytes": size, "etag": response.headers.get("ETag"),
                    "last_modified": response.headers.get("Last-Modified")}
    except urllib.error.HTTPError as exc:
        raise DownloadError(f"LANL returned HTTP {exc.code}; refresh the download link if 401/403") from None


def fetch_part(url, path, start, end, total, attempts=8):
    """Never append an ignored Range response or bytes belonging to another offset."""
    path = Path(path)
    expected = end - start + 1
    for attempt in range(attempts):
        have = path.stat().st_size if path.exists() else 0
        if have == expected:
            return expected
        if have > expected:
            raise DownloadError(f"oversized partial file: {path.name}")
        try:
            with request(url, start=start + have, end=end) as response:
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", response.headers.get("Content-Range", ""))
                if response.status != 206 or not match or tuple(map(int, match.groups())) != (start + have, end, total):
                    raise DownloadError("server did not honor the exact requested byte range")
                with path.open("ab") as output:
                    remaining = expected - have
                    while remaining:
                        block = response.read(min(1024 * 1024, remaining))
                        if not block:
                            break
                        output.write(block)
                        remaining -= len(block)
                    if response.read(1):
                        raise DownloadError("response exceeds the requested byte range")
            if path.stat().st_size == expected:
                return expected
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403, 404):
                raise DownloadError(f"LANL returned HTTP {exc.code}") from None
        except (OSError, TimeoutError):
            pass
        if attempt + 1 < attempts:
            time.sleep(min(2 ** attempt, 10))
    raise DownloadError(f"incomplete range after {attempts} attempts: {path.name}")


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def download_days(base_url, output, days=(90, 89, 88), workers=36, part_size=64 * 1024 * 1024):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if any(not 2 <= day <= 90 for day in days) or workers < 1 or part_size < 1:
        raise ValueError("invalid days or download limits")
    if not base_url.startswith("https://csr.lanl.gov/data-fence/"):
        raise ValueError("expected the user-provided HTTPS LANL data-fence URL")
    plans, jobs = {}, []
    for day in days:
        name = f"netflow_day-{day:02d}.bz2"
        url = base_url.rstrip("/") + "/netflow/" + name
        info = metadata(url)
        folder = output / (name + ".parts")
        folder.mkdir(exist_ok=True)
        record = folder / "remote.json"
        if record.exists() and json.loads(record.read_text()) != info:
            raise DownloadError(f"remote archive changed: {name}; existing partial data left intact")
        atomic_json(record, info)
        final = output / name
        receipt = output / (name + ".json")
        if final.exists():
            if final.stat().st_size != info["bytes"] or not receipt.exists():
                raise DownloadError(f"existing archive has no matching completion record: {name}")
            print(f"already downloaded: {name}", flush=True)
            continue
        plans[day] = {"url": url, "info": info, "folder": folder, "name": name}
        for number, start in enumerate(range(0, info["bytes"], part_size)):
            jobs.append((day, folder / f"{number:05d}.part", start, min(start + part_size, info["bytes"]) - 1))
    jobs.sort(key=lambda job: job[2])  # interleave days: twelve connections per archive
    completed = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(fetch_part, plans[day]["url"], part, start, end, plans[day]["info"]["bytes"]): day
                   for day, part, start, end in jobs}
        pending = {day: sum(job[0] == day for job in jobs) for day in plans}
        for future in as_completed(futures):
            day = futures[future]
            completed += future.result()
            pending[day] -= 1
            atomic_json(output / "progress.json", {"completed_bytes": completed, "total_bytes": sum(p["info"]["bytes"] for p in plans.values()),
                                                      "pending_ranges": pending})
            print(f"range complete: day {day}, {completed / 1024**3:.2f} GiB, {pending}", flush=True)
            if pending[day]:
                continue
            plan = plans[day]
            temporary = output / (plan["name"] + ".assembling")
            digest = hashlib.sha256()
            with temporary.open("wb") as target:
                for part in sorted(plan["folder"].glob("*.part")):
                    with part.open("rb") as source:
                        for block in iter(lambda: source.read(1024 * 1024), b""):
                            digest.update(block)
                            target.write(block)
            if temporary.stat().st_size != plan["info"]["bytes"]:
                raise DownloadError("assembled archive size mismatch")
            with temporary.open("rb") as source:
                if source.read(3) != b"BZh":
                    raise DownloadError("download is not a bzip2 archive")
            os.replace(temporary, output / plan["name"])
            atomic_json(output / (plan["name"] + ".json"), {**plan["info"], "sha256": digest.hexdigest(),
                        "source_host": urlsplit(base_url).hostname, "file": plan["name"],
                        "validation": "exact HTTP ranges, total length, bzip2 header; CRC verified during extraction"})
            for part in plan["folder"].glob("*.part"):
                part.unlink()
            print(f"archive ready: {plan['name']}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--access", type=Path, default=Path("data/late_benchmark/download_access.json"))
    parser.add_argument("--output", type=Path, default=Path("data/late_benchmark/raw"))
    parser.add_argument("--days", type=int, nargs="+", default=[90, 89, 88])
    parser.add_argument("--workers", type=int, default=36)
    args = parser.parse_args()
    try:
        args.output.mkdir(parents=True, exist_ok=True)
        with (args.output / ".download.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise DownloadError("a downloader already owns this output directory") from None
            download_days(json.loads(args.access.read_text())["base_url"], args.output, args.days, args.workers)
    except (DownloadError, OSError) as exc:
        # URLs contain access credentials; never include urllib exception URLs in logs.
        print(f"Download failed: {exc if isinstance(exc, DownloadError) else type(exc).__name__}", flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
