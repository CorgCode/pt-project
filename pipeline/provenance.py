"""What the pipeline can and cannot know about where its input bytes came from.

LANL publishes no checksums that this pipeline can compare against, so provenance is NEVER
"verified" here. Three kinds exist, in decreasing order of what they claim:

    lanl_declared       the operator explicitly declared the files to be LANL netflow
                        (--data-kind lanl plus a non-empty --source-note: where/when obtained),
                        and the files pass the cheap sanity checks below. This is a DECLARATION.
    unverified          default for any unmarked input. Nothing is claimed about its origin.
    synthetic_fixture   output of pipeline.fixtures (marker file or recognised generator signature).

Downstream artifacts carry `data_kind`, `provenance_verified: false`, `synthetic` and the
declaration, and every report/dashboard prints which of the three applies. Synthetic or
unverified inputs can never be presented as LANL.

Sanity checks for a LANL declaration (they catch mistakes, they do not prove authenticity):
  * the SYNTHETIC_FIXTURE marker must be absent;
  * the content must not match the fixture generator's signature (device numbers are
    100000 + 7*i in pipeline.fixtures; real LANL device numbers are not an arithmetic progression);
  * every file name must look like netflow_day-NN.bz2.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from pipeline.lanl_netflow import RAW_COLUMNS, PipelineError

SYNTHETIC_MARKER = "SYNTHETIC_FIXTURE"
KIND_SYNTHETIC = "synthetic_fixture"
KIND_UNVERIFIED = "unverified"
KIND_LANL_DECLARED = "lanl_declared"
KINDS = (KIND_SYNTHETIC, KIND_UNVERIFIED, KIND_LANL_DECLARED)
FILENAME_RE = re.compile(r"^netflow_day-\d{2}\.bz2$")
_COMP_RE = re.compile(r"^Comp(\d+)$")

BADGE = {
    KIND_SYNTHETIC: "SYNTHETIC FIXTURE - not LANL traffic",
    KIND_UNVERIFIED: "UNVERIFIED INPUT - origin not established, not presented as LANL",
    KIND_LANL_DECLARED: "LANL (declared by the operator; provenance not verified by the pipeline)",
}


def fixture_signature(devices: set[str]) -> bool:
    """True if the device IDs look like pipeline.fixtures output (Comp100000 + 7*i)."""
    nums = [int(m.group(1)) for d in devices if (m := _COMP_RE.match(d))]
    if len(nums) < 20 or len(nums) < 0.9 * len(devices):
        return False
    return sum((n - 100000) % 7 == 0 and 0 <= n - 100000 < 7 * 100000 for n in nums) >= 0.99 * len(nums)


def sample_devices(path: Path, rows: int = 20000) -> tuple[set[str], dict]:
    """Device IDs and format statistics from the first rows of a raw file (bounded work)."""
    df = pd.read_csv(path, header=None, names=RAW_COLUMNS, dtype=str, nrows=rows, keep_default_na=False,
                     on_bad_lines="skip")
    devices = set(df["SrcDevice"].str.strip()) | set(df["DstDevice"].str.strip())
    devices.discard("")
    stats = {
        "sampled_rows": int(len(df)),
        "share_devices_named_CompN": (sum(bool(_COMP_RE.match(d)) for d in devices) / len(devices)) if devices else 0.0,
        "share_ports_numeric_or_PortN": float(df["DstPort"].str.fullmatch(r"\d+|Port\d+").mean()) if len(df) else 0.0,
        "share_protocol_numeric": float(df["Protocol"].str.fullmatch(r"\d+").mean()) if len(df) else 0.0,
    }
    return devices, stats


def classify(inputs: list[Path], declared: str | None = None, source_note: str | None = None) -> dict:
    """Decide the provenance kind of a set of input files; raise on contradictory claims."""
    if declared not in (None, "lanl", KIND_SYNTHETIC, KIND_LANL_DECLARED):
        raise PipelineError(f"unknown data kind {declared!r}")
    wants_lanl = declared in ("lanl", KIND_LANL_DECLARED)
    marker = any((Path(p).parent / SYNTHETIC_MARKER).exists() for p in inputs)
    devices: set[str] = set()
    fmt: dict = {}
    for p in inputs:
        try:
            d, s = sample_devices(Path(p))
        except Exception:                       # unreadable here; the real reader reports it properly
            continue
        devices |= d
        fmt[Path(p).name] = s
    signature = fixture_signature(devices)
    names_ok = all(FILENAME_RE.match(Path(p).name) for p in inputs)
    checks = {"marker_file": marker, "fixture_signature": signature, "filename_pattern": names_ok,
              "format_sample": fmt}
    out = {"verified": False,
           "verification": "none: no published checksums are available to this pipeline; a declaration is not proof",
           "declared_kind": declared, "source_note": (source_note or "").strip() or None, "checks": checks}
    if marker:
        if wants_lanl:
            raise PipelineError("inputs carry a SYNTHETIC_FIXTURE marker; refusing to label them LANL")
        return {**out, "kind": KIND_SYNTHETIC, "basis": "marker_file"}
    if signature:
        if wants_lanl:
            raise PipelineError("input device IDs match the synthetic fixture generator's signature; "
                                "refusing to label them LANL")
        return {**out, "kind": KIND_SYNTHETIC, "basis": "fixture_signature"}
    if declared == KIND_SYNTHETIC:
        return {**out, "kind": KIND_SYNTHETIC, "basis": "operator_declaration"}
    if wants_lanl:
        if not out["source_note"]:
            raise PipelineError("declaring LANL data requires --source-note (where and when the files were obtained)")
        if not names_ok:
            bad = [Path(p).name for p in inputs if not FILENAME_RE.match(Path(p).name)]
            raise PipelineError(f"file names do not look like LANL netflow_day-NN.bz2: {bad[:3]}")
        return {**out, "kind": KIND_LANL_DECLARED, "basis": "operator_declaration"}
    return {**out, "kind": KIND_UNVERIFIED, "basis": "no_declaration"}
