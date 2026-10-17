import bz2
from collections import Counter

import pytest

from pipeline.lanl_netflow import FLOW_COLUMNS, PipelineError, canonicalize, iter_raw_chunks, read_canonical

GOOD = ("100,2,Comp001,Comp002,6,Port5555,443,3,2,1500,900\n"
        "101,0,Comp002,Mail,17,Port7,53,1,1,80,60\n")


def write(tmp_path, text, name="netflow_day-02.bz2"):
    p = tmp_path / name
    with bz2.open(p, "wt") as f:
        f.write(text)
    return p


def test_headerless_bz2_maps_to_canonical(tmp_path):
    (df,) = list(read_canonical(write(tmp_path, GOOD)))
    assert list(df.columns) == FLOW_COLUMNS
    r = df.iloc[0]
    assert (r.src, r.dst, r.dst_port, r.proto) == ("Comp001", "Comp002", "443", "6")
    assert (r.bytes, r.packets, r.ts) == (1500, 3, 100)               # направление инициатора
    assert (r.resp_bytes, r.resp_packets, r.duration, r.src_port) == (900, 2, 2, "Port5555")
    assert df.iloc[1].dst == "Mail" and df.iloc[1].dst_port == "53"


def test_host_ids_are_preserved_verbatim(tmp_path):
    text = "1,0, Comp007 ,comp007,6,1,80,1,1,1,1\n2,0,007,7,6,1,80,1,1,1,1\n"
    (df,) = list(read_canonical(write(tmp_path, text)))
    assert df.src.tolist() == ["Comp007", "007"]          # только обрезка пробелов
    assert df.dst.tolist() == ["comp007", "7"]            # регистр и ведущие нули — разные хосты


def test_ports_stay_strings(tmp_path):
    (df,) = list(read_canonical(write(tmp_path, "1,0,A,B,6,Port1,Port99,1,1,1,1\n")))
    assert df.dst_port.iloc[0] == "Port99"


@pytest.mark.parametrize("line,reason", [
    ("x,0,A,B,6,1,80,1,1,1,1", "invalid_time"),
    ("1,-3,A,B,6,1,80,1,1,1,1", "invalid_duration"),
    ("1,0,,B,6,1,80,1,1,1,1", "blank_device"),
    ("1,0,A,B,6,1,80,1,1,-5,1", "invalid_count"),
    ("1,0,A,B,6,1,80,1,1,abc,1", "invalid_count"),
    ("1,0,A,B,,1,80,1,1,1,1", "blank_protocol_or_port"),
    ("inf,0,A,B,6,1,80,1,1,1,1", "invalid_time"),
])
def test_invalid_rows_are_dropped_and_counted(tmp_path, line, reason):
    bad = Counter()
    frames = list(read_canonical(write(tmp_path, GOOD + line + "\n"), bad=bad))
    assert sum(len(f) for f in frames) == 2 and bad[reason] == 1


def test_wrong_field_count_strict_raises_lenient_counts(tmp_path):
    p = write(tmp_path, GOOD + "1,0,A,B,6,1,80,1,1,1,1,EXTRA\n")
    with pytest.raises(PipelineError):
        list(read_canonical(p))
    bad = Counter()
    frames = list(read_canonical(p, strict=False, bad=bad))
    assert sum(len(f) for f in frames) == 2 and bad["wrong_field_count"] == 1


def test_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        list(iter_raw_chunks(tmp_path / "nope.bz2"))


def test_chunking_gives_same_rows(tmp_path):
    text = "".join(f"{i},0,A{i},B{i},6,1,80,1,1,1,1\n" for i in range(25))
    p = write(tmp_path, text)
    a = list(read_canonical(p, chunksize=7))
    assert len(a) == 4 and sum(len(x) for x in a) == 25
