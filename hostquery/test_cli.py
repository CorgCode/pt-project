"""CLI: JSON совпадает с HTTP-контрактом, ошибки возвращают код 2."""
from __future__ import annotations

import json

import pytest

from hostquery.cli import _read_vector, build_parser, main
from hostquery.errors import HostQueryError

VECTORS = """host_id,emb_0,emb_1
a,1,0
b,0,1
c,1,1
"""

FLOWS = """src,dst,bytes,packets,proto,dst_port
h1,h2,10,1,6,443
h1,h3,20,2,17,53
h2,h1,30,3,6,443
h3,h1,40,4,6,80
"""


def run_json(capsys, *args):
    code = main([*args, "--json"])
    streams = capsys.readouterr()
    assert code == 0, streams.err
    return json.loads(streams.out)


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_hosts_json_and_limit(capsys, demo_path):
    result = run_json(capsys, "hosts", "--bundle", str(demo_path), "--limit", "3")
    assert result["schema"] == "hostquery/1" and result["query"]["kind"] == "hosts"
    assert len(result["hosts"]) == 3 and result["total"] == 300
    assert result["provenance"]["synthetic"] is True
    assert len(result["provenance"]["embeddings_sha256"]) == 64


def test_similar_json_has_reference_marks(capsys, demo_path):
    result = run_json(capsys, "similar", "--bundle", str(demo_path),
                      "--host", "Comp100014", "-k", "5")
    assert result["host"] == "Comp100014" and result["k"] == 5
    assert len(result["neighbors"]) == 5
    assert any("reference_rank" in item for item in result["neighbors"])
    assert all(item["id"] != "Comp100014" for item in result["neighbors"])


def test_similar_text_is_human_readable(capsys, demo_path):
    code = main(["similar", "--bundle", str(demo_path), "--host", "Comp100014", "-k", "3"])
    out = capsys.readouterr().out
    assert code == 0
    assert "#" in out and "косинус" in out and "эталон" in out
    assert out.count("\n") >= 4


def test_similar_with_vector_and_groups(capsys, tmp_path):
    path = write(tmp_path, "vectors.csv", VECTORS)
    result = run_json(capsys, "similar", "--embeddings", str(path), "--vector", "1,0",
                      "-k", "2", "--groups", "2")
    ids = [item["id"] for item in result["neighbors"]]
    assert ids == ["a", "c"]
    assert all("group" in item for item in result["neighbors"])


def test_vector_can_be_read_from_csv_file(tmp_path):
    path = write(tmp_path, "query.csv", "host_id,emb_0,emb_1\nq,0.5,0.5\n")
    assert _read_vector(str(path)) == [0.5, 0.5]
    assert _read_vector("1; 2; 3") == [1.0, 2.0, 3.0]
    with pytest.raises(HostQueryError, match="вектор"):
        _read_vector("a,b")


def test_pair_text_mentions_contacts(capsys, tmp_path):
    path = write(tmp_path, "flows.csv", FLOWS)
    code = main(["pair", "--embeddings", str(path), "--a", "h1", "--b", "h2"])
    out = capsys.readouterr().out
    assert code == 0
    assert "<->" in out and "общие собеседники" in out
    assert "Исходящие записи" in out


def test_groups_and_group_json(capsys, demo_path):
    result = run_json(capsys, "groups", "--bundle", str(demo_path), "-n", "3")
    assert result["k"] == 3 and len(result["labels"]) == 300
    assert len(result["profiles"]) == 3 and len(result["projection"]) == 300
    assert sum(profile["size"] for profile in result["profiles"]) == 300
    one = run_json(capsys, "group", "--bundle", str(demo_path), "-n", "3", "--group", "2")
    assert one["group"] == 2 and one["members"] and one["top_features"]


def test_empty_groups_are_rendered_without_error(capsys, tmp_path):
    path = write(tmp_path, "identical.csv", "host_id,emb_0,emb_1\na,1,1\nb,1,1\nc,1,1\n")
    result = run_json(capsys, "groups", "--embeddings", str(path), "-n", "3")
    assert result["sizes"] == [3, 0, 0] and len(result["profiles"]) == 3
    assert result["profiles"][1]["empty"] is True
    assert result["profiles"][1]["centroid"] is None

    one = run_json(capsys, "group", "--embeddings", str(path), "-n", "3", "--group", "1")
    assert one["group"] == 1 and one["size"] == 0 and one["empty"] is True
    assert one["mean_similarity"] is None and one["members"] == []

    code = main(["groups", "--embeddings", str(path), "-n", "3"])
    out = capsys.readouterr().out
    assert code == 0 and "(пусто)" in out


def test_group_out_of_range_is_a_clean_error(capsys, demo_path):
    code = main(["group", "--bundle", str(demo_path), "-n", "3", "--group", "9"])
    streams = capsys.readouterr()
    assert code == 2 and "Ошибка" in streams.err and "от 0 до 2" in streams.err
    code = main(["group", "--bundle", str(demo_path), "-n", "3", "--group", "9", "--json"])
    streams = capsys.readouterr()
    assert code == 2 and "error" in json.loads(streams.out)


def test_consensus_json(capsys, demo_path):
    result = run_json(capsys, "consensus", "--bundle", str(demo_path),
                      "--host", "Comp100014", "-k", "4", "--methods", "behavioral,deepwalk")
    assert result["n_methods"] == 2
    assert all(item["votes"] >= 1 for item in result["neighbors"])


def test_outliers_json(capsys, demo_path):
    result = run_json(capsys, "outliers", "--bundle", str(demo_path), "-k", "5", "--groups", "3")
    assert result["k"] == 5 and len(result["items"]) == 300
    scores = [item["score"] for item in result["items"]]
    assert scores == sorted(scores, reverse=True)
    assert all("group" in item for item in result["items"])


def test_changes_json_between_two_csv(tmp_path, capsys):
    before = write(tmp_path, "before.csv", "src,dst\nx,y\ny,x\n")
    after = write(tmp_path, "after.csv", "src,dst\nx,y\ny,x\nx,z\n")
    result = run_json(capsys, "changes", "--before", str(before), "--after", str(after), "-k", "2")
    assert result["query"]["kind"] == "changes" and result["kind"] == "flows"
    assert result["newHosts"] == ["z"]
    assert next(item for item in result["changes"] if item["id"] == "x")["added"] == ["z"]
    compare_info = result["provenance"]["compare"]
    assert compare_info["before"] == str(before) and compare_info["after"] == str(after)
    assert len(compare_info["after_sha256"]) == 64
    assert len(result["provenance"]["embeddings_sha256"]) == 64


def test_changes_text_shows_counts(tmp_path, capsys):
    before = write(tmp_path, "before.csv", FLOWS)
    after = write(tmp_path, "after.csv", FLOWS + "h1,h4,90,9,6,80\n")
    code = main(["changes", "--before", str(before), "--after", str(after), "-k", "2"])
    out = capsys.readouterr().out
    assert code == 0 and "общих:" in out and "контакты" in out


def test_cli_rejects_ambiguous_sources(capsys, demo_path, tmp_path):
    path = write(tmp_path, "vectors.csv", VECTORS)
    code = main(["similar", "--bundle", str(demo_path), "--embeddings", str(path), "--host", "a"])
    assert code == 2 and "один источник" in capsys.readouterr().err
    code = main(["similar", "--bundle", str(demo_path), "--host", "Comp100014", "--vector", "1,0"])
    assert code == 2 and "ровно один" in capsys.readouterr().err
    code = main(["similar", "--bundle", str(demo_path)])
    assert code == 2 and "ровно один" in capsys.readouterr().err
    code = main(["similar", "--bundle", str(demo_path), "--method", "nope", "--host", "Comp100014"])
    assert code == 2 and "нет в источнике" in capsys.readouterr().err


def test_parser_lists_all_commands():
    parser = build_parser()
    subparsers = next(action for action in parser._actions
                      if action.__class__.__name__ == "_SubParsersAction")
    assert set(subparsers.choices) == {
        "hosts", "similar", "consensus", "groups", "group",
        "pair", "changes", "outliers", "serve",
    }
