"""Верхняя граница score PT-05 в снимке 70347a3.

Запуск: python3 check_score_ceiling.py /путь/к/pt-project
Нужны Python 3, git и локальный репозиторий с указанным коммитом.
Скрипт читает опубликованные эталоны и не изменяет репозиторий.
"""
import argparse
import csv
import io
import json
import subprocess
from collections import defaultdict
from functools import lru_cache
from statistics import mean


def snapshot(repo, path):
    return subprocess.check_output(
        ["git", "-C", repo, "show", "70347a3:" + path], text=True
    )


def bound_window(repo, window, baseline):
    directory = "scoreboard/data/reference/" + window + "/"
    hosts = {row["host_id"] for row in csv.DictReader(io.StringIO(snapshot(repo, directory + "eval_hosts.csv")))}
    universe, source, by_partner, target = set(), defaultdict(dict), defaultdict(dict), defaultdict(set)
    for row in csv.DictReader(io.StringIO(snapshot(repo, directory + "partners.csv"))):
        host, partner = row["host_id"], row["partner"]
        if row["window"] == "source":
            universe.add(partner)
            if host in hosts:
                source[host][partner] = int(row["count"])
                by_partner[partner][host] = int(row["count"])
        elif host in hosts:
            target[host].add(partner)

    @lru_cache(None)
    def dominators(partner):
        # Test every positive row in the column. Other rows have count zero
        # and all count columns are nonnegative. Including the query row
        # in the test is conservative (the query cannot be its own donor).
        support = sorted(by_partner[partner].items(), key=lambda row: len(source[row[0]]))
        if not support:
            return universe - {partner}
        host, count = support[0]
        possible = {q for q, n in source[host].items() if q != partner and n >= count}
        for host, count in support[1:]:
            possible = {q for q in possible if source[host].get(q, 0) >= count}
            if not possible:
                break
        return possible

    bounds, example = [], None
    for host, targets in sorted(target.items()):
        eligible = (targets & universe) - source[host].keys() - {host}
        if not eligible:
            continue
        best_rr = 0.0
        for partner in sorted(eligible):
            competing = dominators(partner) - source[host].keys() - {host}
            best_rr = max(best_rr, 1 / (1 + len(competing) / 2))
            if best_rr == 1:
                break
        bounds.append(best_rr)
        if example is None and len(eligible) == 1 and best_rr < 1:
            example = (host, partner, sorted(competing)[0], len(by_partner[partner]), best_rr)
    mrr_bound = mean(bounds)
    # MRR has exponent 1/2 within Future links and 1/4 across the four tasks.
    normalized_mrr = max(0.01, (mrr_bound - baseline) / (1 - baseline))
    score_bound = 100 * normalized_mrr ** (1 / 8)
    print(window, len(hosts), "eligible", len(bounds), "blocked", sum(x < 1 for x in bounds),
          "MRR <=", round(mrr_bound, 9), "score <=", round(score_bound, 6))
    print("Example (query, only eligible target, dominating candidate, target donor count, RR ceiling):", example)
    return score_bound


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo", help="путь к репозиторию pt-project")
    repo = parser.parse_args().repo
    result = json.loads(snapshot(repo, "scoreboard/results/multiday_codebook_plus.json"))
    scores = []
    for name, data in result["by_window"].items():
        baseline = next(c["chance"] for c in data["summary"] if c["metric"] == "partner_new_mrr")
        scores.append(bound_window(repo, name, baseline))
    print("Mean score <=", round(mean(scores), 6))


if __name__ == "__main__":
    main()
