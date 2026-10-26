"""Общая таблица результатов: положил эмбеддинг — получил место в таблице.

    # оценить одну отправку (папка scoreboard/submissions/<метод>/)
    python -m evaluation.submit scoreboard/submissions/my_method

    # оценить все отправки, у которых нет свежего результата (так делает CI)
    python -m evaluation.submit --all

    # пересчитать бейзлайны и пять графовых методов на общих данных (уже обученные на тех же данных
    # пропускаются; --force — заново; --jobs N — сколько процессов, по умолчанию по ядрам и памяти)
    python -m evaluation.submit --builtin all

Папка отправки:
    embeddings.csv        эмбеддинг train-окна: host_id, emb_0, emb_1, ...   (обязательно)
    test.csv              эмбеддинг test-окна тех же хостов — для «узнаваемости во времени» (по желанию)
    SHARED_COORDINATES    пустой файл, если оба файла в одной системе координат (по желанию)
    AUTHOR                логин на GitHub того, кто прислал метод (по желанию; колонка submitted by)
    README.md             карточка метода по scoreboard/CARD_TEMPLATE.md (обязательно; на неё ссылка из таблицы)

Всё сохраняется в scoreboard/: results/<метод>.json — полный результат, README.md — таблица,
scoreboard.csv — все метрики. Данные для обучения и эталоны — scoreboard/data/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation import scoreboard
from evaluation.evaluate import run
from evaluation.identity import identity_retrieval
from evaluation.io import load_embeddings
from evaluation.validate import validate_submission
from pipeline.windows import is_test, ordered_windows

ROOT = Path(__file__).resolve().parents[1]
BOARD = ROOT / "scoreboard"
BOARD_IN_REPO = BOARD                                   # не подменяется в тестах: отсюда берутся карточки встроенных методов
DATA = BOARD / "data"
SUBMISSIONS = BOARD / "submissions"
RESULTS = BOARD / "results"
N_BOOT = 200
SEED = 0
CARD_SECTIONS = ("Кратко", "Архитектура", "Данные", "Обучение и подбор", "Ограничения")
CARD_URL = "https://github.com/CorgCode/pt-project/blob/main/scoreboard/submissions/{}/README.md"


def processed_dir() -> Path:
    """Подготовленные данные: PT_DATA или scoreboard/data/processed (там только манифест и когорта, без потоков)."""
    return Path(os.environ.get("PT_DATA", DATA / "processed"))


def _sha(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in paths:
        if p.exists():
            h.update(p.name.encode())
            h.update(p.read_bytes())
    return h.hexdigest()


def test_windows() -> list[str]:
    """Test-окна, для которых есть эталоны: test и test_<метка> (другие дни), по времени."""
    ref = DATA / "reference"
    m = json.loads((ref / "test" / "manifest.json").read_text(encoding="utf-8"))
    return [w for w in ordered_windows(m["windows"]) if is_test(w) and (ref / w / "manifest.json").exists()]


def window_file(w: str) -> str:
    """Имя файла эмбеддинга test-окна: test.csv для test, test_day03.csv для test_day03."""
    return f"{w}.csv"


def _train_file(sub: Path, w: str) -> Path:
    """Эмбеддинг, который оценивается на окне w: обычно embeddings.csv (свой — только у reference_ceiling)."""
    own = sub / f"embeddings_{w}.csv"
    return own if own.exists() else sub / "embeddings.csv"


def _inputs(d: Path) -> list[Path]:
    ws = test_windows()
    return ([d / "embeddings.csv"] + [d / window_file(w) for w in ws] + [d / f"embeddings_{w}.csv" for w in ws]
            + [d / "SHARED_COORDINATES", d / "AUTHOR"] + [DATA / "reference" / w / "manifest.json" for w in ws]
            + _evaluation_code())


LOCAL_VECTORS = "*.csv\n"        # .gitignore методов, которые пересобираются из кода и данных LANL


def keep_vectors_local(d: Path) -> None:
    """Векторы метода не идут в git: их можно пересобрать (experiments/lanl/multiday.sh), а каждая пересборка
    добавляла бы в историю сотни мегабайт. В git остаются карточка, параметры и результат. Отправки команды
    (без этого файла) хранят векторы в git как раньше — по ним CI и считает результат."""
    Path(d, ".gitignore").write_text(LOCAL_VECTORS, encoding="utf-8")


def _code_sha256() -> str:
    """Версия оценки: код и эталоны. Не совпала у результата, векторов которого нет, — результат помечается."""
    return _sha(*_evaluation_code(), *[DATA / "reference" / w / "manifest.json" for w in test_windows()])


def _evaluation_code() -> list[Path]:
    """Код оценки: поменялся (например, добавилась метрика) — результаты пересчитываются.
    Без submit.py и scoreboard.py: они только собирают таблицу и текст страницы."""
    return sorted(f for f in (ROOT / "evaluation").glob("*.py")
                  if not f.name.startswith("test_") and f.name not in ("submit.py", "scoreboard.py"))


def check_card(sub: Path) -> None:
    """Карточка README.md: все разделы CARD_SECTIONS есть, не пустые и без TODO."""
    card = Path(sub) / "README.md"
    if not card.exists():
        raise ValueError(f"{card}: нет карточки метода — скопируй scoreboard/CARD_TEMPLATE.md и заполни")
    body: dict[str, str] = {}
    current = None
    for line in card.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = line[3:].strip()
            body[current] = ""
        elif current is not None:
            body[current] += line + "\n"
    bad = [h for h in CARD_SECTIONS
           if not re.sub(r"<!--.*?-->", "", body.get(h, ""), flags=re.S).strip() or "TODO" in body.get(h, "")]
    if bad:
        raise ValueError(f"{card}: заполни разделы {', '.join(bad)} (по scoreboard/CARD_TEMPLATE.md)")


def _score_window(sub: Path, w: str) -> dict:
    """Оценка отправки на одном test-окне w (эталоны scoreboard/data/reference/<w>)."""
    method = sub.name
    emb_file = _train_file(sub, w)
    validate_submission(str(emb_file))
    ref_test, ref_val = DATA / "reference" / w, DATA / "reference" / "validation"
    # классификатор связей учится на кандидатах validation — нужен эмбеддинг и для их хостов
    val_hosts = set(pd.read_csv(ref_val / "eval_hosts.csv", dtype=str)["host_id"]) if (ref_val / "eval_hosts.csv").exists() else set()
    covers_val = val_hosts <= set(pd.read_csv(emb_file, dtype={"host_id": str}, usecols=["host_id"])["host_id"])
    r = run(emb_file, ref_test, method, n_boot=N_BOOT, seed=SEED, fit_reference_dir=ref_val if covers_val else None)
    test_file = sub / window_file(w)
    if test_file.exists():
        add_identity(r, emb_file, test_file, ref_test, shared=(sub / "SHARED_COORDINATES").exists())
    else:
        r["metric_status"]["identity"] = f"unavailable: нет {test_file.name} (эмбеддинга окна {w})"
        r["unavailable_metrics"].append(f"identity: нет {test_file.name} (эмбеддинга окна {w})")
    return r


def add_identity(r: dict, emb_file: Path, later_file: Path, ref_dir: Path, *, shared: bool = False,
                 n_boot: int | None = None) -> None:
    """Узнаваемость: хост по вектору позднего окна (later_file) ищется среди векторов train (emb_file)."""
    validate_submission(str(later_file))
    hosts = pd.read_csv(ref_dir / "eval_hosts.csv", dtype=str)["host_id"].tolist()
    a, b = load_embeddings(emb_file), load_embeddings(later_file)
    missing = sorted(set(hosts) - set(b.index))
    if missing:
        raise ValueError(f"{later_file}: нет {len(missing)} оцениваемых хостов (например {missing[:5]})")
    ident = identity_retrieval(a.loc[hosts], b.loc[hosts], alignment="procrustes", seed=SEED,
                               n_boot=N_BOOT if n_boot is None else n_boot)
    r["metrics"].update({"identity_top1": ident["top1"], "identity_mrr": ident["mrr"],
                         "identity_pair_roc_auc": ident["pair_roc_auc"]})
    if ident.get("top1_ci"):
        r["metrics_ci"]["identity_top1"] = list(ident["top1_ci"])
    r["metric_details"]["identity"] = {**ident, "shared_coordinate_system_declared": shared}
    r["metric_status"]["identity"] = "ok"
    r["summary"].append({"group": "Узнаваемость", "metric": "identity_top1", "title": "Узнаваемость, Top-1",
                         "value": ident["top1"], "ci": r["metrics_ci"].get("identity_top1"),
                         "chance": ident["chance_top1"]})


def validation_score(emb_file: Path, validation_file: Path | None = None) -> tuple[float, dict]:
    """Балл на validation для подбора (test не трогается): те же колонки, Identity — если есть
    эмбеддинг validation-окна (validation.csv). Возвращает (score, баллы по задачам)."""
    ref_val = DATA / "reference" / "validation"
    r = run(emb_file, ref_val, "candidate", n_boot=0)
    if validation_file is not None and Path(validation_file).exists():
        add_identity(r, emb_file, validation_file, ref_val, n_boot=0)
    cols = {s["metric"]: s["title"] for s in r["summary"]}
    return float(scoreboard.overall_score([dict(r, method="candidate")], cols)["candidate"]), \
        {scoreboard.TASK_EN.get(k, k): v for k, v in scoreboard.task_scores(r, cols).items()}


def _mean(xs):
    xs = [x for x in xs if isinstance(x, (int, float)) and not isinstance(x, bool) and x == x]
    return sum(xs) / len(xs) if xs else None


def combine_windows(per: dict[str, dict]) -> dict:
    """Итог по нескольким test-окнам: значения — среднее по окнам, в скобках — от меньшего до большего по окнам.

    Метрика учитывается, только если она есть во всех окнах (иначе среднее сравнивало бы разные дни).
    """
    ws = list(per)
    r = json.loads(json.dumps(per[ws[0]]))
    if len(ws) == 1:
        return r
    keys = set.intersection(*(set(p["metrics"]) for p in per.values()))
    r["metrics"] = {k: _mean([per[w]["metrics"][k] for w in ws]) for k in sorted(keys)}
    r["metrics_ci"] = {}
    for k in keys:
        vals = [per[w]["metrics"][k] for w in ws if isinstance(per[w]["metrics"][k], (int, float))]
        if len(vals) == len(ws):
            r["metrics_ci"][k] = [min(vals), max(vals)]
    summary = []
    for s in r["summary"]:
        rows = [next((x for x in per[w]["summary"] if x["metric"] == s["metric"]), None) for w in ws]
        if any(x is None for x in rows):
            continue
        summary.append({**s, "value": _mean([x["value"] for x in rows]), "chance": _mean([x.get("chance") for x in rows]),
                        "ci": r["metrics_ci"].get(s["metric"])})
    r["summary"] = summary
    r["ci_kind"] = "range over test windows"
    r["test_windows"] = ws
    r["by_window"] = {w: {"metrics": per[w]["metrics"], "summary": per[w]["summary"],
                          "metrics_ci": per[w].get("metrics_ci", {})} for w in ws}
    return r


def score(sub: Path) -> dict:
    """Проверить и оценить одну отправку на всех test-окнах, сохранить results/<метод>.json."""
    sub = Path(sub)
    method = sub.name
    check_card(sub)
    r = combine_windows({w: _score_window(sub, w) for w in test_windows()})
    emb_file = sub / "embeddings.csv"
    author = sub / "AUTHOR"
    r["author"] = author.read_text(encoding="utf-8").strip().lstrip("@") if author.exists() else None
    r["submission"] = str(emb_file.relative_to(ROOT)) if emb_file.is_relative_to(ROOT) else str(emb_file)
    r["inputs_sha256"] = _sha(*_inputs(sub))
    r["code_sha256"] = _code_sha256()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / f"{method}.json").write_text(json.dumps(r, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
                                            encoding="utf-8")
    return r


def is_fresh(sub: Path) -> bool:
    f = RESULTS / f"{sub.name}.json"
    return f.exists() and json.loads(f.read_text(encoding="utf-8")).get("inputs_sha256") == _sha(*_inputs(sub))


def render() -> None:
    """Пересобрать scoreboard/README.md и scoreboard.csv из results/."""
    results = scoreboard.load_results(RESULTS)
    known = {d.name for d in SUBMISSIONS.iterdir() if d.is_dir() and not d.name.startswith(".")} if SUBMISSIONS.exists() else set()
    for r in results:                                   # результат без отправки — устарел
        if r["method"] not in known:
            (RESULTS / f"{r['method']}.json").unlink()
    results = [r for r in results if r["method"] in known]
    if not results:
        return
    for m in sorted(known):
        check_card(SUBMISSIONS / m)
    scoreboard.check_compatible(results)
    scoreboard.full_table(results).to_csv(BOARD / "scoreboard.csv", index=False)
    cols, values, order = scoreboard.ranking(results)
    code = _code_sha256()
    stale = {r["method"] for r in results if r.get("code_sha256") != code}    # векторов нет — пересчитать нечем
    stale_text = (f"\n† — оценено прежней версией оценки: векторов метода нет в git, пересчёт — "
                  f"`experiments/lanl/multiday.sh` ({', '.join(f'`{m}`' for m in sorted(stale))}).\n" if stale else "")
    info = results[0].get("reference_info") or {}
    data = ("Данные — LANL, корпоративная сеть" if info.get("data_kind") == "lanl_declared"
            else "Данные — синтетика (проверка кода, не реальная сеть)")
    labels, plan = _window_labels()
    multi = len(labels) > 1
    ci_text = ("В скобках — от худшего до лучшего значения по test-окнам (дням)." if multi
               else "В скобках — диапазон, в котором значение лежит с вероятностью 95%.")
    hist = [r["metrics"]["tikhon_history_mrr"] for r in results if r["metrics"].get("tikhon_history_mrr") is not None]
    tikhon_base = (f" Без эмбеддинга «с кем общался раньше» даёт {np.mean(hist):.3f} — у Тихона это бейзлайн history."
                   if hist else "")
    by_day = (f"""
## По дням

Score на каждом test-окне отдельно. Хороший метод держится на всех днях, а не только на соседнем с обучением.

{scoreboard.window_scores_table(results, labels)}""" if multi else "")
    text = f"""# Таблица результатов

Каждый метод описывает компьютер в сети набором чисел — вектором. Таблица показывает, насколько эти векторы полезны.
{data}, компьютеров: {results[0].get('n_hosts')}. {plan}

**Score** — итог от 0 до 100: 0 — как у случайных векторов, 100 — идеал. Места — по нему.
Если нет хотя бы одной из четырёх задач, вместо score — incomplete и места нет.
Название метода — ссылка на его карточку: устройство, какие данные берёт, как обучается.

{scoreboard.headline_table(results, card_url=CARD_URL.format, stale=stale)}{stale_text}
| колонка | что проверяет (больше — лучше) |
|---|---|
| Neighbors@10 | у компьютеров с похожими векторами похожее поведение в сети |
| Links, AUC | по двум векторам можно сказать, будут ли компьютеры обмениваться трафиком |
| New partners, MRR | по похожим компьютерам угадываются их новые собеседники |
| Pairs, AUC | компьютеры с общими собеседниками получают близкие векторы |
| Identity, Top-1 | компьютер узнаётся по вектору, построенному по более позднему трафику |
| Tikhon LP, MRR* | собеседники компьютера в будущем — это компьютеры с близкими векторами (идея 1) |
| Tikhon transfer, MRR* | компьютер будет общаться с теми, с кем раньше общались 10 самых похожих на него (идея 3) |
| Tikhon soft transfer, MRR* | то же, но угадываются компьютеры, похожие на собеседников похожих (идея 3, мягкий вариант) |

\\* Метрики Тихона (#40, PR #65, [`evaluation/tikhon.py`](../evaluation/tikhon.py)): показаны для сравнения, в score не входят.
Chance level для них — случайный порядок кандидатов.{tikhon_base}

Жирным — лучшее в колонке. {ci_text}
dim — длина вектора. Chance level — значение при случайном угадывании.
`random` (случайные векторы) и `reference_ceiling` (точный ответ для Neighbors) — ориентиры, в рейтинге не участвуют.

## Как считается score

Каждая колонка переводится в шкалу 0–100 (0 — случайное угадывание, 100 — идеал).
Колонки объединены в четыре задачи: Behavior (Neighbors), Future links (Links и New partners),
Shared partners (Pairs), Identity. Score — среднее геометрическое четырёх задач:
метод, который провалил хотя бы одну задачу, не поднимется высоко за счёт остальных.{" Для нескольких test-окон каждая колонка — среднее по окнам." if multi else ""}
{by_day}"""
    (BOARD / "README.md").write_text(text, encoding="utf-8")


def _span(seconds: float) -> str:
    return f"{seconds / 3600:g} ч" if seconds >= 3600 and seconds % 1800 == 0 else f"{seconds / 60:g} мин"


def _window_labels() -> tuple[dict[str, str], str]:
    """Подписи test-окон («день 03, 30 мин») и одна фраза о том, на чём учатся и проверяются методы."""
    m = json.loads((DATA / "reference" / "test" / "manifest.json").read_text(encoding="utf-8"))["windows"]
    day = lambda w: f"{int(m[w]['start'] // 86400) + 1:02d}"                      # LANL: день 01 — с 0 по 86400
    span = lambda w: _span(m[w]["end"] - m[w]["start"])
    clock = lambda t: f"{int(t % 86400 // 3600):02d}:{int(t % 3600 // 60):02d}"
    ws = test_windows()
    labels = {w: f"день {day(w)}, {clock(m[w]['start'])}–{clock(m[w]['end'])}" for w in ws}
    if len(ws) == 1:
        plan = f"Вектор строится по {span('train')} трафика и проверяется на следующих {span(ws[0])}."
    else:
        plan = (f"Вектор строится по {span('train')} трафика (день {day('train')}, "
                f"{clock(m['train']['start'])}–{clock(m['train']['end'])}) и проверяется на {len(ws)} отрезках: "
                + "; ".join(labels[w] for w in ws) + ". Итог — среднее по ним.")
    return labels, plan


def _copy_card(name: str, d: Path) -> None:
    """Встроенным методам — их карточка из репозитория (если отправки лежат в другом месте, например в тестах)."""
    src = BOARD_IN_REPO / "submissions" / name / "README.md"
    if src.exists() and not (d / "README.md").exists():
        shutil.copyfile(src, d / "README.md")


BUILTIN_ORDER = ("struc2vec", "node2vec_services", "deepwalk_services", "deepwalk", "node2vec", "graphsage", "rolx")
# варианты графовых методов: граф «хост — хост + хост — сервис», веса подобраны на validation (#79)
VARIANTS_FILE = ROOT / "experiments" / "lanl" / "methods_v2.json"
VARIANT_DIM = 32


def variants() -> dict[str, tuple[str, dict]]:
    """deepwalk_services, node2vec_services: (базовый метод, параметры) из experiments/lanl/methods_v2.json."""
    cfg = json.loads(VARIANTS_FILE.read_text(encoding="utf-8"))
    return {f"{base}_services": (base, params) for base, params in cfg.items()}   # долгие — первыми, чтобы процессы не простаивали


def _code_sha() -> str:
    """Код методов и загрузки данных: поменялся — встроенные методы обучаются заново."""
    files = sorted((ROOT / "methods").rglob("*.py")) + [ROOT / "pipeline" / "data.py"]
    return _sha(*[f for f in files if "test_" not in f.name])


def _build_sha(name: str, dim: int, data_dir: Path) -> str:
    h = hashlib.sha256(f"{name}|{dim}|{SEED}|{_code_sha()}|{variants().get(name)}".encode())
    h.update((Path(data_dir) / "manifest.json").read_bytes())
    return h.hexdigest()


def _build_one(job: tuple) -> str:
    """Обучить один встроенный метод и записать отправку (в отдельном процессе или здесь же)."""
    name, dim, data_dir, out_dir = job
    from methods import graph as graph_methods
    from methods.baselines.temporal import default_baselines
    from pipeline.data import ProcessedData

    data = ProcessedData(data_dir)
    base = {b.name: b for b in default_baselines(dim, SEED)}
    if name in variants():
        parent, params = variants()[name]
        m = graph_methods.create([parent], VARIANT_DIM, SEED, config={parent: params})[0]
    else:
        m = base[name] if name in base else graph_methods.create([name], dim, SEED)[0]
    m.fit(data)
    d = Path(out_dir) / name
    d.mkdir(parents=True, exist_ok=True)
    _copy_card(name, d)
    keep_vectors_local(d)
    m.embed(data, "train").to_csv(d / "embeddings.csv", index_label="host_id")
    for w in data.test_windows:
        m.embed(data, w, data.evaluable(w)).to_csv(d / window_file(w), index_label="host_id")
    if "validation" in data.windows:                    # только для подбора на validation; в git не кладётся
        m.embed(data, "validation", data.evaluable("validation")).to_csv(d / "validation.csv", index_label="host_id")
    shared = d / "SHARED_COORDINATES"
    if m.info().get("shared_coordinate_system"):
        shared.touch()
    elif shared.exists():
        shared.unlink()
    (d / ".build").write_text(_build_sha(name, dim, data_dir) + "\n")
    return name


def _train_gb(data_dir: Path) -> float:
    """Сколько памяти нужно одному процессу обучения: ≈ 3 ГБ на час LANL (9.4 млн потоков в train)."""
    m = json.loads((Path(data_dir) / "manifest.json").read_text(encoding="utf-8"))
    return max(1.0, 3.0 * m["windows"]["train"].get("n_flows", 9.4e6) / 9.4e6)


def builtin(names: list[str], dim: int = 16, jobs: int = 1, force: bool = False) -> None:
    """Обучить бейзлайны и графовые методы на общих данных и положить их как обычные отправки.

    Метод, уже обученный на этих же данных этим же кодом (файл .build в его папке), пропускается, если не force.
    jobs > 1 — методы обучаются параллельно в отдельных процессах.
    """
    from methods import graph as graph_methods
    from methods.baselines.temporal import default_baselines
    from pipeline import speed

    data_dir = processed_dir()
    base = [b.name for b in default_baselines(dim, SEED)]
    wanted = base + sorted(graph_methods.available()) + sorted(variants()) if names == ["all"] else names
    todo = []
    for name in wanted:
        stamp = SUBMISSIONS / name / ".build"
        if not force and stamp.exists() and stamp.read_text().strip() == _build_sha(name, dim, data_dir) \
                and (SUBMISSIONS / name / "embeddings.csv").exists():
            print("уже обучен на этих данных, пропускаю:", name)
            continue
        todo.append(name)
    todo.sort(key=lambda n: BUILTIN_ORDER.index(n) if n in BUILTIN_ORDER else len(BUILTIN_ORDER))
    jobs_list = [(n, dim, data_dir, SUBMISSIONS) for n in todo]
    for job, _, secs in speed.run_parallel(_build_one, jobs_list, jobs, "обучение"):
        print(f"готово: {job[0]} за {secs:.0f} с")
    if names == ["all"]:                                # потолок: признаки эталона как эмбеддинг
        d = SUBMISSIONS / "reference_ceiling"
        d.mkdir(parents=True, exist_ok=True)
        _copy_card("reference_ceiling", d)
        keep_vectors_local(d)
        for w in test_windows():                         # у каждого окна свой потолок — его же признаки
            f = load_embeddings(DATA / "reference" / w / "features.csv", "host_id", "ref_")
            f.columns = [c.replace("ref_", "emb_") for c in f.columns]
            f.to_csv(d / ("embeddings.csv" if w == "test" else f"embeddings_{w}.csv"), index_label="host_id")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("submission", nargs="*", type=Path, help="папки scoreboard/submissions/<метод>")
    ap.add_argument("--all", action="store_true", help="оценить все отправки без свежего результата")
    ap.add_argument("--force", action="store_true", help="пересчитать даже свежие (например, после смены данных)")
    ap.add_argument("--builtin", nargs="+", metavar="METHOD", help="'all' или имена бейзлайнов / графовых методов")
    ap.add_argument("--jobs", type=int, default=None,
                    help="сколько процессов (по умолчанию — по ядрам и памяти: на M2 с 8 ГБ и часе train — 2 на обучение, 4 на оценку)")
    a = ap.parse_args()
    from pipeline import speed
    if a.builtin:
        builtin(a.builtin, jobs=a.jobs or speed.default_jobs(gb_per_job=_train_gb(processed_dir())), force=a.force)
    todo = list(a.submission)
    if a.all or a.builtin:
        todo += sorted(d for d in SUBMISSIONS.iterdir() if d.is_dir() and (d / "embeddings.csv").exists())
    todo = [d for d in dict.fromkeys(todo) if a.force or a.submission.count(d) or not is_fresh(d)]
    for d, r, secs in speed.run_parallel(score, todo, a.jobs or speed.default_jobs(gb_per_job=1.5), "оценка"):
        print(f"оценено за {secs:.0f} с: {d.name}  " + ", ".join(f"{s['title']}: {s['value']:.3f}" for s in r["summary"]
                                                             if s["value"] is not None))
    render()
    if (BOARD / "README.md").exists():
        print((BOARD / "README.md").read_text(encoding="utf-8").split("**Жирным**")[0])


if __name__ == "__main__":
    main()
