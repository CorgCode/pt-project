"""Изменения между двумя окнами: сходство состояния, смена группы, новые контакты.

Для трафика оба окна приводятся к ОДНОЙ шкале (log1p и максимум по объединению) —
иначе рост объёмов выглядел бы как изменение профиля. Группы обучаются на первом окне,
второе раскладывается по тем же центрам: так номер группы сохраняет смысл.
"""
from __future__ import annotations

import numpy as np

from hostquery import groups
from hostquery.errors import HostQueryError
from hostquery.explain import _contacts
from hostquery.store import Dataset


def _number(value):
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    return value


def _prepared(before: Dataset, after: Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if before.kind != after.kind:
        raise HostQueryError("окна разного формата: сравнивать трафик с готовыми векторами нельзя")
    if before.dim != after.dim:
        raise HostQueryError(f"размерности окон различаются ({before.dim} и {after.dim})")
    if before.kind == "flows":
        if before.log_values is None or after.log_values is None:
            raise HostQueryError("в окнах трафика нет промежуточных значений для общей шкалы")
        joined = np.maximum(before.log_values.max(axis=0), after.log_values.max(axis=0))
        scales = np.maximum(1.0, joined)
        left, right = before.log_values / scales, after.log_values / scales
    else:
        left, right = before.vectors, after.vectors
    return left, np.linalg.norm(left, axis=1), right, np.linalg.norm(right, axis=1)


def _align(left: np.ndarray, right: np.ndarray, common: list[str], before: Dataset, after: Dataset,
           alignment: str) -> np.ndarray:
    if alignment == "none":
        return right
    if alignment != "procrustes":
        raise HostQueryError("alignment должен быть 'none' или 'procrustes'")
    from scipy.linalg import orthogonal_procrustes

    before_pos = {host: index for index, host in enumerate(before.ids)}
    after_pos = {host: index for index, host in enumerate(after.ids)}
    li = [before_pos[host] for host in common]
    ri = [after_pos[host] for host in common]
    a, b = left[li], right[ri]
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    rotation, _ = orthogonal_procrustes(b - mu_b, a - mu_a)
    return (right - mu_b) @ rotation + mu_a


def compare(before: Dataset, after: Dataset, *, k: int = 3, alignment: str = "none") -> dict:
    if alignment not in {"none", "procrustes"}:
        raise HostQueryError("alignment должен быть 'none' или 'procrustes'")
    left, left_norms, right, right_norms = _prepared(before, after)
    common = [host for host in after.ids if host in set(before.ids)]
    if not common:
        raise HostQueryError("в окнах нет общих хостов",
                             hint="для сравнения нужны те же устройства в обоих окнах")
    if alignment == "procrustes":
        right = _align(left, right, common, before, after, alignment)
        right_norms = np.linalg.norm(right, axis=1)

    left_unit = left / left_norms[:, None]
    right_unit = right / right_norms[:, None]
    fitted = groups.fit_unit(left_unit, k)
    # Как в core.js: центры берутся только для непустых меток, а номер группы
    # остаётся исходным. Иначе пустой кластер сдвинул бы нумерацию второго окна.
    present = sorted(set(fitted["labels"]))
    centers = np.asarray([fitted["centers"][label] for label in present], dtype=float)
    assigned = groups.assign_unit(right_unit, centers)
    after_labels = [present[int(index)] for index in assigned]

    left_pos = {host: i for i, host in enumerate(before.ids)}
    right_pos = {host: i for i, host in enumerate(after.ids)}
    before_contacts = _contacts(before) if before.kind == "flows" else {}
    after_contacts = _contacts(after) if after.kind == "flows" else {}

    def host_item(dataset: Dataset, index: int, vector: np.ndarray, norm: float) -> dict:
        item = {"id": dataset.ids[index],
                "v": [float(x) for x in vector[index]],
                "norm": float(norm[index])}
        if dataset.kind == "flows":
            item["logValues"] = [float(x) for x in dataset.log_values[index]]
            item["activity"] = {key: _number(value)
                                for key, value in dataset.activity[dataset.ids[index]].items()}
        return item

    changes = []
    for host in common:
        i, j = left_pos[host], right_pos[host]
        similarity = float(np.clip(left_unit[i] @ right_unit[j], -1.0, 1.0))
        item = {
            "id": host,
            "similarity": similarity,
            "drift": float(1.0 - similarity),
            "before": host_item(before, i, left, left_norms),
            "after": host_item(after, j, right, right_norms),
            "beforeGroup": int(fitted["labels"][i]),
            "afterGroup": int(after_labels[j]),
        }
        if before.kind == "flows":
            old, new = before_contacts.get(host, []), after_contacts.get(host, [])
            item["added"] = [other for other in new if other not in set(old)]
            item["removed"] = [other for other in old if other not in set(new)]
            item["beforeContacts"] = old
            item["afterContacts"] = new
        changes.append(item)
    changes.sort(key=lambda item: (item["similarity"], item["id"]))

    warnings = []
    if alignment == "procrustes":
        warnings.append("координаты второго окна повёрнуты к первому по общим хостам (Procrustes)")
    if before.kind == "vectors":
        warnings.append("готовые векторы не содержат записей обмена: новые контакты не показать")
    return {
        "kind": before.kind,
        "changes": changes,
        "newHosts": [host for host in after.ids if host not in left_pos],
        "missingHosts": [host for host in before.ids if host not in right_pos],
        "alignment": alignment,
        "k": int(fitted["k"]),
        "n_common": len(common),
        "warnings": warnings,
    }
