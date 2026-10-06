"""
Оценка Linear / Logistic Probe для host embeddings.

Метод проверяет, насколько хорошо внешний признак хоста
можно восстановить из его embedding с помощью простого
линейного классификатора.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score


@dataclass
class LinearProbeResult:
    """Результаты оценки Linear / Logistic Probe."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    n_train: int
    n_test: int
    n_classes: int

    def __str__(self) -> str:
        return (
            f"Linear / Logistic Probe\n"
            f"  Accuracy: {self.accuracy:.4f}\n"
            f"  Macro F1: {self.macro_f1:.4f}\n"
            f"  Weighted F1: {self.weighted_f1:.4f}\n"
            f"  Объектов в train: {self.n_train}\n"
            f"  Объектов в test: {self.n_test}\n"
            f"Количество классов: {self.n_classes}"
        )


class LinearProbeEvaluator:
    """
    Evaluator для Linear / Logistic Probe.

    Embeddings считаются замороженными: классификатор обучается
    поверх уже готовых embedding и не изменяет сами embedding.

    Параметры
    ---------
    max_iter : int, default=1000
        Максимальное количество итераций для LogisticRegression.
    """

    def __init__(self, max_iter: int = 1000):
        self.max_iter = max_iter
        self.embeddings: Optional[Dict[str, np.ndarray]] = None

    def load_embeddings(self, embeddings_path: str) -> None:
        """
        Загружает embeddings из CSV.

        Формат:

            host_id,emb_0,emb_1,emb_2
            C1,0.12,-0.81,0.44
            C2,0.09,-0.77,0.39
        """

        path = Path(embeddings_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Файл с embeddings не найден: {path}"
            )

        self.embeddings = {}

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(
                    "Файл с embeddings пуст или не содержит заголовка."
                )

            if "host_id" not in reader.fieldnames:
                raise ValueError(
                    "Файл с embeddings должен содержать столбец 'host_id'."
                )

            embedding_columns = [
                column
                for column in reader.fieldnames
                if column.startswith("emb_")
            ]

            if not embedding_columns:
                raise ValueError(
                    "Файл с embeddings должен содержать столбцы, "
                    "начинающиеся с 'emb_'."
                )

            for row in reader:
                host_id = row["host_id"].strip()

                if not host_id:
                    raise ValueError(
                        "Обнаружен хост с пустым host_id."
                    )

                if host_id in self.embeddings:
                    raise ValueError(
                        f"Обнаружен дублирующийся host_id: {host_id}"
                    )

                try:
                    embedding = np.array(
                        [float(row[column]) for column in embedding_columns],
                        dtype=np.float32,
                    )
                except (ValueError, TypeError):
                    raise ValueError(
                        f"Некорректное числовое значение в embedding "
                        f"для хоста {host_id}."
                    )

                if not np.isfinite(embedding).all():
                    raise ValueError(
                        f"Embedding хоста {host_id} содержит NaN или Inf."
                    )

                self.embeddings[host_id] = embedding

    def evaluate(self, labels_path: str) -> LinearProbeResult:
        """
        Обучает Logistic Regression на train и оценивает её на test.

        Формат labels.csv:

            host_id,target,split
            C1,workstation,train
            C2,server,train
            C3,workstation,test
            C4,server,test
        """

        if self.embeddings is None:
            raise RuntimeError(
                "Сначала необходимо загрузить embeddings "
                "с помощью load_embeddings()."
            )

        path = Path(labels_path)

        if not path.exists():
            raise FileNotFoundError(
                f"Файл с reference labels не найден: {path}"
            )

        train_x = []
        train_y = []
        test_x = []
        test_y = []

        seen_hosts = set()

        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)

            if reader.fieldnames is None:
                raise ValueError(
                    "Файл с labels пуст или не содержит заголовка."
                )

            required_columns = {"host_id", "target", "split"}

            if not required_columns.issubset(reader.fieldnames):
                missing = required_columns - set(reader.fieldnames)

                raise ValueError(
                    "В файле labels отсутствуют обязательные столбцы: "
                    f"{', '.join(sorted(missing))}."
                )

            for row in reader:
                host_id = row["host_id"].strip()
                target = row["target"].strip()
                split = row["split"].strip()

                if not host_id:
                    raise ValueError(
                        "В labels.csv обнаружен пустой host_id."
                    )

                if host_id in seen_hosts:
                    raise ValueError(
                        f"В labels.csv обнаружен дублирующийся "
                        f"host_id: {host_id}"
                    )

                seen_hosts.add(host_id)

                if not target:
                    raise ValueError(
                        f"Для хоста {host_id} отсутствует target."
                    )

                if split not in {"train", "test"}:
                    raise ValueError(
                        f"Для хоста {host_id} указан неизвестный split "
                        f"'{split}'. Ожидаются 'train' или 'test'."
                    )

                if host_id not in self.embeddings:
                    continue

                embedding = self.embeddings[host_id]

                if split == "train":
                    train_x.append(embedding)
                    train_y.append(target)

                else:
                    test_x.append(embedding)
                    test_y.append(target)

        if not train_x:
            raise ValueError(
                "Не найдено ни одного объекта для обучения "
                "(split='train')."
            )

        if not test_x:
            raise ValueError(
                "Не найдено ни одного объекта для проверки "
                "(split='test')."
            )

        if len(set(train_y)) < 2:
            raise ValueError(
                "В train должно быть как минимум два разных класса target."
            )

        X_train = np.stack(train_x)
        X_test = np.stack(test_x)

        classifier = LogisticRegression(
            max_iter=self.max_iter,
            random_state=0,
        )

        classifier.fit(X_train, train_y)

        predictions = classifier.predict(X_test)

        accuracy = accuracy_score(test_y, predictions)

        macro_f1 = f1_score(
            test_y,
            predictions,
            average="macro",
            zero_division=0,
        )

        weighted_f1 = f1_score(
            test_y,
            predictions,
            average="weighted",
            zero_division=0,
        )

        return LinearProbeResult(
            accuracy=float(accuracy),
            macro_f1=float(macro_f1),
            weighted_f1=float(weighted_f1),
            n_train=len(train_y),
            n_test=len(test_y),
            n_classes=len(set(train_y)),
        )


def run_evaluation(
    embeddings_path: str,
    labels_path: str,
    max_iter: int = 1000,
) -> LinearProbeResult:
    """
    Удобная функция для запуска оценки.
    """

    evaluator = LinearProbeEvaluator(max_iter=max_iter)
    evaluator.load_embeddings(embeddings_path)

    return evaluator.evaluate(labels_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Оценка host embeddings с помощью Linear / Logistic Probe."
    )

    parser.add_argument(
        "embeddings",
        help="Путь к CSV с embeddings.",
    )

    parser.add_argument(
        "labels",
        help="Путь к CSV с reference labels.",
    )

    args = parser.parse_args()

    try:
        result = run_evaluation(
            args.embeddings,
            args.labels,
        )
        print(result)

    except (FileNotFoundError, ValueError, RuntimeError) as exc:
        print(f"Ошибка: {exc}")
        raise SystemExit(1)
