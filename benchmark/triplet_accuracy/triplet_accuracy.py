import csv
from pathlib import Path
from typing import Dict, Optional
from dataclasses import dataclass

import numpy as np
from scipy.spatial.distance import cdist


@dataclass
class TripletAccuracyResult:
    """Результаты оценки triplet accuracy."""
    overall_accuracy: float
    accuracies_by_split: Dict[str, float]
    total_triplets: int
    correct_predictions: int
    split_stats: Dict[str, Dict[str, float | int]]  # total: int, accuracy: float
    
    def __str__(self) -> str:
        lines = [
            f"Triplet Accuracy: {self.overall_accuracy:.4f}",
            f"Correct: {self.correct_predictions}/{self.total_triplets}",
        ]
        for split in sorted(self.split_stats.keys()):
            stats = self.split_stats[split]
            lines.append(
                f"  {split}: accuracy={stats['accuracy']:.4f}, "
                f"total={stats['total']}"
            )
        return "\n".join(lines)


class TripletAccuracyEvaluator:
    """
    Evaluator для triplet accuracy метрики.
    
    Вычисляет долю triplets, где расстояние(anchor, positive) < 
    расстояние(anchor, negative).
    
    Parameters
    ----------
    metric : str, default='euclidean'
        Метрика расстояния: 'euclidean' или 'cosine'
    splits_to_eval : list[str] | None, default=None
        Какие split'ы считать. Если None, считает все найденные в данных.
    """
    
    def __init__(
        self, 
        metric: str = "euclidean",
        splits_to_eval: Optional[list[str]] = None
    ):
        if metric not in ("euclidean", "cosine"):
            raise ValueError(f"metric должна быть 'euclidean' или 'cosine', получено {metric}")
        self.metric = metric
        self.splits_to_eval = splits_to_eval
        self.embeddings: Optional[Dict[str, np.ndarray]] = None
        
    def load_embeddings(self, embeddings_path: str) -> None:
        """
        Загружает эмбеддинги из CSV.
        
        CSV должен иметь колонки: id, embedding (space-separated)
        
        Parameters
        ----------
        embeddings_path : str
            Путь до CSV с эмбеддингами
        """
        self.embeddings = {}
        embeddings_path = Path(embeddings_path)
        
        if not embeddings_path.exists():
            raise FileNotFoundError(f"Файл не найден: {embeddings_path}")
        
        with open(embeddings_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            if reader.fieldnames is None or "id" not in reader.fieldnames:
                raise ValueError("CSV должен содержать колонку 'id'")
            
            embedding_col = "embedding"
            if embedding_col not in reader.fieldnames:
                raise ValueError(f"CSV должен содержать колонку '{embedding_col}'")
            
            for row in reader:
                obj_id = row["id"].strip()
                embedding_str = row[embedding_col].strip()
                
                try:
                    embedding = np.array(
                        [float(x) for x in embedding_str.split()],
                        dtype=np.float32
                    )
                    self.embeddings[obj_id] = embedding
                except ValueError:
                    continue
    
    def evaluate(self, triplets_path: str) -> TripletAccuracyResult:
        """
        Вычисляет triplet accuracy.
        
        Parameters
        ----------
        triplets_path : str
            Путь до CSV с triplets (колонки: anchor, positive, negative, split)
        
        Returns
        -------
        TripletAccuracyResult
            Объект с результатами
        """
        if self.embeddings is None:
            raise RuntimeError("Сначала загрузите эмбеддинги через load_embeddings()")
        
        triplets_path = Path(triplets_path)
        if not triplets_path.exists():
            raise FileNotFoundError(f"Файл не найден: {triplets_path}")
        
        triplets = self._load_triplets(triplets_path)
        results = self._compute_accuracy(triplets)
        return results
    
    def _load_triplets(self, triplets_path: Path) -> list[Dict[str, str]]:
        """Загружает triplets из CSV."""
        triplets = []
        
        with open(triplets_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            required_cols = {"anchor", "positive", "negative", "split"}
            
            if reader.fieldnames is None:
                raise ValueError("CSV не может быть пуст")
            
            if not required_cols.issubset(set(reader.fieldnames)):
                raise ValueError(
                    f"CSV должен содержать колонки: {required_cols}, "
                    f"получено: {set(reader.fieldnames)}"
                )
            
            for row in reader:
                split = row["split"].strip()
                
                # Фильтруем по выбранным split'ам, если они указаны
                if self.splits_to_eval is not None and split not in self.splits_to_eval:
                    continue
                
                triplets.append({
                    "anchor": row["anchor"].strip(),
                    "positive": row["positive"].strip(),
                    "negative": row["negative"].strip(),
                    "split": split,
                })
        
        return triplets
    
    def _compute_accuracy(self, triplets: list[Dict[str, str]]) -> TripletAccuracyResult:
        """Вычисляет accuracy для набора triplets."""
        split_correct = {}
        split_total = {}
        
        total_correct = 0
        total_count = len(triplets)
        skipped = 0
        
        for triplet in triplets:
            anchor_id = triplet["anchor"]
            positive_id = triplet["positive"]
            negative_id = triplet["negative"]
            split = triplet["split"]
            
            # Проверяем наличие эмбеддингов
            if (anchor_id not in self.embeddings or 
                positive_id not in self.embeddings or 
                negative_id not in self.embeddings):
                skipped += 1
                total_count -= 1
                continue
            
            # Вычисляем расстояния
            anchor_emb = self.embeddings[anchor_id]
            positive_emb = self.embeddings[positive_id]
            negative_emb = self.embeddings[negative_id]
            
            dist_positive = cdist(
                [anchor_emb], [positive_emb], metric=self.metric
            )[0, 0]
            dist_negative = cdist(
                [anchor_emb], [negative_emb], metric=self.metric
            )[0, 0]
            
            # Проверяем условие triplet
            is_correct = dist_positive < dist_negative
            
            if is_correct:
                total_correct += 1
            
            # Собираем статистику по split'ам
            if split not in split_correct:
                split_correct[split] = 0
                split_total[split] = 0
            
            split_correct[split] += int(is_correct)
            split_total[split] += 1
        
        # Формируем результаты
        accuracies_by_split = {
            split: split_correct[split] / split_total[split]
            for split in split_total
        }
        
        split_stats = {
            split: {
                "total": split_total[split],
                "accuracy": split_correct[split] / split_total[split]
            }
            for split in split_total
        }
        
        overall_accuracy = total_correct / total_count if total_count > 0 else 0.0
        
        return TripletAccuracyResult(
            overall_accuracy=overall_accuracy,
            accuracies_by_split=accuracies_by_split,
            total_triplets=total_count,
            correct_predictions=total_correct,
            split_stats=split_stats,
        )


def run_evaluation(
    embeddings_path: str,
    triplets_path: str,
    metric: str = "euclidean",
    splits_to_eval: Optional[list[str]] = None
) -> TripletAccuracyResult:
    """
    Удобная функция для запуска оценки из benchmark'а.
    
    Parameters
    ----------
    embeddings_path : str
        Путь до CSV с эмбеддингами
    triplets_path : str
        Путь до CSV с triplets
    metric : str, default='euclidean'
        Метрика расстояния
    splits_to_eval : list[str] | None, default=None
        Какие split'ы считать (например, ['test'] или ['train', 'val', 'test']).
        Если None, считает все найденные в данных.
    
    Returns
    -------
    TripletAccuracyResult
        Результаты оценки
    """
    evaluator = TripletAccuracyEvaluator(metric=metric, splits_to_eval=splits_to_eval)
    evaluator.load_embeddings(embeddings_path)
    result = evaluator.evaluate(triplets_path)
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 3:
        print("Использование: python evaluator.py <embeddings.csv> <triplets.csv> [metric] [splits]")
        sys.exit(1)
    
    embeddings_file = sys.argv[1]
    triplets_file = sys.argv[2]
    metric = sys.argv[3] if len(sys.argv) > 3 else "euclidean"
    splits = sys.argv[4].split(",") if len(sys.argv) > 4 else None
    
    result = run_evaluation(embeddings_file, triplets_file, metric, splits)
    print(result)
