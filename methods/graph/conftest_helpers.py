"""Общие помощники для тестов методов (не conftest: импортируются явно)."""
from pipeline.data import ProcessedData


class TrainOnlyGuard(ProcessedData):
    """Ошибка, если обучение трогает любое окно, кроме train (потоки или граф)."""
    fitting = False

    def flows(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} window")
        return super().flows(window)

    def graph(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} graph")
        return super().graph(window)
