"""Shared helpers for method tests (not a conftest: imported explicitly)."""
from pipeline.data import ProcessedData


class TrainOnlyGuard(ProcessedData):
    """Raises if fitting touches any window but train (flows or graph)."""
    fitting = False

    def flows(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} window")
        return super().flows(window)

    def graph(self, window):
        if self.fitting and window != "train":
            raise AssertionError(f"fit() read the {window} graph")
        return super().graph(window)
