"""Ошибка запроса: сообщение и подсказка, которые можно показать в интерфейсе."""
from __future__ import annotations


class HostQueryError(ValueError):
    """Запрос нельзя выполнить; текст рассчитан и на человека, и на UI."""

    def __init__(self, message: str, hint: str | None = None):
        super().__init__(message)
        self.message = message
        self.hint = hint

    def as_dict(self) -> dict:
        out = {"error": self.message}
        if self.hint:
            out["hint"] = self.hint
        return out

    def __str__(self) -> str:  # pragma: no cover - короткая форма для CLI
        return f"{self.message} ({self.hint})" if self.hint else self.message
