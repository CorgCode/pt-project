"""Serve the PT Network site and its signed-in lab on the same local origin."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.operator.server import main  # noqa: E402


if __name__ == "__main__":
    main(site=True)
