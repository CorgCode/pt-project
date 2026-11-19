"""Проверить оригинальный checkpoint по SHA без загрузки pickle; только stdlib."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("artifacts", type=Path, help="папка PT_PROJECT_ML")
    args = parser.parse_args()
    path = args.artifacts / "fingerprint_codebook_recent/final/model.pkl"
    metadata = ROOT / "scoreboard/submissions/multiday_codebook/params.json"
    expected = json.loads(metadata.read_text(encoding="utf-8"))["checkpoint_sha256"]
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                digest.update(block)
    except OSError as error:
        print(f"ERROR: {error}")
        return 1
    if digest.hexdigest() != expected:
        print(f"ERROR: SHA-256 не совпадает: {digest.hexdigest()}")
        return 1
    print(f"OK: multiday_codebook, SHA-256 {expected}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
