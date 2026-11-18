"""Проверка основных документов и целостности опубликованных эталонов; только stdlib."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import subprocess
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    "README.md", "PROJECT_STATUS.md", "CONTRIBUTING.md", "docs/README.md",
    "docs/METHODOLOGY.md", "docs/REPRODUCIBILITY.md",
    "docs/ARTIFACTS.md", "scoreboard/submissions/multiday_codebook/README.md",
    "scoreboard/submissions/multiday_codebook_plus/README.md",
)


def reference_errors(root: Path) -> tuple[list[str], set[Path]]:
    errors, legacy = [], set()
    manifests = sorted((root / "scoreboard/data/reference").glob("*/manifest.json"))
    if not manifests:
        errors.append("нет манифестов общей таблицы")
    for manifest in manifests:
        try:
            hashes = json.loads(manifest.read_text(encoding="utf-8"))["files_sha256"]
            if not hashes:
                raise ValueError("пустой files_sha256")
            for name, expected in hashes.items():
                if Path(name).name != name or name in {".", ".."}:
                    errors.append(f"{manifest.relative_to(root)}: недопустимое имя {name!r}")
                    continue
                file = manifest.parent / name
                if not file.is_file():
                    errors.append(f"нет файла эталона: {file.relative_to(root)}")
                    continue
                actual = hashlib.sha256(file.read_bytes()).hexdigest()
                if actual != expected:
                    errors.append(f"хеш не совпадает: {file.relative_to(root)}")
                if name.startswith("._"):
                    legacy.add(file.relative_to(root))
        except (ValueError, KeyError, TypeError, OSError) as error:
            errors.append(f"{manifest.relative_to(root)}: {error}")
    return errors, legacy


def link_errors(root: Path, documents=DOCUMENTS) -> list[str]:
    errors = []
    for name in documents:
        doc = root / name
        if not doc.is_file():
            errors.append(f"нет документа: {name}")
            continue
        text = doc.read_text(encoding="utf-8")
        for target in re.findall(r"\[[^\]]*\]\(([^\s)]+)\)", text):
            url = urlsplit(target.strip("<>"))
            if url.scheme or url.netloc or not url.path:
                continue
            path = unquote(url.path)
            resolved = root / path.lstrip("/") if path.startswith("/") else doc.parent / path
            if not resolved.exists():
                errors.append(f"{name}: битая ссылка {target}")
    return errors


def main() -> int:
    errors, legacy = reference_errors(ROOT)
    tracked = subprocess.run(["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"], cwd=ROOT,
                             capture_output=True, check=True).stdout.decode().split("\0")
    documents = set(DOCUMENTS) | {p for p in tracked if p.endswith(".md") and (ROOT / p).is_file()}
    errors += link_errors(ROOT, sorted(documents))
    for name in filter(None, tracked):
        path = Path(name)
        if not (ROOT / path).is_file():
            continue  # удаление, ещё не записанное в индекс; манифесты проверены выше
        if any(p == ".DS_Store" or p.startswith("._") or p == "__pycache__" for p in path.parts):
            if path not in legacy:
                errors.append(f"служебный файл в Git: {name}")
    for error in errors:
        print("ERROR:", error)
    if errors:
        return 1
    print(f"OK: документы, ссылки и хеши эталонов; {len(legacy)} исторических AppleDouble в манифестах")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
