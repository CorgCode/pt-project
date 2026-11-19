"""Проверка, что уборка не маскирует изменение или потерю научных артефактов."""
import hashlib
import json

from tools.check_repo import reference_errors


def test_changed_or_missing_reference_is_rejected(tmp_path):
    folder = tmp_path / "scoreboard/data/reference/test"
    folder.mkdir(parents=True)
    data = folder / "features.csv"
    data.write_bytes(b"original")
    (folder / "manifest.json").write_text(json.dumps({
        "files_sha256": {"features.csv": hashlib.sha256(data.read_bytes()).hexdigest()}}))
    assert reference_errors(tmp_path)[0] == []
    data.write_bytes(b"changed")
    assert any("хеш не совпадает" in error for error in reference_errors(tmp_path)[0])
    data.unlink()
    assert any("нет файла" in error for error in reference_errors(tmp_path)[0])


def test_manifest_cannot_refer_outside_its_directory(tmp_path):
    folder = tmp_path / "scoreboard/data/reference/test"
    folder.mkdir(parents=True)
    (folder / "manifest.json").write_text(json.dumps({"files_sha256": {"../outside.csv": "x"}}))
    assert any("недопустимое имя" in error for error in reference_errors(tmp_path)[0])
