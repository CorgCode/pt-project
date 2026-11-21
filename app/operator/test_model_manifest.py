import hashlib
import json

import pytest

from deploy.inference.service import verified_artifacts


def test_verified_artifacts(tmp_path):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"weights")
    manifest = tmp_path / "manifest.json"

    def write_manifest(name):
        manifest.write_text(json.dumps({"version": "v1", "files": {name: hashlib.sha256(b"weights").hexdigest()}}))
        return hashlib.sha256(manifest.read_bytes()).hexdigest()

    digest = write_manifest(checkpoint.name)
    assert verified_artifacts(tmp_path, "v1", digest) == {checkpoint.name: checkpoint}
    with pytest.raises(ValueError, match="manifest"):
        verified_artifacts(tmp_path, "v1", "0" * 64)
    with pytest.raises(ValueError, match="version"):
        verified_artifacts(tmp_path, "v2", digest)
    checkpoint.write_bytes(b"changed")
    with pytest.raises(ValueError, match="Artifact SHA"):
        verified_artifacts(tmp_path, "v1", digest)
    with pytest.raises(ValueError, match="Invalid artifact"):
        verified_artifacts(tmp_path, "v1", write_manifest("../checkpoint.bin"))
