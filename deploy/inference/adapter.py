"""Versioned adapter boundary shared by the gateway and model service."""

import hashlib
import math
import re
from typing import Protocol

from hostquery.schema import MAX_CSV_BYTES, MAX_CSV_DIMENSIONS, MAX_CSV_HOSTS, SCHEMA

ADAPTER_API = "pt-embedding-adapter/1"
MAX_EMBEDDING_RESPONSE = 16 * 1024 * 1024


class Adapter(Protocol):
    metadata: dict

    def preprocess(self, csv: str): ...

    def infer(self, prepared) -> list[dict]: ...

    def validate_output(self, prepared, items: list[dict]) -> None: ...


def csv_digest(csv):
    return hashlib.sha256(csv.encode("utf-8")).hexdigest()


def input_csv(payload):
    csv = payload.get("csv") if isinstance(payload, dict) else None
    if not isinstance(csv, str) or not csv.strip() or len(csv.encode("utf-8")) > MAX_CSV_BYTES:
        raise ValueError("Expected a CSV up to 3 MiB")
    return csv


def validate_response(result, digest):
    if not isinstance(result, dict) or result.get("schema") != SCHEMA:
        raise ValueError("Invalid hostquery envelope")
    source = result.get("provenance")
    if (not isinstance(source, dict) or source.get("input_sha256") != digest
            or source.get("adapter_api") != ADAPTER_API or source.get("source") != "csv"
            or source.get("synthetic") is not False):
        raise ValueError("Invalid model provenance")
    artifact = source.get("artifact_manifest_sha256")
    if artifact is not None and (not isinstance(artifact, str) or not re.fullmatch(r"[0-9a-f]{64}", artifact)):
        raise ValueError("Invalid artifact provenance")
    for key in ("method", "model_version"):
        value = source.get(key)
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", value):
            raise ValueError("Invalid model metadata")
    if result.get("query") != {"kind": "dataset", "method": source["method"]}:
        raise ValueError("Invalid dataset query")
    dimension = source.get("dimensions")
    items = result.get("dataset")
    if (type(dimension) is not int or not 1 <= dimension <= MAX_CSV_DIMENSIONS
            or not isinstance(items, list) or not 2 <= len(items) <= MAX_CSV_HOSTS):
        raise ValueError("Invalid embedding shape")
    if (source.get("hosts") != len(items) or result.get("kind") != "vectors"
            or result.get("stats") != {"hosts": len(items), "dimensions": dimension}
            or result.get("warnings") != []):
        raise ValueError("Invalid dataset metadata")
    seen = set()
    for item in items:
        if not isinstance(item, dict) or set(item) != {"id", "v", "norm"}:
            raise ValueError("Invalid embedding row")
        host, vector, norm = item["id"], item["v"], item["norm"]
        if (not isinstance(host, str) or not host or len(host) > 500 or host in seen
                or not isinstance(vector, list) or len(vector) != dimension
                or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)
                or type(norm) not in (int, float) or not math.isfinite(norm) or norm <= 0
                or not math.isclose(norm, math.hypot(*vector), rel_tol=1e-8)):
            raise ValueError("Invalid embedding row")
        seen.add(host)
    return result
