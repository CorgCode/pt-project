"""Inference API for a trusted, image-bundled adapter and verified artifacts."""

import asyncio
import hashlib
import hmac
import importlib
import json
import os
import re
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import JSONResponse

from deploy.inference.adapter import (
    ADAPTER_API, MAX_EMBEDDING_RESPONSE, csv_digest, input_csv, validate_response,
)
from hostquery.schema import SCHEMA, envelope

MAX_BODY = 8 * 1024 * 1024


def verified_artifacts(root, version, expected_digest):
    root = root.resolve()
    manifest = (root / "manifest.json").read_bytes()
    if not hmac.compare_digest(hashlib.sha256(manifest).hexdigest(), expected_digest):
        raise ValueError("Artifact manifest SHA-256 mismatch")
    data = json.loads(manifest)
    if data["version"] != version or not isinstance(data["files"], dict) or not data["files"]:
        raise ValueError("Wrong version or empty artifact manifest")
    result = {}
    for name, expected in data["files"].items():
        file = root / name
        if (not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected)
                or not file.resolve().is_relative_to(root) or not file.is_file()):
            raise ValueError("Invalid artifact entry")
        with file.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        if not hmac.compare_digest(actual, expected):
            raise ValueError("Artifact SHA-256 mismatch")
        result[name] = file
    return result


@asynccontextmanager
async def lifespan(app):
    version = os.getenv("PT_MODEL_VERSION", "traffic-profile-v1")
    module = os.getenv("PT_MODEL_ADAPTER", "deploy.inference.traffic_profile")
    digest = os.getenv("PT_MODEL_MANIFEST_SHA256")
    if digest:
        artifacts = verified_artifacts(Path(os.environ["PT_MODEL_DIR"]), version, digest)
    elif module == "deploy.inference.traffic_profile":
        artifacts = {}
    else:
        raise ValueError("External adapters require a verified artifact manifest")
    adapter = importlib.import_module(module).load(artifacts, version)
    if adapter.metadata.get("model_version") != version:
        raise ValueError("Adapter model version differs from configuration")
    app.state.adapter = adapter
    app.state.manifest_digest = digest
    app.state.version = version
    app.state.lock = asyncio.Lock()
    yield


app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)


@app.get("/ready")
async def ready():
    return {"status": "ready", "version": app.state.version}


@app.post("/v1/predict")
async def predict(request: Request):
    if request.headers.get("content-type", "").split(";")[0] != "application/json":
        return JSONResponse({"error": "Expected JSON object"}, status_code=415)
    if app.state.lock.locked():
        return JSONResponse({"error": "Model busy"}, status_code=503)
    async with app.state.lock:
        body = bytearray()
        async for chunk in request.stream():
            if len(body) + len(chunk) > MAX_BODY:
                return JSONResponse({"error": "Request too large"}, status_code=413)
            body.extend(chunk)
        try:
            csv = input_csv(json.loads(body))
        except (ValueError, RecursionError):
            return JSONResponse({"error": "Invalid CSV input"}, status_code=422)
        adapter = app.state.adapter
        try:
            prepared = await run_in_threadpool(adapter.preprocess, csv)
        except (ValueError, OverflowError, RecursionError):
            return JSONResponse({"error": "Invalid CSV input"}, status_code=422)

        def infer():
            items = adapter.infer(prepared)
            adapter.validate_output(prepared, items)
            digest = csv_digest(csv)
            provenance = {**adapter.metadata, "schema": SCHEMA, "source": "csv", "synthetic": False,
                          "adapter_api": ADAPTER_API, "artifact_manifest_sha256": app.state.manifest_digest,
                          "input_sha256": digest, "hosts": len(items)}
            result = envelope("dataset", provenance=provenance, dataset=items, kind="vectors",
                              query={"method": provenance["method"]},
                              stats={"hosts": len(items), "dimensions": provenance["dimensions"]})
            return validate_response(result, digest)

        try:
            result = await run_in_threadpool(infer)
        except (TypeError, ValueError, OverflowError, KeyError):
            return JSONResponse({"error": "Invalid model output"}, status_code=502)
        response = JSONResponse(result)
        if len(response.body) > MAX_EMBEDDING_RESPONSE:
            return JSONResponse({"error": "Response too large"}, status_code=413)
        return response
