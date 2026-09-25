"""Веб-сервис для SOC-оператора: исследование хостов, кластеров и аномалий.

Читает артефакты пайплайна (JSON) и отдаёт их через REST API + статический UI.
Если артефактов нет — подсказывает, как их сгенерировать.
"""
from __future__ import annotations

import json
import os

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

ARTIFACTS_DIR = os.environ.get("ARTIFACTS_DIR", "artifacts")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(title="host2vec — векторное представление хостов", version="0.1.0")


def _load(name: str) -> dict:
    path = os.path.join(ARTIFACTS_DIR, name)
    if not os.path.exists(path):
        raise HTTPException(
            status_code=503,
            detail=f"Артефакт {name} не найден. Запустите пайплайн: "
                   f"python -m hostembed.datagen --out data && python -m hostembed.pipeline")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/summary")
def summary():
    return _load("summary.json")


@app.get("/api/hosts")
def hosts():
    return _load("hosts.json")


@app.get("/api/host/{device}")
def host(device: str):
    data = _load("hosts.json")["hosts"]
    for h in data:
        if h["device"] == device:
            return h
    raise HTTPException(status_code=404, detail=f"Хост {device} не найден")


@app.get("/api/clusters")
def clusters():
    return _load("clusters.json")


@app.get("/api/anomalies")
def anomalies():
    return _load("anomalies.json")


@app.get("/api/health")
def health():
    ready = os.path.exists(os.path.join(ARTIFACTS_DIR, "summary.json"))
    return JSONResponse({"status": "ok", "artifacts_ready": ready})


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
