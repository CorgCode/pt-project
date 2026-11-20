"""Dataset metadata and inference job records, scoped to the authenticated owner."""

from datetime import datetime, timezone
from uuid import uuid4

from fastapi import Request
from sqlalchemy import Column, ForeignKey, String, Table, insert, select, update

from .database import metadata, workspaces

jobs = Table(
    "jobs", metadata,
    Column("id", String(36), primary_key=True),
    Column("owner_id", ForeignKey("users.id", ondelete="CASCADE"), nullable=False, index=True),
    Column("status", String(16), nullable=False),
    Column("created", String(40), nullable=False),
    Column("finished", String(40)),
)


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def start_job(engine, owner):
    job = str(uuid4())
    with engine.begin() as db:
        db.execute(insert(jobs).values(id=job, owner_id=owner, created=timestamp(), status="running"))
    return job


def finish_job(engine, job, status):
    with engine.begin() as db:
        db.execute(update(jobs).where(jobs.c.id == job).values(status=status, finished=timestamp()))


def add_routes(app, engine, user, error):
    @app.get("/api/datasets")
    def datasets(request: Request):
        owner = user(request)
        if not owner:
            return error(401, "Войдите, чтобы открыть данные.")
        with engine.connect() as db:
            return [dict(row) for row in db.execute(
                select(workspaces.c.id, workspaces.c.name, workspaces.c.updated)
                .where(workspaces.c.owner_id == owner)
                .order_by(workspaces.c.updated.desc())
                .limit(100)
            ).mappings()]

    @app.get("/api/jobs/{key}")
    def read_job(key: str, request: Request):
        owner = user(request)
        if not owner:
            return error(401, "Войдите, чтобы открыть запуск.")
        if len(key) != 36:
            return error(404, "Запуск не найден.")
        with engine.connect() as db:
            row = db.execute(
                select(jobs.c.id, jobs.c.status, jobs.c.created, jobs.c.finished)
                .where(jobs.c.id == key, jobs.c.owner_id == owner)
            ).mappings().one_or_none()
        return dict(row) if row is not None else error(404, "Запуск не найден.")
