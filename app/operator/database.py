"""SQLAlchemy persistence for the hosted operator application."""

import hmac
import re
import secrets
import time
import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    JSON, BigInteger, Column, ForeignKey, LargeBinary, MetaData, String, Table,
    and_, create_engine, delete, insert, select, update,
)
from sqlalchemy.exc import IntegrityError

from .accounts import Accounts, SESSION_TTL, password_hash
from .server import validate

metadata = MetaData()
users = Table(
    "users", metadata,
    Column("id", String(32), primary_key=True),
    Column("name", String(64), nullable=False, unique=True),
    Column("salt", LargeBinary, nullable=False),
    Column("verifier", LargeBinary, nullable=False),
    Column("role", String(16), nullable=False, server_default="analyst"),
)
sessions = Table(
    "sessions", metadata,
    Column("token", LargeBinary, primary_key=True),
    Column("user_id", ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
    Column("expires", BigInteger, nullable=False, index=True),
)
workspaces = Table(
    "workspaces", metadata,
    Column("owner_id", ForeignKey("users.id", ondelete="CASCADE"), primary_key=True),
    Column("id", String(36), primary_key=True),
    Column("name", String(120), nullable=False),
    Column("payload", JSON, nullable=False),
    Column("updated", String(40), nullable=False),
)


def connect_database(url):
    return create_engine(url, pool_pre_ping=True)


class DatabaseAccounts:
    def __init__(self, engine):
        self.engine = engine

    def add(self, name, password):
        name = name.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", name):
            raise ValueError("Имя: 3–64 латинских символа, цифры, точка, дефис или подчёркивание.")
        if not 8 <= len(password) <= 256:
            raise ValueError("Пароль должен содержать от 8 до 256 символов.")
        salt = secrets.token_bytes(16)
        verifier = password_hash(password, salt)
        try:
            with self.engine.begin() as db:
                db.execute(insert(users).values(id=secrets.token_hex(16), name=name,
                                                salt=salt, verifier=verifier))
        except IntegrityError as error:
            raise ValueError("Такое имя уже занято.") from error

    def login(self, name, password):
        if len(password) > 256 or len(name) > 256:
            return None
        with self.engine.connect() as db:
            row = db.execute(select(users.c.id, users.c.salt, users.c.verifier).where(
                users.c.name == name.strip().lower())).one_or_none()
        verifier = password_hash(password, row.salt if row else bytes(16))
        if not row or not hmac.compare_digest(verifier, row.verifier):
            return None
        token = secrets.token_urlsafe(32)
        now = int(time.time())
        with self.engine.begin() as db:
            db.execute(delete(sessions).where(sessions.c.expires <= now))
            db.execute(insert(sessions).values(token=Accounts.digest(token), user_id=row.id,
                                               expires=now + SESSION_TTL))
        return token

    def identify(self, token):
        if not token or len(token) != 43:
            return None
        with self.engine.connect() as db:
            return db.execute(select(sessions.c.user_id).join(users).where(
                sessions.c.token == Accounts.digest(token),
                sessions.c.expires > int(time.time()),
                users.c.role.in_(("analyst", "admin")),
            )).scalar_one_or_none()

    def logout(self, token):
        if token and len(token) == 43:
            with self.engine.begin() as db:
                db.execute(delete(sessions).where(sessions.c.token == Accounts.digest(token)))


class DatabaseWorkspaces:
    def __init__(self, engine, owner):
        self.engine = engine
        self.owner = owner

    def owned(self, key):
        return and_(workspaces.c.owner_id == self.owner, workspaces.c.id == key)

    def list(self):
        with self.engine.connect() as db:
            return [dict(row) for row in db.execute(
                select(workspaces.c.id, workspaces.c.name, workspaces.c.updated)
                .where(workspaces.c.owner_id == self.owner)
                .order_by(workspaces.c.updated.desc())
            ).mappings()]

    def get(self, key):
        if len(key) > 36:
            return None
        with self.engine.connect() as db:
            return db.execute(select(workspaces.c.payload).where(self.owned(key))).scalar_one_or_none()

    def save(self, payload, key=None):
        payload = validate(payload)
        values = dict(name=payload["name"], payload=payload,
                      updated=datetime.now(timezone.utc).isoformat())
        with self.engine.begin() as db:
            if key is None:
                key = str(uuid.uuid4())
                db.execute(insert(workspaces).values(owner_id=self.owner, id=key, **values))
            elif not db.execute(update(workspaces).where(self.owned(key)).values(**values)).rowcount:
                return None
        return key


def check_database(engine):
    from .resources import jobs

    with engine.connect() as db:
        db.execute(select(users.c.id).limit(0))
        db.execute(select(sessions.c.token).limit(0))
        db.execute(select(workspaces.c.id).limit(0))
        db.execute(select(jobs.c.id).limit(0))
