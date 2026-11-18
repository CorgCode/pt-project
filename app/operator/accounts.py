"""Local lab accounts and revocable sessions."""
import hashlib
import hmac
import re
import secrets
import sqlite3
import time
from contextlib import closing

SESSION_TTL = 12 * 60 * 60
COOKIE = "corgcode_session"


def password_hash(password, salt):
    return hashlib.scrypt(password.encode(), salt=salt, n=2**15, r=8, p=1, maxmem=64 * 1024 * 1024)


class Accounts:
    def __init__(self, directory):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.path = directory / "accounts.sqlite3"
        with closing(self.connect()) as db, db:
            db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    id TEXT PRIMARY KEY, name TEXT NOT NULL UNIQUE,
                    salt BLOB NOT NULL, verifier BLOB NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sessions (
                    token BLOB PRIMARY KEY, user_id TEXT NOT NULL, expires INTEGER NOT NULL
                );
            """)

    def connect(self):
        return sqlite3.connect(self.path, timeout=10)

    def add(self, name, password):
        name = name.strip().lower()
        if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]{2,63}", name):
            raise ValueError("Имя: 3–64 латинских символа, цифры, точка, дефис или подчёркивание.")
        if not 12 <= len(password) <= 256:
            raise ValueError("Пароль должен содержать от 12 до 256 символов.")
        salt = secrets.token_bytes(16)
        with closing(self.connect()) as db, db:
            try:
                db.execute("INSERT INTO users VALUES (?, ?, ?, ?)",
                           (secrets.token_hex(16), name, salt, password_hash(password, salt)))
            except sqlite3.IntegrityError as error:
                raise ValueError("Такое имя уже занято.") from error

    def login(self, name, password):
        if len(password) > 256:
            return None
        with closing(self.connect()) as db, db:
            row = db.execute("SELECT id, salt, verifier FROM users WHERE name=?",
                             (name.strip().lower(),)).fetchone()
            verifier = password_hash(password, row[1] if row else bytes(16))
            if not row or not hmac.compare_digest(verifier, row[2]):
                return None
            token = secrets.token_urlsafe(32)
            now = int(time.time())
            db.execute("DELETE FROM sessions WHERE expires<=?", (now,))
            db.execute("INSERT INTO sessions VALUES (?, ?, ?)",
                       (self.digest(token), row[0], now + SESSION_TTL))
            return token

    @staticmethod
    def digest(token):
        return hashlib.sha256(token.encode()).digest()

    def identify(self, token):
        if not token:
            return None
        with closing(self.connect()) as db:
            row = db.execute("SELECT user_id FROM sessions WHERE token=? AND expires>?",
                             (self.digest(token), int(time.time()))).fetchone()
        return row[0] if row else None

    def logout(self, token):
        if token:
            with closing(self.connect()) as db, db:
                db.execute("DELETE FROM sessions WHERE token=?", (self.digest(token),))
