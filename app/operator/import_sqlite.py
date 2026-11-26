"""Import a quiesced legacy snapshot into a migrated, empty database."""

import argparse
import json
import os
import re
import sqlite3
from contextlib import closing
from pathlib import Path

from sqlalchemy import func, insert, select

from .database import connect_database, sessions, users, workspaces


def sqlite_rows(path, query):
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as db:
        db.row_factory = sqlite3.Row
        yield from db.execute(query)


def import_snapshot(source, engine):
    account_db = source / "accounts.sqlite3"
    user_count = workspace_count = 0
    with engine.begin() as dest:
        if dest.dialect.name == "postgresql":
            dest.exec_driver_sql("LOCK TABLE users, sessions, workspaces IN SHARE ROW EXCLUSIVE MODE")
        for table in (users, sessions, workspaces):
            if dest.scalar(select(func.count()).select_from(table)):
                raise ValueError("Destination tables must all be empty.")
        for row in sqlite_rows(account_db, "SELECT id, name, salt, verifier FROM users"):
            owner = row["id"]
            if not re.fullmatch(r"[0-9a-f]{32}", owner):
                raise ValueError("Invalid legacy user identifier.")
            dest.execute(insert(users), dict(row))
            user_count += 1
            workspace_db = source / "users" / owner / "workspaces.sqlite3"
            if workspace_db.exists():
                for workspace in sqlite_rows(workspace_db, "SELECT id, name, payload, updated FROM workspaces"):
                    values = dict(workspace)
                    values["payload"] = json.loads(values["payload"])
                    dest.execute(insert(workspaces), dict(values, owner_id=owner))
                    workspace_count += 1
    return {"users": user_count, "workspaces": workspace_count}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Read-only copy of the quiesced legacy data tree")
    args = parser.parse_args()
    engine = connect_database(Path(os.environ["PT_DATABASE_URL_FILE"]).read_text().strip())
    try:
        print(json.dumps(import_snapshot(args.source, engine)))
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
