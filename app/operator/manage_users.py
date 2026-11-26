"""Provision a hosted analyst using the database secret and an interactive password."""

import argparse
import os
from getpass import getpass
from pathlib import Path

from .database import DatabaseAccounts, connect_database


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("username")
    args = parser.parse_args()
    password = getpass()
    if getpass("Confirm password: ") != password:
        parser.error("Passwords do not match.")
    engine = connect_database(Path(os.environ["PT_DATABASE_URL_FILE"]).read_text().strip())
    try:
        DatabaseAccounts(engine).add(args.username, password)
    finally:
        engine.dispose()


if __name__ == "__main__":
    main()
