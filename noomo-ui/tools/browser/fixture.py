"""Serve a disposable account and database for the browser test."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.operator.accounts import Accounts  # noqa: E402
from app.operator.server import make_server  # noqa: E402


with TemporaryDirectory(prefix="pt-lab-browser-") as directory:
    Accounts(Path(directory)).add("browser-test", os.environ["PT_LAB_TEST_PASSWORD"])
    with make_server(18845, directory, site=True) as server:
        server.serve_forever()
