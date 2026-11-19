"""Serve a disposable account and database for the browser test."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from app.operator.accounts import Accounts  # noqa: E402
from app.operator.web import create_app  # noqa: E402
import uvicorn  # noqa: E402


with TemporaryDirectory(prefix="pt-lab-browser-") as directory:
    Accounts(Path(directory)).add("browser-test", os.environ["PT_LAB_TEST_PASSWORD"])
    app = create_app(Path(directory), "http://127.0.0.1:18845")
    uvicorn.run(app, host="127.0.0.1", port=18845, access_log=False)
