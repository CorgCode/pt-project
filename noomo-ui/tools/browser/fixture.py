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
    data_dir = Path(directory)
    database_url = os.getenv("PT_TEST_DATABASE_URL")
    origin = "http://127.0.0.1:18845"
    if database_url:
        from alembic import command
        from alembic.config import Config
        from app.operator.database import DatabaseAccounts, connect_database
        import socket
        import subprocess
        from time import monotonic, sleep
        from urllib.request import urlopen

        secret = data_dir / "url"
        secret.write_text(database_url)
        os.environ["PT_DATABASE_URL_FILE"] = str(secret)
        repo = Path(__file__).resolve().parents[3]
        config = Config(repo / "deploy/alembic.ini")
        config.set_main_option("prepend_sys_path", str(repo))
        command.upgrade(config, "head")
        engine = connect_database(database_url)
        DatabaseAccounts(engine).add("browser-test", os.environ["PT_LAB_TEST_PASSWORD"])
        env = {**os.environ, "PT_MODEL_VERSION": "traffic-profile-v1", "OPENBLAS_NUM_THREADS": "1"}
        with socket.socket() as listener:
            listener.bind(("127.0.0.1", 0))
            port = listener.getsockname()[1]
            with subprocess.Popen(
                [sys.executable, "-m", "uvicorn", "deploy.inference.service:app",
                 "--fd", str(listener.fileno()), "--no-access-log"],
                env=env, cwd=repo, pass_fds=(listener.fileno(),),
            ) as upstream:
                try:
                    deadline = monotonic() + 10
                    while True:
                        try:
                            with urlopen(f"http://127.0.0.1:{port}/ready", timeout=0.5):
                                break
                        except OSError:
                            if upstream.poll() is not None or monotonic() > deadline:
                                raise
                            sleep(0.05)
                    app = create_app(data_dir, origin, f"http://127.0.0.1:{port}/v1/predict", engine=engine)
                    uvicorn.run(app, host="127.0.0.1", port=18845, access_log=False)
                finally:
                    upstream.terminate()
    else:
        Accounts(data_dir).add("browser-test", os.environ["PT_LAB_TEST_PASSWORD"])
        app = create_app(data_dir, origin)
        uvicorn.run(app, host="127.0.0.1", port=18845, access_log=False)
