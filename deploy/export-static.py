"""Build the Nginx document root from the same image revision as the API."""

import shutil
import sys
from pathlib import Path

from app.operator.web import FILES, PAGES, SITE, STATIC_DIRS

destination = Path(sys.argv[1])
destination.mkdir(parents=True, exist_ok=True)
for directory in STATIC_DIRS:
    shutil.copytree(
        SITE / directory, destination / directory, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".*", "*.md", "*.map"),
    )
paths = {**FILES, **PAGES, "/lab/index.html": SITE / "lab/index.html",
         "/product/index.html": SITE / "product/index.html"}
for route, source in paths.items():
    target = destination / route.lstrip("/")
    if route.endswith("/"):
        target /= "index.html"
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
