"""Create per-environment Podman secrets once. Existing values are preserved."""

import secrets
import subprocess
import sys


def provision(environment):
    if environment not in ("production", "staging"):
        raise ValueError("Use staging or production")
    prefix = f"pt-{environment}"
    names = ("pg-admin", "pg-owner", "pg-app", "runtime-url", "migration-url", "metrics")
    existing = [
        name for name in names if subprocess.run(
            ["podman", "secret", "exists", f"{prefix}-{name}"], check=False).returncode == 0
    ]
    if len(existing) == len(names):
        return
    if existing:
        raise SystemExit("Partial secret set. Resolve it before provisioning.")
    admin, owner, runtime, metrics = (secrets.token_hex(32) for _ in range(4))
    host = f"{prefix}-postgres"
    values = (admin, owner, runtime, f"postgresql+psycopg://pt_app:{runtime}@{host}/pt",
              f"postgresql+psycopg://pt_owner:{owner}@{host}/pt", metrics)
    for name, value in zip(names, values, strict=True):
        subprocess.run(["podman", "secret", "create", f"{prefix}-{name}", "-"],
                       input=value.encode(), stdout=subprocess.DEVNULL, check=True)


if __name__ == "__main__":
    provision(sys.argv[1])
