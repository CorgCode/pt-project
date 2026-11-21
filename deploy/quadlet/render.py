"""Render Podman 4.9 Quadlets and the Nginx vhost from an explicit environment."""

import os
import re
import sys
from pathlib import Path

here = Path(__file__).resolve().parent


def render(environment, output):
    if environment not in ("staging", "production"):
        raise ValueError("Use staging or production")
    prefix = f"pt-{environment}"
    state = Path.home() / ".local/share" / prefix
    state.mkdir(mode=0o700, parents=True, exist_ok=True)
    for directory in ("models", "datasets", "results"):
        (state / directory).mkdir(mode=0o700, exist_ok=True)

    def value(key, pattern):
        result = os.environ[key]
        if not re.fullmatch(pattern, result):
            raise ValueError(f"Invalid {key}")
        return result

    image = r"[a-zA-Z0-9._:/-]+@sha256:[0-9a-f]{64}"
    host = value("PT_HOST", r"[a-zA-Z0-9-]+(?:\.[a-zA-Z0-9-]+)+")
    port = int(value("PT_API_PORT", r"[0-9]{4,5}"))
    if not 1024 <= port <= 65535:
        raise ValueError("Invalid loopback API port")
    values = {
        "PREFIX": prefix, "STATE": str(state), "HOST": host,
        "PUBLIC_ORIGIN": "https://" + host, "API_PORT": str(port),
        "API_IMAGE": value("PT_API_IMAGE", image),
        "POSTGRES_IMAGE": value("PT_POSTGRES_IMAGE", image),
        "CERT": value("PT_CERT", r"/[a-zA-Z0-9._/-]+"),
        "CERT_KEY": value("PT_CERT_KEY", r"/[a-zA-Z0-9._/-]+"),
        "STATIC_ROOT": value("PT_STATIC_ROOT", r"/[a-zA-Z0-9._/-]+"),
        "ALLOW_REGISTRATION": os.getenv("PT_ALLOW_REGISTRATION", "0"),
        "INFERENCE_ENV": "",
    }
    if values["ALLOW_REGISTRATION"] not in {"0", "1"}:
        raise ValueError("PT_ALLOW_REGISTRATION must be 0 or 1")
    inference = bool(os.getenv("PT_INFERENCE_IMAGE"))
    if inference:
        values.update({
            "INFERENCE_IMAGE": value("PT_INFERENCE_IMAGE", image),
            "MODEL_MANIFEST_SHA256": os.getenv("PT_MODEL_MANIFEST_SHA256", ""),
            "MODEL_VERSION": os.getenv("PT_MODEL_VERSION", "traffic-profile-v1"),
            "MODEL_MEMORY": value("PT_MODEL_MEMORY", r"[1-9][0-9]{0,4}[mg]"),
            "MODEL_CPUS": value("PT_MODEL_CPUS", r"[1-9][0-9]?(?:\.[0-9])?"),
            "INFERENCE_ENV": f"Environment=PT_INFERENCE_URL=http://{prefix}-inference:8001/v1/predict",
        })

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9._-]{0,63}", values["MODEL_VERSION"]):
            raise ValueError("Invalid PT_MODEL_VERSION")
        if not re.fullmatch(r"(?:[0-9a-f]{64})?", values["MODEL_MANIFEST_SHA256"]):
            raise ValueError("Invalid PT_MODEL_MANIFEST_SHA256")

    output.mkdir(parents=True, exist_ok=True)
    for template in sorted(here.iterdir()):
        if template.name.startswith(".") or template.suffix not in {".container", ".volume", ".network"}:
            continue
        if template.name == "inference.container" and not inference:
            continue
        content = re.sub(r"@([A-Z_0-9]+)@", lambda m: values[m[1]], template.read_text())
        (output / (prefix + "-" + template.name)).write_text(content)
    template = here.parent / "nginx" / "pt.conf"
    (output / (prefix + ".conf")).write_text(
        re.sub(r"@([A-Z_0-9]+)@", lambda m: values[m[1]], template.read_text()))
    (state / "init-postgres.sh").write_bytes((here.parent / "init-postgres.sh").read_bytes())
    (state / "init-postgres.sh").chmod(0o755)


if __name__ == "__main__":
    render(sys.argv[1], Path(sys.argv[2]))
