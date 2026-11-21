"""Low-cardinality request metrics and JSON request logs."""

import hmac
import json
import logging
from time import monotonic

from fastapi import Request, Response
from prometheus_client import CONTENT_TYPE_LATEST, CollectorRegistry, Counter, Histogram, generate_latest


def instrument(app, token):
    if not token.isascii() or len(token) < 32:
        raise ValueError("Metrics token must contain at least 32 ASCII characters")
    authorization = ("Bearer " + token).encode()
    registry = CollectorRegistry()
    requests = Counter("pt_http_requests_total", "HTTP requests", ("route", "status"), registry=registry)
    latency = Histogram("pt_http_request_seconds", "HTTP latency", ("route",), registry=registry)
    logger = logging.getLogger("uvicorn.error")

    @app.middleware("http")
    async def observe(request, call_next):
        start = monotonic()
        status = 500
        try:
            response = await call_next(request)
            status = response.status_code
            return response
        finally:
            route = getattr(request.scope.get("route"), "path", "unmatched")
            if route != "/internal/metrics":
                elapsed = monotonic() - start
                requests.labels(route, status).inc()
                latency.labels(route).observe(elapsed)
                logger.info(json.dumps({"event": "http_request", "route": route,
                                        "status": status, "seconds": round(elapsed, 4)}))

    @app.get("/internal/metrics")
    def metrics(request: Request):
        provided = request.headers.get("authorization", "").encode()
        if not hmac.compare_digest(provided, authorization):
            return Response(status_code=404)
        return Response(generate_latest(registry), headers={"Content-Type": CONTENT_TYPE_LATEST})
