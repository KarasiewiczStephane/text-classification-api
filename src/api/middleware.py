"""Request logging middleware for the API."""

import hashlib
import logging
import time

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger("api.requests")


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware that logs prediction requests with latency and text hash."""

    async def dispatch(self, request: Request, call_next):
        """Log request details and response latency.

        Args:
            request: Incoming HTTP request.
            call_next: Next middleware/handler in the chain.

        Returns:
            The HTTP response.
        """
        start_time = time.time()

        body = None
        if request.url.path.startswith("/predict"):
            body = await request.body()
            request._body = body

        response = await call_next(request)

        latency = (time.time() - start_time) * 1000

        if body:
            text_hash = hashlib.md5(body).hexdigest()[:8]  # noqa: S324
            logger.info(
                "path=%s text_hash=%s latency_ms=%.2f status=%d",
                request.url.path,
                text_hash,
                latency,
                response.status_code,
            )

        return response
