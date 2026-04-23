"""Pure-Python Express middleware simulator.

Middleware is stored as JS text and evaluated via pattern matching —
never executed. Three recognised patterns cover the defender's toolkit.
"""
from __future__ import annotations

import re

_VALID_ROUTES = frozenset({"/login", "/api/data", "/api/process", "/api/admin", "/health"})

# Each pattern both detects the middleware type AND extracts parameters.
_BODY_FIELD_RE = re.compile(r"req\.body\.(\w+)\s*===?\s*['\"]([^'\"]+)['\"]")
_HEADER_RE = re.compile(r"req\.headers\[['\"]([^'\"]+)['\"]\]")
_IP_RE = re.compile(r"req\.ip\s*===?\s*['\"]([^'\"]+)['\"]")


def _is_recognized(js: str) -> bool:
    return bool(_BODY_FIELD_RE.search(js) or _HEADER_RE.search(js) or _IP_RE.search(js))


def _evaluate(js: str, req: dict) -> int | None:
    """Pattern-match and evaluate one JS middleware string against a request.

    Args:
        js: JS source string.
        req: Request dict.

    Returns:
        HTTP status code if the middleware fires, None otherwise.
    """
    m = _BODY_FIELD_RE.search(js)
    if m:
        field, value = m.group(1), m.group(2)
        if req.get("body", {}).get(field) == value:
            return 403
        return None

    m = _HEADER_RE.search(js)
    if m:
        header = m.group(1)
        if req.get("headers", {}).get(header):
            return 403
        return None

    m = _IP_RE.search(js)
    if m:
        ip = m.group(1)
        if req.get("ip") == ip:
            return 403
        return None

    return None


class SimulatedExpress:
    """Stateful Express middleware engine for the SRE Arena simulator.

    Middleware is stored as JS source text and evaluated by pattern matching.
    Unrecognised middleware is rejected (returns False from add_middleware)
    and counted for rubric penalty scoring.

    Routes: /login, /api/data, /api/process, /api/admin, /health.
    Last ``add_middleware`` call for a route wins (overwrites previous).
    """

    def __init__(self) -> None:
        self._middleware: dict[str, str] = {}  # route -> latest JS
        self._unrecognized_count: int = 0

    # ── public API ────────────────────────────────────────────────────────────

    def add_middleware(self, route: str, js: str) -> bool:
        """Store middleware for a route if it matches a recognised pattern.

        Args:
            route: Express route path, e.g. ``"/api/process"``.
            js: JS middleware source string.

        Returns:
            True if accepted and stored. False if route is invalid or
            the JS is unrecognised (unrecognised writes increment the
            penalty counter regardless).
        """
        if route not in _VALID_ROUTES:
            return False
        if not _is_recognized(js):
            self._unrecognized_count += 1
            return False
        self._middleware[route] = js
        return True

    def process_request(self, req: dict) -> int:
        """Evaluate one request through stored middleware.

        Args:
            req: Request dict with ``ip``, ``path``, ``body``, ``headers`` keys.

        Returns:
            403 if middleware fires, 200 otherwise.
        """
        route = req.get("path", "/")
        js = self._middleware.get(route)
        if js is not None:
            status = _evaluate(js, req)
            if status is not None:
                return status
        return 200

    def get_middleware_summary(self) -> dict[str, str]:
        """Return mapping of route → JS source for all stored middleware."""
        return dict(self._middleware)

    @property
    def unrecognized_count(self) -> int:
        """Cumulative count of unrecognised middleware writes this episode."""
        return self._unrecognized_count
