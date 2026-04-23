from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import AsyncIterator

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from sse_starlette import EventSourceResponse

_log = logging.getLogger(__name__)

router = APIRouter()
_DASHBOARD_DIR = Path(__file__).parent

_subscribers: list[asyncio.Queue] = []
_loop_ref: asyncio.AbstractEventLoop | None = None


def set_loop(loop: asyncio.AbstractEventLoop) -> None:
    """Store the running event loop reference. Called from app startup handler."""
    global _loop_ref
    _loop_ref = loop


async def publish(event: dict) -> None:
    """Broadcast event to all connected SSE subscribers. Never raises."""
    for q in list(_subscribers):
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            try:
                q.get_nowait()
                q.put_nowait(event)
            except (asyncio.QueueEmpty, asyncio.QueueFull):
                pass


def fire_publish(event: dict) -> None:
    """Thread-safe fire-and-forget publish. Safe to call from sync env code running
    in a thread executor. Silently drops if no loop is set or loop is closed."""
    if _loop_ref is None or not _loop_ref.is_running():
        return
    coro = publish(event)
    try:
        asyncio.run_coroutine_threadsafe(coro, _loop_ref)
    except Exception:
        coro.close()  # prevent "coroutine never awaited" warning
        _log.debug("dashboard publish dropped", exc_info=True)


@router.get("/dashboard/subscriber-count")
async def subscriber_count() -> dict:
    return {"count": len(_subscribers)}


@router.get("/")
async def serve_dashboard() -> FileResponse:
    """Serve the single-file SOC dashboard."""
    return FileResponse(_DASHBOARD_DIR / "index.html", media_type="text/html")


@router.get("/dashboard/events")
async def sse_stream(request: Request) -> EventSourceResponse:
    """SSE stream of env activity. One event per env.step() call."""
    q: asyncio.Queue = asyncio.Queue(maxsize=100)
    _subscribers.append(q)

    async def _gen() -> AsyncIterator[dict]:
        try:
            while True:
                try:
                    event = await asyncio.wait_for(q.get(), timeout=30.0)
                except asyncio.TimeoutError:
                    if await request.is_disconnected():
                        break
                    continue
                yield {"event": event["type"], "data": json.dumps(event)}
        finally:
            if q in _subscribers:
                _subscribers.remove(q)

    return EventSourceResponse(_gen(), ping=15)
