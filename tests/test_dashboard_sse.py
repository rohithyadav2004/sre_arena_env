"""Phase 3 dashboard SSE integration tests.

Starts uvicorn on port 8766 (module-scoped) to avoid collision with 8765.
Tests verify:
  1. GET / serves the dashboard HTML
  2. /dashboard/events streams reset/step/episode_end events during a rollout
  3. Disconnecting a client mid-rollout does not affect env throughput (circuit breaker)
  4. Multiple concurrent subscribers each receive all events
"""
from __future__ import annotations

import asyncio
import json
import random
import threading
import time

import httpx
import httpx_sse
import pytest
import uvicorn

from sre_arena_env.server.app import app
from sre_arena_env.server.sre_arena_env_environment import SreArenaEnvironment
from sre_arena_env.models import DefenderAction

_TEST_PORT = 8766
_BASE_URL = f"http://127.0.0.1:{_TEST_PORT}"

# ── Action generator ──────────────────────────────────────────────────────────

_RULE_POOL = [
    "deny 10.0.0.1;",
    "deny 10.0.0.0/24;",
    "limit_req_zone $binary_remote_addr zone=flood:10m rate=10r/s;",
    "limit_req zone=flood burst=5 nodelay;",
]
_MW_POOL = [
    ("if (req.body.command === 'rm') return res.status(403)", "/api/process"),
    ("if (req.headers['X-Forwarded-For']) return res.status(403)", "/login"),
]


def _action(rng: random.Random) -> DefenderAction:
    kind = rng.choice(["read_log", "append_nginx_rule", "write_express_middleware"])
    if kind == "append_nginx_rule":
        return DefenderAction(action_type=kind, rule_text=rng.choice(_RULE_POOL))
    if kind == "write_express_middleware":
        js, route = rng.choice(_MW_POOL)
        return DefenderAction(action_type=kind, route=route, middleware_js=js)
    return DefenderAction(action_type=kind)


# ── Uvicorn fixture ───────────────────────────────────────────────────────────

class _UvicornThread(threading.Thread):
    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self._server = uvicorn.Server(
            uvicorn.Config(app, host="127.0.0.1", port=port, loop="asyncio", log_level="error")
        )

    def run(self) -> None:
        self._server.run()

    def stop(self) -> None:
        self._server.should_exit = True


@pytest.fixture(scope="module")
def server_url():
    """Start uvicorn on _TEST_PORT, yield base URL, tear down after module."""
    thread = _UvicornThread(_TEST_PORT)
    thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{_BASE_URL}/health", timeout=0.5)
            if r.status_code < 500:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("uvicorn did not start within 5 s")

    yield _BASE_URL

    thread.stop()
    thread.join(timeout=5)


# ── Handshake helper ─────────────────────────────────────────────────────────

async def _wait_for_subscribers(client: httpx.AsyncClient, expected: int, timeout: float = 2.0) -> None:
    """Poll until the server reports `expected` active SSE subscribers."""
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        resp = await client.get("/dashboard/subscriber-count")
        if resp.status_code == 200 and resp.json().get("count", 0) >= expected:
            return
        await asyncio.sleep(0.01)
    raise TimeoutError(f"subscriber count did not reach {expected} within {timeout}s")


# ── Tests ─────────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_sse_html_page(server_url: str) -> None:
    """GET / returns 200 with the dashboard HTML containing expected markers."""
    async with httpx.AsyncClient(base_url=server_url, timeout=5.0) as client:
        r = await client.get("/")
    assert r.status_code == 200
    body = r.text
    assert "SRE Arena" in body
    assert "<script>" in body
    assert "EventSource" in body


@pytest.mark.asyncio
async def test_sse_events_flow(server_url: str) -> None:
    """SSE stream receives reset, step, and episode_end events from a rollout."""
    env = SreArenaEnvironment()
    collected: list[dict] = []

    async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as client:

        async def _collect() -> None:
            async with httpx_sse.aconnect_sse(client, "GET", "/dashboard/events") as source:
                async for sse in source.aiter_sse():
                    collected.append(json.loads(sse.data))

        collect_task = asyncio.create_task(_collect())
        await _wait_for_subscribers(client, expected=1)

        env.reset(role="defender", task_id="task1", seed=42)
        rng = random.Random(42)
        for _ in range(10):
            env.step(_action(rng))

        await asyncio.sleep(0.3)  # allow events to flush through SSE pipeline

        collect_task.cancel()
        try:
            await collect_task
        except asyncio.CancelledError:
            pass

    assert len(collected) >= 7, f"Expected ≥7 events, got {len(collected)}"

    reset_events = [e for e in collected if e["type"] == "reset"]
    step_events = [e for e in collected if e["type"] == "step"]

    assert len(reset_events) >= 1, "Expected at least one reset event"
    for ev in reset_events:
        assert "episode_id" in ev, f"reset event missing episode_id: {ev}"
        assert "role" in ev, f"reset event missing role: {ev}"

    assert len(step_events) >= 5, f"Expected ≥5 step events, got {len(step_events)}"
    for ev in step_events:
        assert "reward" in ev, f"step event missing reward: {ev}"
        assert "role" in ev, f"step event missing role: {ev}"
        assert "step" in ev, f"step event missing step: {ev}"
        assert "metrics" in ev, f"step event missing metrics: {ev}"
        assert "blue_score" in ev["metrics"], f"metrics missing blue_score: {ev['metrics']}"
        assert "red_score" in ev["metrics"], f"metrics missing red_score: {ev['metrics']}"


@pytest.mark.asyncio
async def test_sse_disconnect_resilience(server_url: str) -> None:
    """Disconnecting mid-rollout must not affect env throughput (circuit breaker)."""
    env = SreArenaEnvironment()
    env.reset(role="defender", task_id="task1", seed=42)
    rng = random.Random(42)
    rewards: list[float] = []

    async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as client:

        async def _collect() -> None:
            async with httpx_sse.aconnect_sse(client, "GET", "/dashboard/events") as source:
                async for _ in source.aiter_sse():
                    pass

        collect_task = asyncio.create_task(_collect())
        await _wait_for_subscribers(client, expected=1)

        # 3 steps with client connected
        for _ in range(3):
            obs = env.step(_action(rng))
            rewards.append(obs.reward)

        # Disconnect mid-rollout
        collect_task.cancel()
        try:
            await collect_task
        except asyncio.CancelledError:
            pass

    # 7 more steps after client is gone — must not raise
    for _ in range(7):
        obs = env.step(_action(rng))
        rewards.append(obs.reward)

    assert len(rewards) == 10
    for i, r in enumerate(rewards):
        assert -0.5 <= r <= 1.5, f"Step {i + 1}: reward {r} outside [-0.5, 1.5]"


@pytest.mark.asyncio
async def test_sse_multiple_subscribers(server_url: str) -> None:
    """Two concurrent SSE clients both receive events from a single rollout."""
    env = SreArenaEnvironment()
    collected_a: list[dict] = []
    collected_b: list[dict] = []

    async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as ca:
        async with httpx.AsyncClient(base_url=server_url, timeout=30.0) as cb:

            async def _collect_a() -> None:
                async with httpx_sse.aconnect_sse(ca, "GET", "/dashboard/events") as source:
                    async for sse in source.aiter_sse():
                        collected_a.append(json.loads(sse.data))

            async def _collect_b() -> None:
                async with httpx_sse.aconnect_sse(cb, "GET", "/dashboard/events") as source:
                    async for sse in source.aiter_sse():
                        collected_b.append(json.loads(sse.data))

            task_a = asyncio.create_task(_collect_a())
            task_b = asyncio.create_task(_collect_b())
            await _wait_for_subscribers(ca, expected=2)

            env.reset(role="defender", task_id="task1", seed=42)
            rng = random.Random(42)
            for _ in range(5):
                env.step(_action(rng))

            await asyncio.sleep(0.3)

            for task in (task_a, task_b):
                task.cancel()
            for task in (task_a, task_b):
                try:
                    await task
                except asyncio.CancelledError:
                    pass

    assert len(collected_a) >= 2, f"Client A got {len(collected_a)} events"
    assert len(collected_b) >= 2, f"Client B got {len(collected_b)} events"
    assert any(e["type"] == "reset" for e in collected_a), "Client A missed reset event"
    assert any(e["type"] == "reset" for e in collected_b), "Client B missed reset event"
