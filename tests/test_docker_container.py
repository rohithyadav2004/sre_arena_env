"""Phase 4 Docker container integration tests.

Builds server/Dockerfile from the project root, runs the container on host port
8767 (avoids collision with Phase 2's 8765 and Phase 3's 8766), and exercises
the same HTTP endpoints as Phases 2 and 3 but through Docker's network layer.

Run manually with:
    pytest tests/test_docker_container.py -v

Skipped automatically when Docker daemon is unavailable.
"""
from __future__ import annotations

import asyncio
import json
import random
import subprocess
import time
from pathlib import Path

import httpx
import httpx_sse
import pytest

from sre_arena_env.server.sre_arena_env_environment import SreArenaEnvironment
from sre_arena_env.models import DefenderAction
from sre_arena_env.client import SreArenaEnvClient

# ── Constants ─────────────────────────────────────────────────────────────────

_IMAGE_TAG = "sre-arena-env:test"
_CONTAINER_NAME = "sre-arena-test"
_HOST_PORT = 8767
_BASE_HTTP = f"http://127.0.0.1:{_HOST_PORT}"
_BASE_WS = f"ws://127.0.0.1:{_HOST_PORT}"
_PROJECT_ROOT = Path(__file__).parent.parent  # tests/ -> sre_arena_env/

pytestmark = pytest.mark.docker

# ── Action helper ─────────────────────────────────────────────────────────────

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


# ── Skip guard ────────────────────────────────────────────────────────────────

def _docker_available() -> bool:
    try:
        result = subprocess.run(
            ["docker", "version"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


@pytest.fixture(scope="session", autouse=True)
def require_docker() -> None:  # type: ignore[return]
    if not _docker_available():
        pytest.skip("Docker not available — skipping all docker tests")


# ── Container fixture ─────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def docker_container(require_docker: None) -> str:
    """Build image, start container, wait for health, yield HTTP base URL, teardown."""
    # Build image from project root
    subprocess.run(
        ["docker", "build", "-t", _IMAGE_TAG, "-f", "server/Dockerfile", "."],
        cwd=_PROJECT_ROOT,
        check=True,
        timeout=300,
    )

    # Remove any leftover container from a previous interrupted run
    subprocess.run(
        ["docker", "rm", "-f", _CONTAINER_NAME],
        cwd=_PROJECT_ROOT,
        capture_output=True,
    )

    # Start container
    subprocess.run(
        [
            "docker", "run", "-d",
            "-p", f"{_HOST_PORT}:8000",
            "--name", _CONTAINER_NAME,
            _IMAGE_TAG,
        ],
        check=True,
        timeout=30,
    )

    try:
        # Wait for HTTP health (up to 30 s)
        deadline = time.monotonic() + 30.0
        healthy = False
        while time.monotonic() < deadline:
            try:
                r = httpx.get(f"{_BASE_HTTP}/", timeout=1.0)
                if r.status_code == 200:
                    healthy = True
                    break
            except Exception:
                pass
            time.sleep(0.5)

        if not healthy:
            logs = subprocess.run(
                ["docker", "logs", _CONTAINER_NAME],
                capture_output=True, text=True,
            )
            pytest.fail(f"Container not healthy within 30 s.\nLogs:\n{logs.stdout}\n{logs.stderr}")

        yield _BASE_HTTP

    finally:
        # Teardown — runs even if health check or test raises
        subprocess.run(["docker", "stop", _CONTAINER_NAME], timeout=15, capture_output=True)
        subprocess.run(["docker", "rm", _CONTAINER_NAME], timeout=10, capture_output=True)


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_dashboard_html(docker_container: str) -> None:
    """GET / returns 200 with dashboard HTML containing expected markers."""
    r = httpx.get(f"{docker_container}/", timeout=5.0)
    assert r.status_code == 200
    assert "SRE Arena" in r.text
    assert "<script>" in r.text
    assert "EventSource" in r.text


def test_subscriber_count(docker_container: str) -> None:
    """subscriber-count endpoint returns zero when no SSE clients are connected."""
    r = httpx.get(f"{docker_container}/dashboard/subscriber-count", timeout=5.0)
    assert r.status_code == 200
    assert r.json() == {"count": 0}


def test_whoami(docker_container: str) -> None:
    """Container process runs as appuser (UID 1000), not root."""
    result = subprocess.run(
        ["docker", "exec", _CONTAINER_NAME, "whoami"],
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "appuser", (
        f"Expected 'appuser', got '{result.stdout.strip()}'"
    )


def test_dashboard_ownership(docker_container: str) -> None:
    """Files under /app are owned by appuser."""
    result = subprocess.run(
        ["docker", "exec", _CONTAINER_NAME, "ls", "-la", "/app/"],
        capture_output=True, text=True, timeout=10,
    )
    assert "appuser" in result.stdout, (
        f"Expected appuser ownership in /app/ listing:\n{result.stdout}"
    )


def test_wheel_install_not_editable(docker_container: str) -> None:
    """sse module resolves to site-packages, confirming wheel (not editable) install."""
    result = subprocess.run(
        [
            "docker", "exec", _CONTAINER_NAME,
            "python", "-c",
            "from sre_arena_env.server.dashboard import sse; print(sse.__file__)",
        ],
        capture_output=True, text=True, timeout=10,
    )
    path = result.stdout.strip()
    assert result.returncode == 0, f"Import failed:\n{result.stderr}"
    assert "site-packages" in path, (
        f"Expected site-packages path (wheel install), got: {path}"
    )
    assert not path.startswith("/app/"), (
        f"Path is under /app/ — editable install leaked into container: {path}"
    )


@pytest.mark.asyncio
async def test_seeded_rollout_matches_inprocess(docker_container: str) -> None:
    """10-step seeded defender rollout via container == in-process env (same seed)."""
    # In-process reference
    env = SreArenaEnvironment()
    env.reset(role="defender", task_id="task1", seed=42)
    rng = random.Random(42)
    local_rewards = [env.step(_action(rng)).reward for _ in range(10)]

    # Container rollout
    container_rewards: list[float] = []
    ws_url = docker_container.replace("http://", "ws://")
    async with SreArenaEnvClient(ws_url) as client:
        await client.reset(role="defender", task_id="task1", seed=42)
        rng = random.Random(42)
        for _ in range(10):
            result = await client.step(_action(rng))
            container_rewards.append(result.observation.reward)

    assert len(container_rewards) == 10
    for i, (local_r, ctr_r) in enumerate(zip(local_rewards, container_rewards)):
        assert abs(local_r - ctr_r) < 1e-6, (
            f"Step {i + 1}: in-process={local_r:.6f}  container={ctr_r:.6f}  "
            f"diff={abs(local_r - ctr_r):.2e}"
        )


@pytest.mark.asyncio
async def test_dashboard_sse_end_to_end_through_docker(docker_container: str) -> None:
    """SSE events from the containerised server arrive during a live rollout.

    Asserts >= 1 reset event and >= 5 step events, confirming the end-to-end
    SSE path works through Docker's network layer (not just localhost uvicorn).
    """
    collected: list[dict] = []

    async with httpx.AsyncClient(base_url=docker_container, timeout=30.0) as http:

        async def _collect() -> None:
            async with httpx_sse.aconnect_sse(http, "GET", "/dashboard/events") as src:
                async for sse in src.aiter_sse():
                    collected.append(json.loads(sse.data))

        collect_task = asyncio.create_task(_collect())

        # Subscriber-count handshake: wait until server registers our SSE client
        loop = asyncio.get_running_loop()
        deadline = loop.time() + 5.0
        while loop.time() < deadline:
            resp = await http.get("/dashboard/subscriber-count")
            if resp.status_code == 200 and resp.json().get("count", 0) >= 1:
                break
            await asyncio.sleep(0.05)
        else:
            collect_task.cancel()
            pytest.fail("SSE subscriber count never reached 1 within 5 s")

        # 5-step defender rollout against the same container
        ws_url = docker_container.replace("http://", "ws://")
        async with SreArenaEnvClient(ws_url) as client:
            await client.reset(role="defender", task_id="task1", seed=42)
            rng = random.Random(42)
            for _ in range(5):
                await client.step(_action(rng))

        await asyncio.sleep(0.5)  # let events flush through SSE pipeline

        collect_task.cancel()
        try:
            await collect_task
        except asyncio.CancelledError:
            pass

    reset_events = [e for e in collected if e.get("type") == "reset"]
    step_events = [e for e in collected if e.get("type") == "step"]

    assert len(reset_events) >= 1, (
        f"Expected >= 1 reset event, got {len(reset_events)}. "
        f"All events: {[e.get('type') for e in collected]}"
    )
    for ev in step_events:
        assert "reward" in ev, f"step event missing reward field: {ev}"
    assert len(step_events) >= 5, (
        f"Expected >= 5 step events, got {len(step_events)}. "
        f"All events: {[e.get('type') for e in collected]}"
    )
