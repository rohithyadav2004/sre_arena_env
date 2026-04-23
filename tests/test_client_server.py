"""Client-server integration tests for Phase 2.

Runs a seeded 50-step rollout against both the in-process env and the HTTP
client (backed by a uvicorn server in a background thread). Asserts that
per-step rewards and final observation fields are numerically identical.

Design notes:
- Both envs receive seed=42 (or seed=99) via reset(), so they generate
  identical per-step traffic → identical rewards without any monkeypatching.
- uvicorn is started once per module on port 8765 (module-scoped fixture).
- Action generators are identical to test_env_rollout.py for direct comparison.
"""
from __future__ import annotations

import random
import threading
import time

import httpx
import pytest
import uvicorn

from sre_arena_env.server.app import app
from sre_arena_env.server.sre_arena_env_environment import SreArenaEnvironment
from sre_arena_env.models import AttackerAction, DefenderAction
from sre_arena_env.client import SreArenaEnvClient

# ── Action generators (identical to test_env_rollout.py) ─────────────────────

_TEMPLATES = [
    "single_ip_flood", "ip_spray", "credential_stuffing", "payload_injection",
    "header_spoof", "slow_drip", "path_traversal", "mixed_legit_cover",
]
_DEFENDER_RULE_POOL = [
    "deny 10.0.0.1;",
    "deny 10.0.0.2;",
    "deny 10.0.0.0/24;",
    "limit_req_zone $binary_remote_addr zone=flood:10m rate=10r/s;",
    "limit_req zone=flood burst=5 nodelay;",
    "allow 192.168.1.1;",
]
_DEFENDER_MIDDLEWARE_POOL = [
    ("if (req.body.command === 'rm') return res.status(403)", "/api/process"),
    ("if (req.headers['X-Forwarded-For']) return res.status(403)", "/login"),
    ("if (req.ip === '10.0.0.1') return res.status(403)", "/api/admin"),
    ("console.log('bad middleware');", "/api/process"),
]
_TEST_PORT = 8765


def _random_defender_action(rng: random.Random) -> DefenderAction:
    action_type = rng.choice(["read_log", "append_nginx_rule", "write_express_middleware"])
    if action_type == "append_nginx_rule":
        return DefenderAction(action_type=action_type, rule_text=rng.choice(_DEFENDER_RULE_POOL))
    if action_type == "write_express_middleware":
        js, route = rng.choice(_DEFENDER_MIDDLEWARE_POOL)
        return DefenderAction(action_type=action_type, route=route, middleware_js=js)
    return DefenderAction(action_type=action_type)


def _random_attacker_action(rng: random.Random) -> AttackerAction:
    return AttackerAction(
        template=rng.choice(_TEMPLATES),
        count=rng.randint(5, 20),
        target_path=rng.choice(["/login", "/api/data", "/api/process", "/api/admin"]),
    )


# ── Server fixture ────────────────────────────────────────────────────────────

class _UvicornThread(threading.Thread):
    def __init__(self, port: int) -> None:
        super().__init__(daemon=True)
        self._config = uvicorn.Config(
            app,
            host="127.0.0.1",
            port=port,
            loop="asyncio",
            log_level="error",
        )
        self._server = uvicorn.Server(self._config)

    def run(self) -> None:
        self._server.run()

    def stop(self) -> None:
        self._server.should_exit = True


@pytest.fixture(scope="module")
def server_url():
    """Start uvicorn on _TEST_PORT, yield ws:// URL, tear down after module."""
    thread = _UvicornThread(_TEST_PORT)
    thread.start()

    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://127.0.0.1:{_TEST_PORT}/health", timeout=0.5)
            if r.status_code < 500:
                break
        except Exception:
            time.sleep(0.1)
    else:
        pytest.fail("uvicorn server did not start within 5 seconds")

    yield f"ws://127.0.0.1:{_TEST_PORT}"

    thread.stop()
    thread.join(timeout=5)


# ── Defender reward comparison ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_defender_rewards_match_inprocess(server_url: str) -> None:
    """HTTP client produces the same per-step rewards as the in-process env."""
    # In-process rollout
    env = SreArenaEnvironment()
    env.reset(role="defender", task_id="task1", seed=42)
    rng_local = random.Random(42)
    local_rewards = [
        env.step(_random_defender_action(rng_local)).reward
        for _ in range(50)
    ]

    # HTTP rollout through the WebSocket client
    http_rewards = []
    async with SreArenaEnvClient(server_url) as client:
        await client.reset(role="defender", task_id="task1", seed=42)
        rng_http = random.Random(42)
        for _ in range(50):
            result = await client.step(_random_defender_action(rng_http))
            http_rewards.append(result.observation.reward)

    assert len(http_rewards) == 50
    for i, (local_r, http_r) in enumerate(zip(local_rewards, http_rewards)):
        assert abs(local_r - http_r) < 1e-6, (
            f"Step {i + 1}: in-process={local_r} http={http_r} "
            f"diff={abs(local_r - http_r):.2e}"
        )


@pytest.mark.asyncio
async def test_defender_final_obs_fields(server_url: str) -> None:
    """HTTP client correctly parses final defender observation fields."""
    env = SreArenaEnvironment()
    env.reset(role="defender", task_id="task1", seed=42)
    rng_local = random.Random(42)
    local_obs = None
    for _ in range(50):
        local_obs = env.step(_random_defender_action(rng_local))

    async with SreArenaEnvClient(server_url) as client:
        await client.reset(role="defender", task_id="task1", seed=42)
        rng_http = random.Random(42)
        http_result = None
        for _ in range(50):
            http_result = await client.step(_random_defender_action(rng_http))

    assert http_result is not None and local_obs is not None
    http_obs = http_result.observation
    assert http_obs.done is True, "Final step must have done=True"
    assert http_obs.episode_step == 50
    assert isinstance(http_obs.current_rules, list)
    assert isinstance(http_obs.current_middleware, dict)
    assert isinstance(http_obs.last_step_metrics, dict)
    assert http_obs.last_step_metrics["requests_total"] > 0
    assert http_obs.episode_step == local_obs.episode_step
    assert http_obs.current_rules == local_obs.current_rules
    assert http_obs.current_middleware == local_obs.current_middleware


# ── Attacker reward comparison ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_attacker_rewards_match_inprocess(server_url: str) -> None:
    """HTTP client produces the same per-step rewards as the in-process env (attacker)."""
    env = SreArenaEnvironment()
    env.reset(role="attacker", task_id="task1", seed=99)
    rng_local = random.Random(99)
    local_rewards = [
        env.step(_random_attacker_action(rng_local)).reward
        for _ in range(50)
    ]

    http_rewards = []
    async with SreArenaEnvClient(server_url) as client:
        await client.reset(role="attacker", task_id="task1", seed=99)
        rng_http = random.Random(99)
        for _ in range(50):
            result = await client.step(_random_attacker_action(rng_http))
            http_rewards.append(result.observation.reward)

    assert len(http_rewards) == 50
    for i, (local_r, http_r) in enumerate(zip(local_rewards, http_rewards)):
        assert abs(local_r - http_r) < 1e-6, (
            f"Step {i + 1}: in-process={local_r} http={http_r} "
            f"diff={abs(local_r - http_r):.2e}"
        )


@pytest.mark.asyncio
async def test_attacker_final_obs_fields(server_url: str) -> None:
    """HTTP client correctly parses final attacker observation fields."""
    env = SreArenaEnvironment()
    env.reset(role="attacker", task_id="task1", seed=99)
    rng_local = random.Random(99)
    local_obs = None
    for _ in range(50):
        local_obs = env.step(_random_attacker_action(rng_local))

    async with SreArenaEnvClient(server_url) as client:
        await client.reset(role="attacker", task_id="task1", seed=99)
        rng_http = random.Random(99)
        http_result = None
        for _ in range(50):
            http_result = await client.step(_random_attacker_action(rng_http))

    assert http_result is not None and local_obs is not None
    http_obs = http_result.observation
    assert http_obs.done is True
    assert http_obs.episode_step == 50
    assert isinstance(http_obs.last_response_summary, dict)
    assert http_obs.steps_remaining == 0
    assert http_obs.episode_step == local_obs.episode_step
    assert http_obs.steps_remaining == local_obs.steps_remaining


# ── State endpoint ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_state_endpoint(server_url: str) -> None:
    """State endpoint returns ArenaState with correct step_count and role."""
    async with SreArenaEnvClient(server_url) as client:
        await client.reset(role="defender", task_id="task1")
        await client.step(DefenderAction(action_type="read_log"))
        state = await client.state()

    assert state.step_count == 1
    assert state.current_role == "defender"
    assert state.current_task == "task1"
    assert state.episode_id is not None


# ── Reward range sanity over HTTP ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rewards_in_bounds_over_http(server_url: str) -> None:
    """All HTTP-returned rewards fall within the expected [-0.5, 1.5] range."""
    async with SreArenaEnvClient(server_url) as client:
        await client.reset(role="defender", task_id="task1")
        rng = random.Random(7)
        for i in range(50):
            result = await client.step(_random_defender_action(rng))
            assert -0.5 <= result.observation.reward <= 1.5, (
                f"Step {i + 1}: reward {result.observation.reward} out of [-0.5, 1.5]"
            )
