"""Phase 5 smoke tests against the live HF Space.

Run manually (not in CI) with:
    HF_TOKEN=<your-token> pytest tests/test_hf_space_live.py -v -m hf_space

The Space is private during development, so HTTP tests require Bearer auth.
The WebSocket rollout test requires the Space to be flipped to Public (the
openenv EnvClient does not support injecting auth headers for WS). Flip the
Space at https://huggingface.co/spaces/blitz1809/sre-arena → Settings →
Visibility, then run that test.
"""
from __future__ import annotations

import os

import pytest
import requests

SPACE_URL = "https://blitz1809-sre-arena.hf.space"


def _token() -> str:
    tok = os.environ.get("HF_TOKEN", "")
    if not tok:
        import pathlib
        cache = pathlib.Path.home() / ".cache" / "huggingface" / "token"
        if cache.exists():
            tok = cache.read_text().strip()
    return tok


def _headers() -> dict[str, str]:
    tok = _token()
    if tok:
        return {"Authorization": f"Bearer {tok}"}
    return {}


pytestmark = pytest.mark.hf_space


class TestHFSpaceLive:
    def test_root_returns_dashboard_html(self) -> None:
        r = requests.get(SPACE_URL, headers=_headers(), timeout=15)
        assert r.status_code == 200, f"Expected 200, got {r.status_code}: {r.text[:200]}"
        assert "<!DOCTYPE html" in r.text or "<!doctype html" in r.text.lower()
        assert "SRE Arena" in r.text

    def test_subscriber_count_endpoint(self) -> None:
        r = requests.get(f"{SPACE_URL}/dashboard/subscriber-count", headers=_headers(), timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert "count" in data
        assert isinstance(data["count"], int)

    def test_health_endpoint(self) -> None:
        r = requests.get(f"{SPACE_URL}/health", headers=_headers(), timeout=15)
        assert r.status_code == 200
        data = r.json()
        assert data.get("status") in ("ok", "healthy")

    @pytest.mark.skip(
        reason=(
            "WebSocket rollout requires the Space to be Public — flip at "
            "https://huggingface.co/spaces/blitz1809/sre-arena → Settings → "
            "Visibility, then remove this skip decorator."
        )
    )
    def test_seeded_rollout_matches_local(self) -> None:
        """Seeded rollout against live Space must match in-process rewards."""
        from sre_arena_env.client import SreArenaEnvClient
        from sre_arena_env.models import DefenderAction
        from sre_arena_env.server.sre_arena_env_environment import SreArenaEnvironment

        env = SreArenaEnvironment()
        local_result = env.reset(seed=42, role="defender")
        local_reward_after_step = None

        client = SreArenaEnvClient(SPACE_URL).sync()
        with client:
            client.reset(role="defender", seed=42)
            remote_step = client.step(DefenderAction(action_type="read_log"))
            remote_reward = remote_step.reward

        local_step_result = env.step(DefenderAction(action_type="read_log"))
        local_reward_after_step = local_step_result.reward

        assert remote_reward == pytest.approx(local_reward_after_step, abs=1e-6), (
            f"Remote reward {remote_reward} != local reward {local_reward_after_step}"
        )
