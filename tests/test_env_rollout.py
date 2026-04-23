"""Full 50-step random-action rollout tests for both agent roles.

Tests do NOT check specific reward values — only invariants:
- No exceptions raised during rollout
- All rewards within [-0.5, 1.5]
- episode_id remains stable across the 50-step episode
- step_count increments consistently
- done=True on the final step
"""
from __future__ import annotations

import random

import pytest

from sre_arena_env.server.sre_arena_env_environment import SreArenaEnvironment
from sre_arena_env.models import AttackerAction, DefenderAction


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
    ("console.log('bad middleware');", "/api/process"),  # unrecognized — tests penalty path
]


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


class TestDefenderRollout:
    def test_50_steps_no_exception(self):
        env = SreArenaEnvironment()
        env.reset(role="defender", task_id="task1")
        rng = random.Random(42)
        for _ in range(50):
            env.step(_random_defender_action(rng))

    def test_rewards_in_bounds(self):
        env = SreArenaEnvironment()
        env.reset(role="defender", task_id="task1")
        rng = random.Random(42)
        for i in range(50):
            obs = env.step(_random_defender_action(rng))
            assert -0.5 <= obs.reward <= 1.5, (
                f"Step {i + 1}: reward {obs.reward} out of [-0.5, 1.5]"
            )

    def test_episode_id_stable_across_50_steps(self):
        env = SreArenaEnvironment()
        env.reset(role="defender", task_id="task1")
        episode_id = env.state.episode_id
        rng = random.Random(42)
        for _ in range(50):
            env.step(_random_defender_action(rng))
            assert env.state.episode_id == episode_id, "episode_id changed mid-episode"

    def test_step_count_increments(self):
        env = SreArenaEnvironment()
        env.reset(role="defender", task_id="task1")
        rng = random.Random(42)
        for i in range(1, 6):
            env.step(_random_defender_action(rng))
            assert env.state.step_count == i

    def test_reset_produces_new_episode_id(self):
        env = SreArenaEnvironment()
        env.reset(role="defender")
        id1 = env.state.episode_id
        env.reset(role="defender")
        id2 = env.state.episode_id
        assert id1 != id2

    def test_done_true_on_last_step(self):
        env = SreArenaEnvironment()
        env.reset(role="defender", task_id="task1")
        rng = random.Random(42)
        obs = None
        for _ in range(50):
            obs = env.step(_random_defender_action(rng))
        assert obs.done is True

    def test_obs_fields_populated(self):
        env = SreArenaEnvironment()
        env.reset(role="defender")
        obs = env.step(DefenderAction(action_type="read_log"))
        assert isinstance(obs.current_rules, list)
        assert isinstance(obs.current_middleware, dict)
        assert isinstance(obs.last_step_metrics, dict)
        assert obs.episode_step == 1


class TestAttackerRollout:
    def test_50_steps_no_exception(self):
        env = SreArenaEnvironment()
        env.reset(role="attacker", task_id="task1")
        rng = random.Random(99)
        for _ in range(50):
            env.step(_random_attacker_action(rng))

    def test_rewards_in_bounds(self):
        env = SreArenaEnvironment()
        env.reset(role="attacker", task_id="task1")
        rng = random.Random(99)
        for i in range(50):
            obs = env.step(_random_attacker_action(rng))
            assert -0.5 <= obs.reward <= 1.5, (
                f"Step {i + 1}: reward {obs.reward} out of [-0.5, 1.5]"
            )

    def test_episode_id_stable_across_50_steps(self):
        env = SreArenaEnvironment()
        env.reset(role="attacker", task_id="task1")
        episode_id = env.state.episode_id
        rng = random.Random(99)
        for _ in range(50):
            env.step(_random_attacker_action(rng))
            assert env.state.episode_id == episode_id

    def test_steps_remaining_decreases(self):
        env = SreArenaEnvironment()
        env.reset(role="attacker", task_id="task1")
        rng = random.Random(99)
        obs1 = env.step(_random_attacker_action(rng))
        obs2 = env.step(_random_attacker_action(rng))
        assert obs2.steps_remaining < obs1.steps_remaining

    def test_attacker_obs_has_response_summary(self):
        env = SreArenaEnvironment()
        env.reset(role="attacker", task_id="task1")
        rng = random.Random(99)
        obs = env.step(_random_attacker_action(rng))
        assert isinstance(obs.last_response_summary, dict)
        assert len(obs.last_response_summary) > 0


class TestRoleIsolation:
    def test_defender_then_attacker_reset_is_clean(self):
        env = SreArenaEnvironment()
        env.reset(role="defender")
        for _ in range(5):
            env.step(DefenderAction(action_type="read_log"))
        env.reset(role="attacker")
        for _ in range(5):
            env.step(AttackerAction(template="single_ip_flood"))

    def test_state_reflects_current_role(self):
        env = SreArenaEnvironment()
        env.reset(role="defender")
        assert env.state.current_role == "defender"
        env.reset(role="attacker")
        assert env.state.current_role == "attacker"
