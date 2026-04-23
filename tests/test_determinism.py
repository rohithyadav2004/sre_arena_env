"""Determinism tests for Phase 2.5.

Two env instances reset with the same seed must produce identical reward
vectors. Two instances reset with different seeds must produce different
observable output wherever the seed controls meaningful variance.

Design note — Phase 1 attacker reward invariance:
  In Phase 1 the scripted defender is passive (reads log, no rules), so all
  malicious requests receive HTTP 200. AttackerRubric = max(0, success *
  diversity - decoy_penalty) = 1.0 regardless of seed. Seed controls which
  credentials and IPs appear in the traffic, but not whether they are blocked.
  For attacker "different seeds → different output" we therefore test at the
  traffic-generation level rather than the reward level.
"""
from __future__ import annotations

from sre_arena_env.server.sre_arena_env_environment import SreArenaEnvironment
from sre_arena_env.server.simulator.traffic import generate_episode_traffic
from sre_arena_env.models import AttackerAction, DefenderAction


# ── Helpers ───────────────────────────────────────────────────────────────────


def _attacker_rewards(seed: int) -> list[float]:
    env = SreArenaEnvironment()
    env.reset(role="attacker", task_id="task1", seed=seed)
    return [
        env.step(
            AttackerAction(
                template="credential_stuffing",
                count=10,
                target_path="/login",
            )
        ).reward
        for _ in range(5)
    ]


def _defender_rewards_ratelimited(seed: int) -> list[float]:
    """Rate-limit scenario: burst=1 means each IP is allowed once per step.

    With 80 legit requests drawn from 50 IPs (random), the number of unique
    IPs in each seeded batch varies — so legit_allowed_ratio varies — so the
    product malicious_blocked_ratio × legit_allowed_ratio varies by seed.
    """
    env = SreArenaEnvironment()
    env.reset(role="defender", task_id="task1", seed=seed)
    results = []
    for i in range(5):
        if i == 0:
            act = DefenderAction(
                action_type="append_nginx_rule",
                rule_text="limit_req zone=r burst=1 nodelay;",
            )
        else:
            act = DefenderAction(action_type="read_log")
        results.append(env.step(act).reward)
    return results


# ── Same-seed determinism (both roles) ───────────────────────────────────────


def test_attacker_same_seed_is_deterministic() -> None:
    """Two attacker episodes seeded identically produce identical reward vectors."""
    assert _attacker_rewards(42) == _attacker_rewards(42)


def test_defender_same_seed_is_deterministic() -> None:
    """Two defender episodes seeded identically produce identical reward vectors."""
    env_a = SreArenaEnvironment()
    env_a.reset(role="defender", task_id="task1", seed=42)
    env_b = SreArenaEnvironment()
    env_b.reset(role="defender", task_id="task1", seed=42)
    action = DefenderAction(action_type="append_nginx_rule", rule_text="deny 10.0.0.1;")
    rewards_a = [env_a.step(action).reward for _ in range(5)]
    rewards_b = [env_b.step(action).reward for _ in range(5)]
    assert rewards_a == rewards_b


# ── Different-seed variance ───────────────────────────────────────────────────


def test_defender_different_seeds_differ() -> None:
    """Defender rewards vary between episodes seeded differently.

    The rate-limit rule (burst=1) makes legit_allowed_ratio depend on how
    many unique source IPs appear in the seeded legit traffic batch, which
    differs between seeds.
    """
    r42 = _defender_rewards_ratelimited(42)
    r99 = _defender_rewards_ratelimited(99)
    assert r42 != r99, (
        f"Expected different rewards for seeds 42 vs 99, both got {r42}"
    )


def test_attacker_seed_controls_traffic_generation() -> None:
    """Seed produces deterministic traffic and different seeds produce different traffic.

    Tests both legit and attack request generation (covers the attacker role's
    traffic path). With a passive defender in Phase 1, attack outcomes are
    structurally invariant to seed (all succeed), so we verify at the traffic
    level instead of the reward level.
    """
    attack_params = {
        "template": "credential_stuffing",
        "count": 10,
        "target_path": "/login",
        "_seed": 42,
    }
    t1 = generate_episode_traffic("task1", attack_params, seed=42)
    t2 = generate_episode_traffic("task1", attack_params, seed=42)
    t3 = generate_episode_traffic("task1", dict(attack_params, _seed=99), seed=99)

    assert t1 == t2, "Same seed must produce identical traffic"
    assert t1 != t3, "Different seeds must produce different traffic"


# ── Smoke tests ───────────────────────────────────────────────────────────────


def test_unseeded_env_still_runs() -> None:
    """seed=None (default) must not raise — env runs with random entropy."""
    env = SreArenaEnvironment()
    env.reset(role="attacker", task_id="task1")
    obs = env.step(
        AttackerAction(
            template="credential_stuffing",
            count=5,
            target_path="/login",
        )
    )
    assert obs.reward is not None
