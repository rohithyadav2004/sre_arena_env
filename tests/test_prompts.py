"""Tests for training.prompts — system prompts and observation formatting."""
from __future__ import annotations

import pytest

from sre_arena_env.models import DefenderObservation, AttackerObservation
from sre_arena_env.training.prompts import (
    DEFENDER_SYSTEM_PROMPT,
    ATTACKER_SYSTEM_PROMPT,
    build_defender_prompt,
    build_attacker_prompt,
)

DEFENDER_ACTION_TYPES = ["read_log", "append_nginx_rule", "write_express_middleware"]
ATTACKER_TEMPLATES = [
    "single_ip_flood", "ip_spray", "credential_stuffing", "payload_injection",
    "header_spoof", "slow_drip", "path_traversal", "mixed_legit_cover",
]


@pytest.fixture
def defender_obs() -> DefenderObservation:
    return DefenderObservation(
        episode_step=3,
        log_tail="10.0.0.1 - GET /login 200\n10.0.0.2 - GET /login 403",
        current_rules=["deny 10.0.0.5;", "limit_req zone=flood;"],
        current_middleware={"/api/process": "next();"},
        last_step_metrics={"200_count": 8, "403_count": 2, "429_count": 0},
    )


@pytest.fixture
def attacker_obs() -> AttackerObservation:
    return AttackerObservation(
        episode_step=2,
        steps_remaining=8,
        last_response_summary={"200": 5, "403": 3, "429": 2},
        probed_rules=["deny 10.0.0.5;"],
    )


class TestDefenderSystemPrompt:
    def test_mentions_all_action_types(self):
        for action_type in DEFENDER_ACTION_TYPES:
            assert action_type in DEFENDER_SYSTEM_PROMPT, f"Missing: {action_type}"

    def test_is_concise(self):
        assert len(DEFENDER_SYSTEM_PROMPT) < 800


class TestAttackerSystemPrompt:
    def test_mentions_all_templates(self):
        for template in ATTACKER_TEMPLATES:
            assert template in ATTACKER_SYSTEM_PROMPT, f"Missing: {template}"

    def test_is_concise(self):
        assert len(ATTACKER_SYSTEM_PROMPT) < 800


class TestBuildDefenderPrompt:
    def test_includes_episode_step(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert "Episode step: 3" in prompt

    def test_includes_log_lines(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert "10.0.0.1" in prompt
        assert "10.0.0.2" in prompt

    def test_includes_status_counts(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert "200=8" in prompt
        assert "403=2" in prompt

    def test_includes_current_rules(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert "deny 10.0.0.5;" in prompt

    def test_includes_current_middleware(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert "/api/process" in prompt

    def test_ends_with_assistant_marker(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert "ASSISTANT:" in prompt

    def test_does_not_raise_on_empty_obs(self):
        obs = DefenderObservation()
        build_defender_prompt(obs)  # must not raise

    def test_total_length_under_4000(self, defender_obs):
        prompt = build_defender_prompt(defender_obs)
        assert len(prompt) < 4000


class TestBuildAttackerPrompt:
    def test_includes_episode_step(self, attacker_obs):
        prompt = build_attacker_prompt(attacker_obs)
        assert "Episode step: 2" in prompt

    def test_includes_steps_remaining(self, attacker_obs):
        prompt = build_attacker_prompt(attacker_obs)
        assert "Steps remaining: 8" in prompt

    def test_includes_response_summary(self, attacker_obs):
        prompt = build_attacker_prompt(attacker_obs)
        assert "200" in prompt

    def test_includes_probed_rules(self, attacker_obs):
        prompt = build_attacker_prompt(attacker_obs)
        assert "deny 10.0.0.5;" in prompt

    def test_ends_with_assistant_marker(self, attacker_obs):
        prompt = build_attacker_prompt(attacker_obs)
        assert "ASSISTANT:" in prompt

    def test_does_not_raise_on_empty_obs(self):
        obs = AttackerObservation()
        build_attacker_prompt(obs)  # must not raise

    def test_total_length_under_4000(self, attacker_obs):
        prompt = build_attacker_prompt(attacker_obs)
        assert len(prompt) < 4000
