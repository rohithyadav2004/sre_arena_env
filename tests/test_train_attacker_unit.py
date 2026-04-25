"""Unit tests for attacker training modules — no GPU required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from sre_arena_env.models import AttackerObservation
from sre_arena_env.training.dataset_builder import build_attacker_rollout_dataset
from sre_arena_env.training.reward_function import make_attacker_reward_function
from sre_arena_env.training.train_attacker import _collect_attacker_observations, load_config

_CONFIGS = Path(__file__).parent.parent / "configs"


def _make_attacker_obs() -> AttackerObservation:
    return AttackerObservation(
        episode_step=0,
        steps_remaining=50,
    )


# ── build_attacker_rollout_dataset ──────────────────────────────────────────


class TestBuildAttackerRolloutDataset:
    def test_returns_dataset_instance(self):
        ds = build_attacker_rollout_dataset([_make_attacker_obs()], [42])
        assert isinstance(ds, Dataset)

    def test_has_correct_columns(self):
        ds = build_attacker_rollout_dataset([_make_attacker_obs()], [42])
        assert set(ds.column_names) == {"prompt", "episode_seed"}

    def test_length_matches_input(self):
        obs = [_make_attacker_obs() for _ in range(5)]
        seeds = list(range(5))
        assert len(build_attacker_rollout_dataset(obs, seeds)) == 5

    def test_prompt_contains_system_marker(self):
        ds = build_attacker_rollout_dataset([_make_attacker_obs()], [42])
        assert "SYSTEM:" in ds[0]["prompt"]

    def test_episode_seed_preserved(self):
        ds = build_attacker_rollout_dataset([_make_attacker_obs()], [999])
        assert ds[0]["episode_seed"] == 999

    def test_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            build_attacker_rollout_dataset([_make_attacker_obs()], [])


# ── make_attacker_reward_function ────────────────────────────────────────────

_FLOOD_JSON = '{"template": "single_ip_flood", "count": 10, "target_path": "/login"}'
_GARBAGE = "not json at all"


class TestMakeAttackerRewardFunction:
    def test_returns_callable(self):
        assert callable(make_attacker_reward_function())

    def test_malformed_completion_returns_penalty(self):
        fn = make_attacker_reward_function(parse_failure_penalty=-0.1)
        rewards = fn(prompts=["p"], completions=[_GARBAGE], episode_seed=[42])
        assert rewards == [-0.1]

    def test_valid_attack_returns_float(self):
        fn = make_attacker_reward_function()
        rewards = fn(prompts=["p"], completions=[_FLOOD_JSON], episode_seed=[42])
        assert len(rewards) == 1
        assert isinstance(rewards[0], float)

    def test_valid_attack_reward_in_valid_range(self):
        fn = make_attacker_reward_function()
        rewards = fn(prompts=["p"], completions=[_FLOOD_JSON], episode_seed=[42])
        assert -0.5 <= rewards[0] <= 1.5

    def test_custom_penalty_respected(self):
        fn = make_attacker_reward_function(parse_failure_penalty=-0.5)
        rewards = fn(prompts=["p"], completions=["{}"], episode_seed=[42])
        assert rewards[0] == pytest.approx(-0.5)

    def test_batch_of_completions(self):
        fn = make_attacker_reward_function(parse_failure_penalty=-0.1)
        rewards = fn(
            prompts=["p", "p"],
            completions=[_GARBAGE, _FLOOD_JSON],
            episode_seed=[42, 42],
        )
        assert len(rewards) == 2
        assert rewards[0] == pytest.approx(-0.1)
        assert isinstance(rewards[1], float)


# ── _collect_attacker_observations ───────────────────────────────────────────


def _minimal_cfg(num_episodes: int = 3) -> dict:
    return {
        "training": {"num_episodes": num_episodes},
        "env": {"base_seed": 42, "task_id": "task1"},
    }


class TestCollectAttackerObservations:
    def test_returns_two_lists_of_equal_length(self):
        obs, seeds = _collect_attacker_observations(_minimal_cfg(num_episodes=3))
        assert len(obs) == len(seeds) == 3

    def test_seeds_are_base_plus_episode_index(self):
        _, seeds = _collect_attacker_observations(_minimal_cfg(num_episodes=3))
        assert seeds == [42, 43, 44]

    def test_returns_attacker_observations(self):
        obs, _ = _collect_attacker_observations(_minimal_cfg(num_episodes=2))
        assert all(isinstance(o, AttackerObservation) for o in obs)

    def test_steps_remaining_is_positive(self):
        obs, _ = _collect_attacker_observations(_minimal_cfg(num_episodes=2))
        assert all(o.steps_remaining > 0 for o in obs)
