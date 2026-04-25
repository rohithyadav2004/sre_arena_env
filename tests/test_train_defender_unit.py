"""Unit tests for training modules — no GPU required."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from datasets import Dataset

from sre_arena_env.models import DefenderObservation
from sre_arena_env.training.dataset_builder import build_rollout_dataset
from sre_arena_env.training.reward_function import make_reward_function
from sre_arena_env.training.train_defender import _collect_observations, load_config

_CONFIGS = Path(__file__).parent.parent / "configs"


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_obs() -> DefenderObservation:
    return DefenderObservation(
        episode_step=1,
        last_step_metrics={
            "requests_total": 90,
            "200_count": 80,
            "403_count": 10,
            "429_count": 0,
        },
    )


def _make_attacker_dict() -> dict:
    return {
        "template": "single_ip_flood",
        "count": 10,
        "target_path": "/login",
        "source_ips": ["10.0.0.1"],
        "payload": {},
    }


# ── build_rollout_dataset ────────────────────────────────────────────────────

class TestBuildRolloutDataset:
    def test_returns_dataset_instance(self):
        ds = build_rollout_dataset([_make_obs()], [_make_attacker_dict()], [42])
        assert isinstance(ds, Dataset)

    def test_has_correct_columns(self):
        ds = build_rollout_dataset([_make_obs()], [_make_attacker_dict()], [42])
        assert set(ds.column_names) == {"prompt", "attacker_params", "episode_seed"}

    def test_length_matches_input(self):
        obs = [_make_obs() for _ in range(5)]
        dicts = [_make_attacker_dict() for _ in range(5)]
        seeds = list(range(5))
        assert len(build_rollout_dataset(obs, dicts, seeds)) == 5

    def test_prompt_contains_system_marker(self):
        ds = build_rollout_dataset([_make_obs()], [_make_attacker_dict()], [42])
        assert "SYSTEM:" in ds[0]["prompt"]

    def test_attacker_params_is_valid_json(self):
        ds = build_rollout_dataset([_make_obs()], [_make_attacker_dict()], [42])
        parsed = json.loads(ds[0]["attacker_params"])
        assert parsed["template"] == "single_ip_flood"

    def test_episode_seed_preserved(self):
        ds = build_rollout_dataset([_make_obs()], [_make_attacker_dict()], [999])
        assert ds[0]["episode_seed"] == 999

    def test_length_mismatch_raises_value_error(self):
        with pytest.raises(ValueError, match="Length mismatch"):
            build_rollout_dataset([_make_obs()], [], [42])


# ── make_reward_function ─────────────────────────────────────────────────────

_SINGLE_IP_PARAMS = json.dumps({
    "template": "single_ip_flood",
    "count": 10,
    "target_path": "/login",
    "source_ips": ["10.0.0.1"],
    "payload": {},
})


class TestMakeRewardFunction:
    def test_returns_callable(self):
        assert callable(make_reward_function())

    def test_malformed_completion_returns_penalty(self):
        fn = make_reward_function(parse_failure_penalty=-0.1)
        rewards = fn(
            prompts=["p"],
            completions=["not json at all"],
            attacker_params=[_SINGLE_IP_PARAMS],
            episode_seed=[42],
        )
        assert rewards == [-0.1]

    def test_read_log_completion_returns_zero_reward(self):
        # No blocking rules → malicious_blocked_ratio = 0 → reward = 0*anything = 0
        fn = make_reward_function()
        rewards = fn(
            prompts=["p"],
            completions=['{"action_type": "read_log", "log_tail_lines": 20}'],
            attacker_params=[_SINGLE_IP_PARAMS],
            episode_seed=[42],
        )
        assert rewards[0] == pytest.approx(0.0)

    def test_deny_rule_against_known_ip_returns_positive_reward(self):
        # deny 10.0.0.1; blocks all single_ip_flood traffic from that IP.
        # 80 legit from 192.168.1.x pass through → reward = 1.0 × 1.0 = 1.0
        fn = make_reward_function()
        rewards = fn(
            prompts=["p"],
            completions=['{"action_type": "append_nginx_rule", "rule_text": "deny 10.0.0.1;"}'],
            attacker_params=[_SINGLE_IP_PARAMS],
            episode_seed=[42],
        )
        assert rewards[0] > 0.0

    def test_reward_in_valid_range(self):
        fn = make_reward_function()
        rewards = fn(
            prompts=["p"],
            completions=['{"action_type": "read_log"}'],
            attacker_params=[_SINGLE_IP_PARAMS],
            episode_seed=[42],
        )
        assert -0.5 <= rewards[0] <= 1.5

    def test_batch_of_three_completions(self):
        fn = make_reward_function(parse_failure_penalty=-0.1)
        rewards = fn(
            prompts=["p", "p", "p"],
            completions=[
                "garbage",
                '{"action_type": "read_log"}',
                '{"action_type": "append_nginx_rule", "rule_text": "deny 10.0.0.1;"}',
            ],
            attacker_params=[_SINGLE_IP_PARAMS, _SINGLE_IP_PARAMS, _SINGLE_IP_PARAMS],
            episode_seed=[42, 42, 42],
        )
        assert len(rewards) == 3
        assert rewards[0] == pytest.approx(-0.1)
        assert rewards[1] == pytest.approx(0.0)
        assert rewards[2] > 0.0

    def test_custom_penalty_respected(self):
        fn = make_reward_function(parse_failure_penalty=-0.5)
        rewards = fn(
            prompts=["p"],
            completions=["{}"],  # missing action_type
            attacker_params=[_SINGLE_IP_PARAMS],
            episode_seed=[42],
        )
        assert rewards[0] == pytest.approx(-0.5)


# ── load_config ──────────────────────────────────────────────────────────────

class TestLoadConfig:
    def test_loads_colab_demo(self):
        cfg = load_config(str(_CONFIGS / "colab_demo.yaml"))
        assert cfg["training"]["num_episodes"] == 20
        assert cfg["training"]["rollouts_per_episode"] == 4
        assert cfg["env"]["task_id"] == "task1"
        assert cfg["model"]["name"] == "Qwen/Qwen2.5-3B-Instruct"

    def test_loads_l4_training(self):
        cfg = load_config(str(_CONFIGS / "l4_training.yaml"))
        assert cfg["training"]["num_episodes"] == 200
        assert cfg["training"]["rollouts_per_episode"] == 8
        assert cfg["training"]["report_to"] == "tensorboard"

    def test_nonexistent_path_raises_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Config not found"):
            load_config(str(_CONFIGS / "does_not_exist.yaml"))


# ── _collect_observations ─────────────────────────────────────────────────────

def _minimal_cfg(num_episodes: int = 3) -> dict:
    return {
        "training": {"num_episodes": num_episodes},
        "env": {"base_seed": 42, "task_id": "task1"},
    }


class TestCollectObservations:
    def test_returns_three_lists_of_equal_length(self):
        obs, dicts, seeds = _collect_observations(_minimal_cfg(num_episodes=3))
        assert len(obs) == len(dicts) == len(seeds) == 3

    def test_episode_step_is_1_for_all_observations(self):
        obs, _, _ = _collect_observations(_minimal_cfg(num_episodes=2))
        assert all(o.episode_step == 1 for o in obs)

    def test_last_step_metrics_populated(self):
        # 80 legit + N malicious = requests_total > 0
        obs, _, _ = _collect_observations(_minimal_cfg(num_episodes=2))
        for o in obs:
            assert "requests_total" in o.last_step_metrics
            assert o.last_step_metrics["requests_total"] > 0

    def test_seeds_are_base_plus_episode_index(self):
        _, _, seeds = _collect_observations(_minimal_cfg(num_episodes=3))
        assert seeds == [42, 43, 44]

    def test_attacker_dicts_have_template_key(self):
        _, dicts, _ = _collect_observations(_minimal_cfg(num_episodes=2))
        assert all("template" in d for d in dicts)
