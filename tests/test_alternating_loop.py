"""Unit tests for alternating_loop orchestrator — CPU-only, no GPU."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import patch, call

import pytest

from sre_arena_env.training.alternating_loop import (
    load_config,
    get_opponent_checkpoint,
    _role_for_gen,
    run_alternating_loop,
)

_CONFIGS = Path(__file__).parent.parent / "configs"


class TestLoadConfig:
    def test_colab_demo_has_num_generations(self):
        cfg = load_config(str(_CONFIGS / "colab_demo.yaml"))
        assert cfg["num_generations"] == 3

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_config("nonexistent_config.yaml")


class TestGetOpponentCheckpoint:
    def test_gen0_returns_none(self):
        assert get_opponent_checkpoint(0, "defender", "./checkpoints/base") is None

    def test_gen1_attacker_training_returns_defender_gen0_path(self):
        result = get_opponent_checkpoint(1, "attacker", "./checkpoints/base")
        assert result == "./checkpoints/base_defender_gen0"

    def test_gen2_defender_training_returns_attacker_gen1_path(self):
        result = get_opponent_checkpoint(2, "defender", "./checkpoints/base")
        assert result == "./checkpoints/base_attacker_gen1"

    def test_gen1_defender_returns_attacker_gen0_path(self):
        result = get_opponent_checkpoint(1, "defender", "./checkpoints/base")
        assert result == "./checkpoints/base_attacker_gen0"


class TestRoleForGen:
    def test_per_gen_list_respected(self):
        per_gen = [{"role": "defender"}, {"role": "attacker"}, {"role": "defender"}]
        assert _role_for_gen(0, per_gen) == "defender"
        assert _role_for_gen(1, per_gen) == "attacker"
        assert _role_for_gen(2, per_gen) == "defender"

    def test_fallback_even_odd_when_empty_list(self):
        assert _role_for_gen(0, []) == "defender"
        assert _role_for_gen(1, []) == "attacker"
        assert _role_for_gen(4, []) == "defender"


class TestRunAlternatingLoop:
    def _cfg(self) -> dict:
        return {
            "num_generations": 3,
            "per_generation": [
                {"role": "defender", "episodes": 20},
                {"role": "attacker", "episodes": 20},
                {"role": "defender", "episodes": 20},
            ],
            "training": {"output_dir": "./checkpoints/base"},
        }

    def test_dry_run_never_calls_training_functions(self):
        with patch("sre_arena_env.training.alternating_loop.train_defender") as td, \
             patch("sre_arena_env.training.alternating_loop.train_attacker") as ta:
            run_alternating_loop(self._cfg(), dry_run=True)
            td.assert_not_called()
            ta.assert_not_called()

    def test_correct_role_call_counts(self):
        with patch("sre_arena_env.training.alternating_loop.train_defender", return_value="ckpt_d") as td, \
             patch("sre_arena_env.training.alternating_loop.train_attacker", return_value="ckpt_a") as ta:
            run_alternating_loop(self._cfg(), dry_run=False)
        assert td.call_count == 2   # gen 0 and gen 2
        assert ta.call_count == 1   # gen 1

    def test_gen0_gets_none_opponent_checkpoint(self):
        with patch("sre_arena_env.training.alternating_loop.train_defender", return_value="ckpt") as td, \
             patch("sre_arena_env.training.alternating_loop.train_attacker", return_value="ckpt"):
            run_alternating_loop(self._cfg(), dry_run=False)
        first_call = td.call_args_list[0]
        assert first_call.kwargs["opponent_checkpoint"] is None
        assert first_call.kwargs["gen_idx"] == 0

    def test_gen1_attacker_gets_defender_gen0_checkpoint(self):
        with patch("sre_arena_env.training.alternating_loop.train_defender", return_value="ckpt"), \
             patch("sre_arena_env.training.alternating_loop.train_attacker", return_value="ckpt") as ta:
            run_alternating_loop(self._cfg(), dry_run=False)
        first_ta_call = ta.call_args_list[0]
        assert first_ta_call.kwargs["opponent_checkpoint"] == "./checkpoints/base_defender_gen0"
        assert first_ta_call.kwargs["gen_idx"] == 1

    def test_dry_run_cli_exits_cleanly(self):
        import subprocess
        result = subprocess.run(
            [
                sys.executable, "-m", "training.alternating_loop",
                "--config", str(_CONFIGS / "colab_demo.yaml"),
                "--dry-run",
            ],
            cwd=str(Path(__file__).parent.parent),
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
