"""Tests for plots module — CPU-only, no GPU."""
from __future__ import annotations

from pathlib import Path

import pytest

from sre_arena_env.training.plots import plot_cross_gen_matrix_stub, plot_reward_curves


class TestPlotCrossGenMatrixStub:
    def test_produces_non_empty_file(self, tmp_path):
        out = tmp_path / "matrix.png"
        plot_cross_gen_matrix_stub(3, str(out))
        assert out.exists()
        assert out.stat().st_size > 0

    def test_creates_parent_dirs(self, tmp_path):
        out = tmp_path / "nested" / "deep" / "matrix.png"
        plot_cross_gen_matrix_stub(2, str(out))
        assert out.exists()

    def test_works_for_single_generation(self, tmp_path):
        out = tmp_path / "single.png"
        plot_cross_gen_matrix_stub(1, str(out))
        assert out.exists()


class TestPlotRewardCurves:
    def test_missing_checkpoint_dir_returns_gracefully(self, tmp_path):
        out = tmp_path / "reward.png"
        plot_reward_curves("/nonexistent/checkpoint/dir", str(out))
        assert not out.exists()
