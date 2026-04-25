"""Unit tests for ScriptedDefender — CPU-only."""
from __future__ import annotations

import pytest

from sre_arena_env.models import DefenderAction
from sre_arena_env.training.scripted_defender import ScriptedDefender


class TestScriptedDefender:
    def test_act_returns_defender_action(self):
        d = ScriptedDefender()
        assert isinstance(d.act(), DefenderAction)

    def test_act_always_returns_read_log(self):
        d = ScriptedDefender(seed=0)
        for _ in range(20):
            assert d.act().action_type == "read_log"

    def test_different_seeds_same_action_type(self):
        d1 = ScriptedDefender(seed=1)
        d2 = ScriptedDefender(seed=999)
        assert d1.act().action_type == d2.act().action_type == "read_log"

    def test_log_tail_lines_is_positive(self):
        d = ScriptedDefender()
        assert d.act().log_tail_lines >= 1

    def test_multiple_calls_same_instance_consistent(self):
        d = ScriptedDefender(seed=42)
        actions = [d.act() for _ in range(5)]
        assert all(a.action_type == "read_log" for a in actions)
