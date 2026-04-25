"""Tests for training.scripted_attacker."""
from __future__ import annotations

import pytest

from sre_arena_env.models import AttackerAction
from sre_arena_env.training.scripted_attacker import ScriptedAttacker, TEMPLATES


class TestScriptedAttacker:
    def test_act_returns_attacker_action(self):
        assert isinstance(ScriptedAttacker(seed=0).act(), AttackerAction)

    def test_templates_list_has_8_entries(self):
        assert len(TEMPLATES) == 8

    def test_templates_match_models_literals(self):
        expected = {
            "single_ip_flood", "ip_spray", "credential_stuffing", "payload_injection",
            "header_spoof", "slow_drip", "path_traversal", "mixed_legit_cover",
        }
        assert set(TEMPLATES) == expected

    def test_all_8_templates_reachable_in_100_calls(self):
        attacker = ScriptedAttacker(seed=42)
        seen = {attacker.act().template for _ in range(100)}
        assert seen == set(TEMPLATES)

    def test_same_seed_produces_same_sequence(self):
        results1 = [ScriptedAttacker(seed=7).act().template for _ in range(20)]
        results2 = [ScriptedAttacker(seed=7).act().template for _ in range(20)]
        assert results1 == results2

    def test_different_seeds_produce_different_sequences(self):
        results1 = [ScriptedAttacker(seed=1).act().template for _ in range(20)]
        results2 = [ScriptedAttacker(seed=99).act().template for _ in range(20)]
        assert results1 != results2

    def test_model_dump_has_required_keys_no_delay_ms(self):
        d = ScriptedAttacker(seed=0).act().model_dump(exclude={"delay_ms", "metadata"})
        assert {"template", "count", "target_path", "source_ips", "payload"} == set(d.keys())

    def test_count_at_least_1_for_all_outputs(self):
        attacker = ScriptedAttacker(seed=42)
        assert all(attacker.act().count >= 1 for _ in range(50))

    def test_target_path_starts_with_slash(self):
        attacker = ScriptedAttacker(seed=42)
        assert all(attacker.act().target_path.startswith("/") for _ in range(50))

    def test_act_never_raises(self):
        attacker = ScriptedAttacker(seed=123)
        for _ in range(50):
            attacker.act()
