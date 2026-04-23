"""Direct rubric baseline tests — no environment involved.

Tests construct synthetic observations with controlled metadata and assert
rubric outputs match the expected shapes from IMPLEMENTATION_PLAN.md.
All tests use fixed data (no RNG) for full reproducibility.
"""
from __future__ import annotations

import pytest

from sre_arena_env.server.simulator.rubrics import (
    ArenaRubric,
    AttackerRubric,
    DefenderRubric,
    LegitAllowedRatio,
    LegitDecoyPenalty,
    MaliciousBlockedRatio,
    MaliciousSuccessRatio,
    MiddlewarePenalty,
    TemplateDiversityBonus,
)
from sre_arena_env.models import (
    AttackerAction,
    AttackerObservation,
    DefenderAction,
    DefenderObservation,
)

# ── Synthetic observation helpers ─────────────────────────────────────────────


def _def_obs(
    malicious_blocked: int = 0,
    total_malicious: int = 10,
    legit_allowed: int = 80,
    total_legit: int = 80,
    unrecognized: int = 0,
) -> DefenderObservation:
    return DefenderObservation(
        done=False,
        metadata={
            "role": "defender",
            "malicious_blocked": malicious_blocked,
            "total_malicious": total_malicious,
            "legit_allowed": legit_allowed,
            "total_legit": total_legit,
            "unrecognized_middleware_count": unrecognized,
        },
    )


def _att_obs(
    malicious_200: int = 0,
    malicious_total: int = 10,
    unique_templates: int = 1,
    legit_decoy_fraction: float = 0.0,
) -> AttackerObservation:
    return AttackerObservation(
        done=False,
        metadata={
            "role": "attacker",
            "malicious_200_count": malicious_200,
            "malicious_total": malicious_total,
            "unique_templates_used": unique_templates,
            "legit_decoy_fraction": legit_decoy_fraction,
        },
    )


_DEF_ACTION = DefenderAction(action_type="read_log")
_ATT_ACTION = AttackerAction(template="single_ip_flood")


# ── Sub-rubric unit tests ─────────────────────────────────────────────────────


class TestMaliciousBlockedRatio:
    def test_all_blocked(self):
        r = MaliciousBlockedRatio()
        assert r(_DEF_ACTION, _def_obs(malicious_blocked=10, total_malicious=10)) == pytest.approx(1.0)

    def test_none_blocked(self):
        r = MaliciousBlockedRatio()
        assert r(_DEF_ACTION, _def_obs(malicious_blocked=0, total_malicious=10)) == pytest.approx(0.0)

    def test_half_blocked(self):
        r = MaliciousBlockedRatio()
        assert r(_DEF_ACTION, _def_obs(malicious_blocked=5, total_malicious=10)) == pytest.approx(0.5)

    def test_zero_total_returns_zero(self):
        r = MaliciousBlockedRatio()
        assert r(_DEF_ACTION, _def_obs(malicious_blocked=0, total_malicious=0)) == pytest.approx(0.0)


class TestLegitAllowedRatio:
    def test_all_allowed(self):
        r = LegitAllowedRatio()
        assert r(_DEF_ACTION, _def_obs(legit_allowed=80, total_legit=80)) == pytest.approx(1.0)

    def test_none_allowed(self):
        r = LegitAllowedRatio()
        assert r(_DEF_ACTION, _def_obs(legit_allowed=0, total_legit=80)) == pytest.approx(0.0)

    def test_zero_total_returns_zero(self):
        r = LegitAllowedRatio()
        assert r(_DEF_ACTION, _def_obs(legit_allowed=0, total_legit=0)) == pytest.approx(0.0)


class TestMiddlewarePenalty:
    def test_zero_unrecognized(self):
        r = MiddlewarePenalty()
        assert r(_DEF_ACTION, _def_obs(unrecognized=0)) == pytest.approx(0.0)

    def test_one_unrecognized(self):
        r = MiddlewarePenalty()
        assert r(_DEF_ACTION, _def_obs(unrecognized=1)) == pytest.approx(-0.05)

    def test_three_unrecognized(self):
        r = MiddlewarePenalty()
        assert r(_DEF_ACTION, _def_obs(unrecognized=3)) == pytest.approx(-0.15)


class TestMaliciousSuccessRatio:
    def test_full_success(self):
        r = MaliciousSuccessRatio()
        assert r(_ATT_ACTION, _att_obs(malicious_200=10, malicious_total=10)) == pytest.approx(1.0)

    def test_zero_success(self):
        r = MaliciousSuccessRatio()
        assert r(_ATT_ACTION, _att_obs(malicious_200=0, malicious_total=10)) == pytest.approx(0.0)

    def test_zero_total_returns_zero(self):
        r = MaliciousSuccessRatio()
        assert r(_ATT_ACTION, _att_obs(malicious_200=0, malicious_total=0)) == pytest.approx(0.0)


class TestTemplateDiversityBonus:
    def test_one_template_gives_base_1(self):
        r = TemplateDiversityBonus()
        # min(1.4, 1.0 + 0.1 * max(0, 1-1)) = 1.0
        assert r(_ATT_ACTION, _att_obs(unique_templates=1)) == pytest.approx(1.0)

    def test_zero_templates_gives_base_1(self):
        r = TemplateDiversityBonus()
        # min(1.4, 1.0 + 0.1 * max(0, 0-1)) = min(1.4, 1.0) = 1.0
        assert r(_ATT_ACTION, _att_obs(unique_templates=0)) == pytest.approx(1.0)

    def test_three_templates_gives_1_2(self):
        r = TemplateDiversityBonus()
        # min(1.4, 1.0 + 0.1 * 2) = 1.2
        assert r(_ATT_ACTION, _att_obs(unique_templates=3)) == pytest.approx(1.2)

    def test_five_templates_gives_1_4(self):
        r = TemplateDiversityBonus()
        # min(1.4, 1.0 + 0.1 * 4) = min(1.4, 1.4) = 1.4
        assert r(_ATT_ACTION, _att_obs(unique_templates=5)) == pytest.approx(1.4)

    def test_cap_at_1_4_for_large_unique(self):
        r = TemplateDiversityBonus()
        assert r(_ATT_ACTION, _att_obs(unique_templates=10)) == pytest.approx(1.4)


class TestLegitDecoyPenalty:
    def test_no_decoy_steps(self):
        r = LegitDecoyPenalty()
        assert r(_ATT_ACTION, _att_obs(legit_decoy_fraction=0.0)) == pytest.approx(0.0)

    def test_all_decoy_steps(self):
        r = LegitDecoyPenalty()
        assert r(_ATT_ACTION, _att_obs(legit_decoy_fraction=1.0)) == pytest.approx(1.0)

    def test_half_decoy_steps(self):
        r = LegitDecoyPenalty()
        assert r(_ATT_ACTION, _att_obs(legit_decoy_fraction=0.5)) == pytest.approx(0.5)


# ── Composite rubric baselines ────────────────────────────────────────────────


class TestDefenderRubricBaselines:
    def test_random_defender_never_writes_rules(self):
        """No rules → nothing blocked → score near 0."""
        rubric = DefenderRubric()
        obs = _def_obs(malicious_blocked=0, total_malicious=10, legit_allowed=80, total_legit=80)
        score = rubric(_DEF_ACTION, obs)
        assert score < 0.1, f"Expected < 0.1, got {score}"

    def test_block_everything_defender(self):
        """Block all traffic incl. legit → legit_allowed=0 → product=0."""
        rubric = DefenderRubric()
        obs = _def_obs(malicious_blocked=10, total_malicious=10, legit_allowed=0, total_legit=80)
        assert rubric(_DEF_ACTION, obs) == pytest.approx(0.0)

    def test_block_attacking_ip_only(self):
        """Ideal defender: blocks all malicious, passes almost all legit → > 0.5."""
        rubric = DefenderRubric()
        obs = _def_obs(malicious_blocked=10, total_malicious=10, legit_allowed=78, total_legit=80)
        score = rubric(_DEF_ACTION, obs)
        assert score > 0.5, f"Expected > 0.5, got {score}"

    def test_unrecognized_middleware_penalises(self):
        rubric = DefenderRubric()
        obs_clean = _def_obs(malicious_blocked=5, total_malicious=10, legit_allowed=80, total_legit=80, unrecognized=0)
        obs_dirty = _def_obs(malicious_blocked=5, total_malicious=10, legit_allowed=80, total_legit=80, unrecognized=2)
        assert rubric(_DEF_ACTION, obs_clean) > rubric(_DEF_ACTION, obs_dirty)


class TestAttackerRubricBaselines:
    def test_random_attacker_has_positive_reward(self):
        """Some malicious through, diverse templates, no decoy steps → > 0."""
        rubric = AttackerRubric()
        obs = _att_obs(malicious_200=5, malicious_total=10, unique_templates=3, legit_decoy_fraction=0.0)
        score = rubric(_ATT_ACTION, obs)
        assert score > 0.0, f"Expected > 0, got {score}"

    def test_same_template_lower_than_diverse(self):
        """Diversity bonus lifts score — unique=1 < unique=3 at equal success."""
        rubric = AttackerRubric()
        obs_same = _att_obs(malicious_200=5, malicious_total=10, unique_templates=1, legit_decoy_fraction=0.0)
        obs_diverse = _att_obs(malicious_200=5, malicious_total=10, unique_templates=3, legit_decoy_fraction=0.0)
        assert rubric(_ATT_ACTION, obs_same) < rubric(_ATT_ACTION, obs_diverse)

    def test_legit_only_attacker_is_zero(self):
        """All steps sent 0 malicious → legit_decoy=1.0 → clamped to 0."""
        rubric = AttackerRubric()
        obs = _att_obs(malicious_200=0, malicious_total=0, unique_templates=0, legit_decoy_fraction=1.0)
        assert rubric(_ATT_ACTION, obs) == pytest.approx(0.0)

    def test_attacker_reward_never_negative(self):
        """AttackerRubric is clamped at 0 — never goes negative."""
        rubric = AttackerRubric()
        obs = _att_obs(malicious_200=0, malicious_total=5, unique_templates=1, legit_decoy_fraction=1.0)
        assert rubric(_ATT_ACTION, obs) >= 0.0


class TestArenaRubricDispatch:
    def test_dispatches_to_defender_rubric(self):
        rubric = ArenaRubric()
        obs = _def_obs(malicious_blocked=10, total_malicious=10, legit_allowed=80, total_legit=80)
        score = rubric(_DEF_ACTION, obs)
        assert 0.0 <= score <= 1.5

    def test_dispatches_to_attacker_rubric(self):
        rubric = ArenaRubric()
        obs = _att_obs(malicious_200=5, malicious_total=10, unique_templates=2)
        score = rubric(_ATT_ACTION, obs)
        assert score >= 0.0
