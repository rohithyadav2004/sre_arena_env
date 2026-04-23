"""Composable reward rubrics for the SRE Arena environment.

All rubrics are stateless — forward() is a pure function of (action, observation).
State and counters live in the environment; rubrics read from observation.metadata.

Hierarchy:
    ArenaRubric
    └── role_rubrics: RubricDict
        ├── "defender": DefenderRubric
        │   ├── malicious_blocked_ratio: MaliciousBlockedRatio
        │   ├── legit_allowed_ratio:     LegitAllowedRatio
        │   └── middleware_penalty:      MiddlewarePenalty
        └── "attacker": AttackerRubric
            ├── malicious_success_ratio:  MaliciousSuccessRatio
            ├── template_diversity_bonus: TemplateDiversityBonus
            └── legit_decoy_penalty:      LegitDecoyPenalty

Formulas (from spec, exact):
    DefenderRubric  = malicious_blocked_ratio × legit_allowed_ratio + middleware_penalty
    AttackerRubric  = max(0, malicious_success × diversity_bonus − legit_decoy_penalty)
    diversity_bonus = min(1.4, 1.0 + 0.1 × max(0, unique_templates − 1))
"""
from __future__ import annotations

from typing import Any

from openenv.core.rubrics.base import Rubric
from openenv.core.rubrics.containers import RubricDict


# ── Defender sub-rubrics ──────────────────────────────────────────────────────


class MaliciousBlockedRatio(Rubric):
    """malicious_blocked / total_malicious. Returns 0.0 when total_malicious=0."""

    def forward(self, action: Any, observation: Any) -> float:
        meta = observation.metadata
        total = meta.get("total_malicious", 0)
        if total == 0:
            return 0.0
        return meta.get("malicious_blocked", 0) / total


class LegitAllowedRatio(Rubric):
    """legit_allowed / total_legit. Returns 0.0 when total_legit=0."""

    def forward(self, action: Any, observation: Any) -> float:
        meta = observation.metadata
        total = meta.get("total_legit", 0)
        if total == 0:
            return 0.0
        return meta.get("legit_allowed", 0) / total


class MiddlewarePenalty(Rubric):
    """-0.05 per unrecognised middleware write this episode."""

    def forward(self, action: Any, observation: Any) -> float:
        count = observation.metadata.get("unrecognized_middleware_count", 0)
        return -0.05 * count


class DefenderRubric(Rubric):
    """Composite defender reward.

    Formula: malicious_blocked_ratio × legit_allowed_ratio + middleware_penalty

    The multiplicative core is the anti-exploit design:
    - Block everything → legit_allowed=0 → product=0
    - Block nothing    → malicious_blocked=0 → product=0
    Only surgically precise defence earns reward.
    """

    def __init__(self) -> None:
        super().__init__()
        self.malicious_blocked_ratio = MaliciousBlockedRatio()
        self.legit_allowed_ratio = LegitAllowedRatio()
        self.middleware_penalty = MiddlewarePenalty()

    def forward(self, action: Any, observation: Any) -> float:
        mbr = self.malicious_blocked_ratio(action, observation)
        lar = self.legit_allowed_ratio(action, observation)
        mp = self.middleware_penalty(action, observation)
        return mbr * lar + mp


# ── Attacker sub-rubrics ──────────────────────────────────────────────────────


class MaliciousSuccessRatio(Rubric):
    """malicious_200_count / malicious_total. Returns 0.0 when malicious_total=0."""

    def forward(self, action: Any, observation: Any) -> float:
        meta = observation.metadata
        total = meta.get("malicious_total", 0)
        if total == 0:
            return 0.0
        return meta.get("malicious_200_count", 0) / total


class TemplateDiversityBonus(Rubric):
    """min(1.4, 1.0 + 0.1 × max(0, unique_templates_used − 1)).

    Cap at 1.4 prevents reward-hacking via random template spam.
    """

    def forward(self, action: Any, observation: Any) -> float:
        unique = observation.metadata.get("unique_templates_used", 0)
        return min(1.4, 1.0 + 0.1 * max(0, unique - 1))


class LegitDecoyPenalty(Rubric):
    """Fraction of attacker steps that generated zero malicious requests."""

    def forward(self, action: Any, observation: Any) -> float:
        return float(observation.metadata.get("legit_decoy_fraction", 0.0))


class AttackerRubric(Rubric):
    """Composite attacker reward.

    Formula: max(0, malicious_success × diversity_bonus − legit_decoy_penalty)

    Clamped at 0 so passive strategies (send only legit) yield 0, not negative.
    """

    def __init__(self) -> None:
        super().__init__()
        self.malicious_success_ratio = MaliciousSuccessRatio()
        self.template_diversity_bonus = TemplateDiversityBonus()
        self.legit_decoy_penalty = LegitDecoyPenalty()

    def forward(self, action: Any, observation: Any) -> float:
        msr = self.malicious_success_ratio(action, observation)
        tdb = self.template_diversity_bonus(action, observation)
        ldp = self.legit_decoy_penalty(action, observation)
        return max(0.0, msr * tdb - ldp)


# ── Top-level arena rubric ────────────────────────────────────────────────────


class ArenaRubric(Rubric):
    """Top-level rubric dispatching by observation.metadata["role"].

    Holds a RubricDict with "defender" and "attacker" keys.
    Reads the active role from metadata and forwards to the right sub-rubric.
    """

    def __init__(self) -> None:
        super().__init__()
        self.role_rubrics = RubricDict(
            {"defender": DefenderRubric(), "attacker": AttackerRubric()}
        )

    def forward(self, action: Any, observation: Any) -> float:
        role = observation.metadata.get("role", "defender")
        return self.role_rubrics[role](action, observation)
