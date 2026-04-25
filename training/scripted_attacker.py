"""Scripted attacker for defender Gen-1 GRPO training.

Provides randomised attack templates so the training dataset covers all 8
attack types, not just single_ip_flood.
"""
from __future__ import annotations

import logging
import random

logger = logging.getLogger(__name__)

try:
    from ..models import AttackerAction
except ImportError:
    from models import AttackerAction

TEMPLATES: list[str] = [
    "single_ip_flood",
    "ip_spray",
    "credential_stuffing",
    "payload_injection",
    "header_spoof",
    "slow_drip",
    "path_traversal",
    "mixed_legit_cover",
]

_DEFAULTS: dict[str, dict] = {
    "single_ip_flood":     {"count": 20, "target_path": "/login"},
    "ip_spray":            {"count": 15, "target_path": "/api/data"},
    "credential_stuffing": {"count": 10, "target_path": "/login"},
    "payload_injection":   {"count": 8,  "target_path": "/api/process"},
    "header_spoof":        {"count": 10, "target_path": "/login"},
    "slow_drip":           {"count": 5,  "target_path": "/"},
    "path_traversal":      {"count": 8,  "target_path": "/api/data"},
    "mixed_legit_cover":   {"count": 12, "target_path": "/"},
}


class ScriptedAttacker:
    """Randomly selects attack templates for defender training episodes.

    Uses a seeded RNG so the same seed always produces the same sequence.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def act(self) -> AttackerAction:
        """Return a random AttackerAction drawn uniformly from all 8 templates."""
        template = self._rng.choice(TEMPLATES)
        defaults = _DEFAULTS[template]
        return AttackerAction(
            template=template,
            count=defaults["count"],
            target_path=defaults["target_path"],
        )
