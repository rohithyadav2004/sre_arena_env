"""Scripted defender for attacker Gen-1 GRPO training.

Always emits read_log — a do-nothing baseline an attacker can easily defeat.
Mirrors scripted_attacker.py structure.
"""
from __future__ import annotations

import random

try:
    from ..models import DefenderAction
except ImportError:
    from models import DefenderAction


class ScriptedDefender:
    """Deterministic baseline defender: always emits read_log.

    Uses a seeded RNG for structural parity with ScriptedAttacker,
    even though the action is deterministic.
    """

    def __init__(self, seed: int = 42) -> None:
        self._rng = random.Random(seed)

    def act(self) -> DefenderAction:
        """Return a read_log DefenderAction."""
        return DefenderAction(action_type="read_log", log_tail_lines=10)
