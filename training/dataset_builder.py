"""Build HuggingFace Datasets from rollout observations."""
from __future__ import annotations

import json

from datasets import Dataset

try:
    from ..models import DefenderObservation, AttackerObservation
    from .prompts import build_defender_prompt, build_attacker_prompt
except ImportError:
    from models import DefenderObservation, AttackerObservation
    from training.prompts import build_defender_prompt, build_attacker_prompt


def build_rollout_dataset(
    observations: list[DefenderObservation],
    attacker_dicts: list[dict],
    episode_seeds: list[int],
) -> Dataset:
    """Build a HF Dataset of defender prompts from rollout observations.

    Each row is one episode's step-1 observation.

    Args:
        observations: Step-1 DefenderObservation per episode.
        attacker_dicts: Attacker parameter dicts matching each observation.
        episode_seeds: RNG seeds used in each episode's env.reset().

    Returns:
        HF Dataset with columns: prompt (str), attacker_params (JSON str),
        episode_seed (int).
    """
    if not (len(observations) == len(attacker_dicts) == len(episode_seeds)):
        raise ValueError(
            f"Length mismatch: {len(observations)} obs, "
            f"{len(attacker_dicts)} attacker_dicts, "
            f"{len(episode_seeds)} seeds"
        )
    return Dataset.from_dict({
        "prompt": [build_defender_prompt(obs) for obs in observations],
        "attacker_params": [json.dumps(d) for d in attacker_dicts],
        "episode_seed": list(episode_seeds),
    })


def build_attacker_rollout_dataset(
    observations: list[AttackerObservation],
    episode_seeds: list[int],
) -> Dataset:
    """Build a HF Dataset of attacker prompts from initial env observations.

    Each row is one episode's initial AttackerObservation (from env.reset()).
    The reward function replays each episode from its seed, so no action
    params need to be stored — only the seed is required.

    Args:
        observations: Initial AttackerObservation per episode (from env.reset()).
        episode_seeds: RNG seeds used in each episode's env.reset().

    Returns:
        HF Dataset with columns: prompt (str), episode_seed (int).
    """
    if len(observations) != len(episode_seeds):
        raise ValueError(
            f"Length mismatch: {len(observations)} obs, {len(episode_seeds)} seeds"
        )
    return Dataset.from_dict({
        "prompt": [build_attacker_prompt(obs) for obs in observations],
        "episode_seed": list(episode_seeds),
    })
