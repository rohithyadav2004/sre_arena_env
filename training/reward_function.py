"""GRPO reward functions for defender and attacker training.

GRPOTrainer (TRL >=0.21.0) calls reward functions as:
    fn(prompts, completions, **extra_columns) -> list[float]

Extra Dataset columns are forwarded by TRL as keyword arguments,
each a list aligned with the completions list.
"""
from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)

try:
    from ..server.sre_arena_env_environment import SreArenaEnvironment
    from .action_parser import parse_defender_action, parse_attacker_action
except ImportError:
    from server.sre_arena_env_environment import SreArenaEnvironment
    from training.action_parser import parse_defender_action, parse_attacker_action


def make_reward_function(
    task_id: str = "task1",
    parse_failure_penalty: float = -0.1,
) -> Any:
    """Return a GRPO reward function bound to the given env task.

    Each completion is evaluated in a fresh SreArenaEnvironment so there is
    no shared state between rollouts. Malformed completions receive
    parse_failure_penalty instead of a reward.

    Args:
        task_id: Scenario task ID passed to env.reset().
        parse_failure_penalty: Reward returned when LLM output cannot be parsed.

    Returns:
        Callable compatible with GRPOTrainer's reward_funcs parameter.
    """

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        attacker_params: list[str],
        episode_seed: list[int],
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, att_json, seed in zip(completions, attacker_params, episode_seed):
            action, err = parse_defender_action(completion)
            if action is None:
                logger.debug("parse failure (seed=%d): %s", seed, err)
                rewards.append(parse_failure_penalty)
                continue

            env = SreArenaEnvironment()
            env.reset(role="defender", seed=int(seed), task_id=task_id)
            env._last_attacker_action = json.loads(att_json)
            obs = env.step(action)
            rewards.append(float(obs.reward))

        return rewards

    return reward_fn


def make_attacker_reward_function(
    task_id: str = "task1",
    parse_failure_penalty: float = -0.1,
) -> Any:
    """Return a GRPO reward function for attacker training.

    Resets env as attacker (fresh, no defender rules) and evaluates the
    parsed AttackerAction. Malformed completions receive parse_failure_penalty.

    Args:
        task_id: Scenario task ID passed to env.reset().
        parse_failure_penalty: Reward returned when LLM output cannot be parsed.

    Returns:
        Callable compatible with GRPOTrainer's reward_funcs parameter.
    """

    def reward_fn(
        prompts: list[str],
        completions: list[str],
        episode_seed: list[int],
        **kwargs: Any,
    ) -> list[float]:
        rewards: list[float] = []
        for completion, seed in zip(completions, episode_seed):
            action, err = parse_attacker_action(completion)
            if action is None:
                logger.debug("attacker parse failure (seed=%d): %s", seed, err)
                rewards.append(parse_failure_penalty)
                continue

            env = SreArenaEnvironment()
            env.reset(role="attacker", seed=int(seed), task_id=task_id)
            obs = env.step(action)
            rewards.append(float(obs.reward))

        return rewards

    return reward_fn
