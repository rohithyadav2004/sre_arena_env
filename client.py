from __future__ import annotations

from typing import Any, Dict, Union

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

try:
    from .models import (
        ArenaState,
        AttackerAction,
        AttackerObservation,
        DefenderAction,
        DefenderObservation,
    )
except ImportError:
    from models import (  # type: ignore[no-redef]
        ArenaState,
        AttackerAction,
        AttackerObservation,
        DefenderAction,
        DefenderObservation,
    )

_AnyAction = Union[DefenderAction, AttackerAction]
_AnyObs = Union[DefenderObservation, AttackerObservation]


class SreArenaEnvClient(EnvClient[_AnyAction, _AnyObs, ArenaState]):
    """WebSocket client for SreArenaEnvironment.

    Handles both defender and attacker roles. Call reset(role="defender") or
    reset(role="attacker") to start an episode; subsequent step() calls must
    use the matching action type.

    Sync usage (recommended for scripting):
        client = SreArenaEnvClient("http://localhost:8000").sync()
        with client:
            result = client.reset(role="defender")
            result = client.step(DefenderAction(action_type="read_log"))

    Async usage:
        async with SreArenaEnvClient("http://localhost:8000") as client:
            result = await client.reset(role="defender")
            result = await client.step(DefenderAction(action_type="read_log"))
    """

    def __init__(self, base_url: str, **kwargs: Any) -> None:
        super().__init__(base_url, **kwargs)
        self._current_role: str = "defender"

    async def reset(self, **kwargs: Any) -> StepResult[_AnyObs]:
        """Reset the environment, capturing role for subsequent observation parsing."""
        self._current_role = kwargs.get("role", "defender")
        return await super().reset(**kwargs)

    def _step_payload(self, action: _AnyAction) -> Dict[str, Any]:
        return action.model_dump()

    def _parse_result(self, payload: Dict[str, Any]) -> StepResult[_AnyObs]:
        obs_data = payload.get("observation", {})
        reward = payload.get("reward")
        done = payload.get("done", False)

        if self._current_role == "defender":
            obs: _AnyObs = DefenderObservation(
                log_tail=obs_data.get("log_tail", ""),
                current_rules=obs_data.get("current_rules", []),
                current_middleware=obs_data.get("current_middleware", {}),
                episode_step=obs_data.get("episode_step", 0),
                last_step_metrics=obs_data.get("last_step_metrics", {}),
                done=done,
                reward=reward,
            )
        else:
            obs = AttackerObservation(
                last_response_summary=obs_data.get("last_response_summary", {}),
                probed_rules=obs_data.get("probed_rules", []),
                episode_step=obs_data.get("episode_step", 0),
                steps_remaining=obs_data.get("steps_remaining", 0),
                done=done,
                reward=reward,
            )

        return StepResult(observation=obs, reward=reward, done=done)

    def _parse_state(self, payload: Dict[str, Any]) -> ArenaState:
        return ArenaState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            current_task=payload.get("current_task", "task1"),
            current_role=payload.get("current_role", self._current_role),
        )
