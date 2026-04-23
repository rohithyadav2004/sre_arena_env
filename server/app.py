from __future__ import annotations

import asyncio

from pydantic import ConfigDict

from openenv.core.env_server.http_server import create_app
from openenv.core.env_server.types import Action

try:
    from ..models import (
        AttackerAction,
        DefenderAction,
        DefenderObservation,
    )
    from .sre_arena_env_environment import SreArenaEnvironment
    from .dashboard.sse import router as _dashboard_router, fire_publish, set_loop
except ImportError:
    from models import (  # type: ignore[no-redef]
        AttackerAction,
        DefenderAction,
        DefenderObservation,
    )
    from server.sre_arena_env_environment import SreArenaEnvironment  # type: ignore[no-redef]
    from server.dashboard.sse import (  # type: ignore[no-redef]
        router as _dashboard_router,
        fire_publish,
        set_loop,
    )

class ArenaAction(Action):
    """Dispatch wrapper: model_validate returns DefenderAction or AttackerAction.

    create_app() accepts one action_cls. This class's model_validate inspects
    the incoming dict and delegates to the correct typed subclass, enabling
    the server to handle both defender and attacker roles without modification.

    Dispatch rule:
      - dict containing "action_type" key  → DefenderAction
      - dict containing "template" key     → AttackerAction
    """

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        extra="allow",
    )

    @classmethod
    def model_validate(cls, obj, **kwargs):  # type: ignore[override]
        if isinstance(obj, (DefenderAction, AttackerAction)):
            return obj
        if isinstance(obj, dict):
            if "action_type" in obj:
                return DefenderAction.model_validate(obj, **kwargs)
            return AttackerAction.model_validate(obj, **kwargs)
        return super().model_validate(obj, **kwargs)


app = create_app(
    SreArenaEnvironment,
    ArenaAction,
    DefenderObservation,
    env_name="sre_arena_env",
    max_concurrent_envs=10,
)

# ── Dashboard wiring ──────────────────────────────────────────────────────────

app.include_router(_dashboard_router)


async def _capture_loop() -> None:
    set_loop(asyncio.get_running_loop())


app.router.on_startup.append(_capture_loop)

# Episode reward tracking for top-bar averages (last known per role)
_episode_rewards: dict[str, list[float]] = {"defender": [], "attacker": []}


def _get_metrics() -> dict:
    def _avg(lst: list[float]) -> float:
        return round(sum(lst) / len(lst), 4) if lst else 0.0

    return {
        "blue_score": _avg(_episode_rewards["defender"]),
        "red_score": _avg(_episode_rewards["attacker"]),
    }


_orig_step = SreArenaEnvironment.step
_orig_reset = SreArenaEnvironment.reset


def _patched_step(self, action, **kwargs):  # type: ignore[no-untyped-def]
    obs = _orig_step(self, action, **kwargs)

    role: str = getattr(self, "_role", "unknown")
    state = getattr(self, "_arena_state", None)
    episode_id: str = state.episode_id if state else "unknown"
    step_num: int = state.step_count if state else 0

    _episode_rewards.setdefault(role, []).append(obs.reward)

    # Action summary
    if hasattr(action, "action_type"):
        action_dict: dict = {
            "action_type": action.action_type,
            "rule_text": getattr(action, "rule_text", ""),
            "route": getattr(action, "route", ""),
            "middleware_js": getattr(action, "middleware_js", ""),
        }
    else:
        action_dict = {
            "template": action.template,
            "count": action.count,
            "target_path": action.target_path,
            "source_ips": list(getattr(action, "source_ips", [])),
        }

    # Real nginx log sample (last 5 lines, strip empties)
    nginx = getattr(self, "_nginx", None)
    raw_tail = nginx.get_log_tail(5) if nginx else ""
    log_sample = [line for line in raw_tail.split("\n") if line]

    if role == "defender":
        obs_dict: dict = {
            "log_sample": log_sample,
            "current_rules": obs.current_rules,
            "current_middleware": obs.current_middleware,
            "last_step_metrics": obs.last_step_metrics,
            "last_response_summary": None,
        }
    else:
        obs_dict = {
            "log_sample": log_sample,
            "current_rules": [],
            "current_middleware": {},
            "last_step_metrics": None,
            "last_response_summary": obs.last_response_summary,
        }

    fire_publish({
        "type": "step",
        "episode_id": episode_id,
        "step": step_num,
        "role": role,
        "action": action_dict,
        "observation": obs_dict,
        "reward": obs.reward,
        "metrics": _get_metrics(),
        "done": obs.done,
    })

    if obs.done:
        fire_publish({
            "type": "episode_end",
            "episode_id": episode_id,
            "role": role,
            "final_reward": obs.reward,
        })

    return obs


def _patched_reset(self, **kwargs):  # type: ignore[no-untyped-def]
    obs = _orig_reset(self, **kwargs)

    role: str = getattr(self, "_role", "unknown")
    state = getattr(self, "_arena_state", None)
    episode_id: str = state.episode_id if state else "unknown"
    task_id: str = getattr(self, "_task_id", "task1")
    seed = getattr(self, "_seed", None)

    # Keep last 200 rewards (rolling) rather than zeroing, so concurrent
    # sessions for other roles are not wiped (SUPPORTS_CONCURRENT_SESSIONS=True)
    _episode_rewards[role] = _episode_rewards.get(role, [])[-200:]

    fire_publish({
        "type": "reset",
        "role": role,
        "task_id": task_id,
        "episode_id": episode_id,
        "seed": seed,
    })

    return obs


SreArenaEnvironment.step = _patched_step  # type: ignore[method-assign]
SreArenaEnvironment.reset = _patched_reset  # type: ignore[method-assign]


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    """Entry point for `uv run --project . server` and pyproject.toml [scripts]."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
