from __future__ import annotations

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
except ImportError:
    from models import (  # type: ignore[no-redef]
        AttackerAction,
        DefenderAction,
        DefenderObservation,
    )
    from server.sre_arena_env_environment import SreArenaEnvironment  # type: ignore[no-redef]


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
