"""SRE Arena two-agent environment.

Single class supports both defender (blue) and attacker (red) roles.
Role is set on reset(); the environment generates a scripted opponent so
each role receives well-formed reward signals in Phase 1.

Phase 1 scripted opponents:
  Defender training: attacker always uses single_ip_flood from 10.0.0.1
  Attacker training: defender always reads log (passive, no rules written)

Phase 6+ replaces scripted opponents with frozen checkpoints passed via
opponent_checkpoint.
"""
from __future__ import annotations

from uuid import uuid4

from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import (
        ArenaState,
        AttackerAction,
        AttackerObservation,
        DefenderAction,
        DefenderObservation,
    )
    from .simulator.rubrics import ArenaRubric
    from .simulator.sim_express import SimulatedExpress
    from .simulator.sim_nginx import SimulatedNginx
    from .simulator.traffic import generate_episode_traffic
except ImportError:
    from models import (  # type: ignore[no-redef]
        ArenaState,
        AttackerAction,
        AttackerObservation,
        DefenderAction,
        DefenderObservation,
    )
    from server.simulator.rubrics import ArenaRubric  # type: ignore[no-redef]
    from server.simulator.sim_express import SimulatedExpress  # type: ignore[no-redef]
    from server.simulator.sim_nginx import SimulatedNginx  # type: ignore[no-redef]
    from server.simulator.traffic import generate_episode_traffic  # type: ignore[no-redef]

MAX_EPISODE_STEPS: int = 50

_SCRIPTED_ATTACKER_ACTION: dict = {
    "template": "single_ip_flood",
    "count": 10,
    "target_path": "/login",
    "source_ips": ["10.0.0.1"],
    "payload": {},
}


class SreArenaEnvironment(Environment):
    """Two-agent SRE arena environment.

    Both roles share this class; differentiated by ``role`` passed to
    ``reset()``. The active role receives a reward from ``ArenaRubric``
    which dispatches to the correct sub-rubric based on
    ``observation.metadata["role"]``.

    Attributes:
        SUPPORTS_CONCURRENT_SESSIONS: True — pure-Python simulator is safe to
            run in parallel (no shared mutable state between instances).
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        """Initialise with a fresh ArenaRubric and blank simulator state."""
        super().__init__(rubric=ArenaRubric())
        self._nginx = SimulatedNginx()
        self._express = SimulatedExpress()
        self._role: str = "defender"
        self._task_id: str = "task1"
        self._arena_state = ArenaState(
            episode_id=str(uuid4()),
            step_count=0,
            current_task="task1",
            current_role="defender",
        )
        # Attacker-episode tracking (reset on each reset() call)
        self._episode_templates_used: set[str] = set()
        self._attacker_steps_with_no_malicious: int = 0
        self._total_attacker_steps: int = 0
        # Last attacker action used when the current role is defender
        self._last_attacker_action: dict = _SCRIPTED_ATTACKER_ACTION
        self._seed: int | None = None

    # ── Environment interface ─────────────────────────────────────────────────

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        role: str = "defender",
        task_id: str | None = None,
        opponent_checkpoint: str | None = None,
        **kwargs,
    ) -> DefenderObservation | AttackerObservation:
        """Reset to a new episode.

        Args:
            seed: Optional RNG seed passed to traffic generator.
            episode_id: Optional explicit episode ID; generated if None.
            role: ``"defender"`` or ``"attacker"``.
            task_id: Scenario task ID; defaults to ``"task1"``.
            opponent_checkpoint: Path to frozen opponent checkpoint
                (Phase 6+; accepted and ignored in Phase 1).

        Returns:
            Initial observation for the active role.
        """
        self._reset_rubric()
        self._nginx = SimulatedNginx()
        self._express = SimulatedExpress()
        self._role = role
        self._task_id = task_id or "task1"
        self._episode_templates_used = set()
        self._attacker_steps_with_no_malicious = 0
        self._total_attacker_steps = 0
        self._last_attacker_action = _SCRIPTED_ATTACKER_ACTION
        self._seed = seed

        ep_id = episode_id or str(uuid4())
        self._arena_state = ArenaState(
            episode_id=ep_id,
            step_count=0,
            current_task=self._task_id,
            current_role=role,
        )

        if role == "defender":
            return DefenderObservation(
                done=False,
                reward=0.0,
                metadata=self._defender_meta(0, 0, 0, 0, 0),
            )
        return AttackerObservation(
            steps_remaining=MAX_EPISODE_STEPS,
            done=False,
            reward=0.0,
            metadata=self._attacker_meta(0, 0, 0, 0.0),
        )

    def step(
        self,
        action: DefenderAction | AttackerAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> DefenderObservation | AttackerObservation:
        """Execute one step for the active role.

        Args:
            action: ``DefenderAction`` or ``AttackerAction`` matching the
                current role.
            timeout_s: Ignored — simulator is synchronous.

        Returns:
            Observation with reward set by ``ArenaRubric`` via
            ``_apply_rubric``.
        """
        self._arena_state.step_count += 1
        step = self._arena_state.step_count
        done = step >= MAX_EPISODE_STEPS

        if self._role == "defender":
            return self._step_defender(action, step, done)  # type: ignore[arg-type]
        return self._step_attacker(action, step, done)  # type: ignore[arg-type]

    @property
    def state(self) -> ArenaState:
        """Current episode state."""
        return self._arena_state

    # ── Role-specific step logic ──────────────────────────────────────────────

    def _step_defender(
        self, action: DefenderAction, step: int, done: bool
    ) -> DefenderObservation:
        # Execute action
        if action.action_type == "append_nginx_rule":
            self._nginx.add_rule(action.rule_text)
        elif action.action_type == "write_express_middleware":
            self._express.add_middleware(action.route, action.middleware_js)

        log_tail = ""
        if action.action_type == "read_log":
            log_tail = self._nginx.get_log_tail(action.log_tail_lines)

        # Generate and process traffic using scripted attacker
        self._nginx.reset_step_counters()
        _attacker_dict_seeded = dict(self._last_attacker_action)
        _attacker_dict_seeded["_seed"] = self._derived_seed(step, 1)
        traffic = generate_episode_traffic(
            self._task_id,
            _attacker_dict_seeded,
            seed=self._derived_seed(step, 0),
        )
        stats = self._process_traffic(traffic)

        obs = DefenderObservation(
            log_tail=log_tail,
            current_rules=self._nginx.get_rules(),
            current_middleware=self._express.get_middleware_summary(),
            episode_step=step,
            last_step_metrics={
                "requests_total": stats["total"],
                "200_count": stats["count_200"],
                "403_count": stats["count_403"],
                "429_count": stats["count_429"],
            },
            done=done,
            metadata=self._defender_meta(
                malicious_blocked=stats["malicious_blocked"],
                total_malicious=stats["total_malicious"],
                legit_allowed=stats["legit_allowed"],
                total_legit=stats["total_legit"],
                unrecognized=self._express.unrecognized_count,
            ),
        )
        obs.reward = self._apply_rubric(action, obs)
        return obs

    def _step_attacker(
        self, action: AttackerAction, step: int, done: bool
    ) -> AttackerObservation:
        self._total_attacker_steps += 1
        self._episode_templates_used.add(action.template)

        attacker_dict: dict = {
            "template": action.template,
            "count": action.count,
            "target_path": action.target_path,
            "source_ips": list(action.source_ips),
            "payload": dict(action.payload),
            "_seed": self._derived_seed(step, 1),
        }
        self._last_attacker_action = attacker_dict

        # Generate and process traffic
        self._nginx.reset_step_counters()
        traffic = generate_episode_traffic(
            self._task_id,
            attacker_dict,
            seed=self._derived_seed(step, 0),
        )
        stats = self._process_traffic(traffic)

        malicious_in_step = sum(1 for r in traffic if r.get("is_malicious"))
        if malicious_in_step == 0:
            self._attacker_steps_with_no_malicious += 1
        legit_decoy_fraction = (
            self._attacker_steps_with_no_malicious / self._total_attacker_steps
        )

        probed_rules: list[str] = []
        if stats["malicious_403"] > 0:
            probed_rules.append(
                f"step {step}: ip/rule block ({stats['malicious_403']} reqs)"
            )
        if stats["malicious_429"] > 0:
            probed_rules.append(
                f"step {step}: rate-limit ({stats['malicious_429']} reqs)"
            )

        obs = AttackerObservation(
            last_response_summary={
                "200": stats["count_200"],
                "403": stats["count_403"],
                "429": stats["count_429"],
            },
            probed_rules=probed_rules,
            episode_step=step,
            steps_remaining=max(0, MAX_EPISODE_STEPS - step),
            done=done,
            metadata=self._attacker_meta(
                malicious_200=stats["malicious_200"],
                malicious_total=malicious_in_step,
                unique_templates=len(self._episode_templates_used),
                legit_decoy_fraction=legit_decoy_fraction,
            ),
        )
        obs.reward = self._apply_rubric(action, obs)
        return obs

    # ── Traffic processing ────────────────────────────────────────────────────

    def _process_traffic(self, traffic: list[dict]) -> dict:
        """Route all requests through nginx then express; return aggregated stats."""
        count_200 = count_403 = count_429 = 0
        malicious_blocked = malicious_200 = malicious_403 = malicious_429 = 0
        legit_allowed = 0
        total_malicious = sum(1 for r in traffic if r.get("is_malicious"))
        total_legit = len(traffic) - total_malicious

        for req in traffic:
            nginx_status = self._nginx.process_request(req)

            if nginx_status == 403:
                status, count_403 = 403, count_403 + 1
            elif nginx_status == 429:
                status, count_429 = 429, count_429 + 1
            else:
                express_status = self._express.process_request(req)
                status = express_status
                if express_status == 200:
                    count_200 += 1
                else:
                    count_403 += 1

            is_mal = req.get("is_malicious", False)
            if is_mal:
                if status == 200:
                    malicious_200 += 1
                elif status == 403:
                    malicious_403 += 1
                    malicious_blocked += 1
                elif status == 429:
                    malicious_429 += 1
                    malicious_blocked += 1
            else:
                if status == 200:
                    legit_allowed += 1

        return {
            "total": len(traffic),
            "count_200": count_200,
            "count_403": count_403,
            "count_429": count_429,
            "total_malicious": total_malicious,
            "total_legit": total_legit,
            "malicious_blocked": malicious_blocked,
            "malicious_200": malicious_200,
            "malicious_403": malicious_403,
            "malicious_429": malicious_429,
            "legit_allowed": legit_allowed,
        }

    # ── Metadata builders ─────────────────────────────────────────────────────

    @staticmethod
    def _defender_meta(
        malicious_blocked: int,
        total_malicious: int,
        legit_allowed: int,
        total_legit: int,
        unrecognized: int,
    ) -> dict:
        return {
            "role": "defender",
            "malicious_blocked": malicious_blocked,
            "total_malicious": total_malicious,
            "legit_allowed": legit_allowed,
            "total_legit": total_legit,
            "unrecognized_middleware_count": unrecognized,
        }

    @staticmethod
    def _attacker_meta(
        malicious_200: int,
        malicious_total: int,
        unique_templates: int,
        legit_decoy_fraction: float,
    ) -> dict:
        return {
            "role": "attacker",
            "malicious_200_count": malicious_200,
            "malicious_total": malicious_total,
            "unique_templates_used": unique_templates,
            "legit_decoy_fraction": legit_decoy_fraction,
        }

    def _derived_seed(self, step: int, channel: int) -> int | None:
        """Return a deterministic per-step seed, or None when episode is unseeded."""
        if self._seed is None:
            return None
        return self._seed * 1_000_003 + step * 997 + channel
