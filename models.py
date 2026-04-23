# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Data models for the SRE Arena Env environment.

Two LLM agents compete on simulated web infrastructure. A defender (blue)
protects an nginx+express service; an attacker (red) tries to get malicious
requests through. Both share one environment class, differentiated by ``role``
on ``reset()``.
"""

from __future__ import annotations

from enum import Enum
from typing import Literal

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


class Role(str, Enum):
    """Agent role within one episode."""

    DEFENDER = "defender"
    ATTACKER = "attacker"


class DefenderAction(Action):
    """Action for the blue (defender) agent.

    Exactly three action types are supported. Fields not relevant to the
    chosen ``action_type`` are ignored by the environment.

    Attributes:
        action_type: Which defender action to execute.
        log_tail_lines: For ``read_log`` — how many trailing log lines to return.
        rule_text: For ``append_nginx_rule`` — the nginx directive to append,
            e.g. ``"deny 1.2.3.4;"`` or ``"limit_req zone=flood burst=5 nodelay;"``.
        route: For ``write_express_middleware`` — target route, e.g. ``"/api/process"``.
        middleware_js: For ``write_express_middleware`` — JS source that must
            match one of ``RECOGNIZED_MIDDLEWARE_PATTERNS`` in sim_express.
    """

    action_type: Literal[
        "read_log",
        "append_nginx_rule",
        "write_express_middleware",
    ] = Field(..., description="Defender action type")
    log_tail_lines: int = Field(
        default=50, ge=1, description="Number of log lines to return for read_log"
    )
    rule_text: str = Field(
        default="", description="Nginx rule text for append_nginx_rule"
    )
    route: str = Field(
        default="", description="Express route for write_express_middleware"
    )
    middleware_js: str = Field(
        default="",
        description="JS middleware source for write_express_middleware",
    )


class DefenderObservation(Observation):
    """Observation returned to the blue (defender) agent after each step.

    The base ``Observation`` class already carries ``done``, ``reward``, and
    ``metadata``. This subclass adds defender-specific fields.

    The ``metadata`` dict is populated by the environment with the fields
    required by ``DefenderRubric``:
    - ``malicious_blocked: int``
    - ``total_malicious: int``
    - ``legit_allowed: int``
    - ``total_legit: int``
    - ``unrecognized_middleware_count: int``
    - ``role: "defender"``

    Attributes:
        log_tail: Trailing nginx log lines (populated for ``read_log`` actions).
        current_rules: All nginx rules written so far this episode.
        current_middleware: Mapping of route → latest JS middleware source.
        episode_step: Step index within the current episode (1-based).
        last_step_metrics: Per-step traffic counters keyed by HTTP status.
    """

    log_tail: str = Field(default="", description="Trailing nginx log lines")
    current_rules: list[str] = Field(
        default_factory=list, description="Nginx rules written this episode"
    )
    current_middleware: dict[str, str] = Field(
        default_factory=dict,
        description="route -> latest JS middleware source",
    )
    episode_step: int = Field(default=0, description="Current step index (1-based)")
    last_step_metrics: dict = Field(
        default_factory=dict,
        description="Traffic counters: requests_total, 200_count, 403_count, 429_count",
    )


class AttackerAction(Action):
    """Action for the red (attacker) agent.

    Exactly eight attack templates are available. The attacker specifies which
    template to use and parameterises it; the environment generates the
    corresponding batch of HTTP requests.

    Attributes:
        template: Which of the 8 attack templates to execute.
        count: Number of malicious requests to generate.
        target_path: Primary path to target.
        source_ips: Explicit source IPs; the environment generates defaults if empty.
        payload: Template-specific payload (used by ``payload_injection``).
        delay_ms: Inter-request delay in ms — accepted but ignored by simulator.
    """

    template: Literal[
        "single_ip_flood",
        "ip_spray",
        "credential_stuffing",
        "payload_injection",
        "header_spoof",
        "slow_drip",
        "path_traversal",
        "mixed_legit_cover",
    ] = Field(..., description="Attack template name")
    count: int = Field(default=10, ge=1, description="Number of requests to generate")
    target_path: str = Field(default="/login", description="Primary target path")
    source_ips: list[str] = Field(
        default_factory=list,
        description="Explicit source IPs (generated if empty)",
    )
    payload: dict = Field(
        default_factory=dict,
        description="Template-specific payload for payload_injection",
    )
    delay_ms: int = Field(
        default=0, ge=0, description="Inter-request delay ms (simulator ignores)"
    )


class AttackerObservation(Observation):
    """Observation returned to the red (attacker) agent after each step.

    The ``metadata`` dict is populated with fields required by ``AttackerRubric``:
    - ``malicious_200_count: int``
    - ``malicious_total: int``
    - ``unique_templates_used: int``
    - ``legit_decoy_fraction: float``
    - ``role: "attacker"``

    Attributes:
        last_response_summary: Breakdown of response codes for last step,
            e.g. ``{"200": 5, "403": 3, "429": 2}``.
        probed_rules: Defender rules the attacker inferred from response codes.
        episode_step: Step index within the current episode (1-based).
        steps_remaining: Steps left before the episode terminates.
    """

    last_response_summary: dict = Field(
        default_factory=dict,
        description='Response code breakdown e.g. {"200": 5, "403": 3}',
    )
    probed_rules: list[str] = Field(
        default_factory=list,
        description="Defender rules inferred from response codes",
    )
    episode_step: int = Field(default=0, description="Current step index (1-based)")
    steps_remaining: int = Field(
        default=0, description="Steps remaining in this episode"
    )


class ArenaState(State):
    """Environment state exposed via ``env.state``.

    Extends the base ``State`` (which has ``episode_id`` and ``step_count``)
    with arena-specific fields.

    Attributes:
        current_task: Scenario task ID (e.g. ``"task1"``).
        current_role: Agent role active this episode (``"defender"`` or ``"attacker"``).
    """

    current_task: str = Field(default="task1", description="Scenario task ID")
    current_role: str = Field(
        default="defender", description='Active role: "defender" or "attacker"'
    )
