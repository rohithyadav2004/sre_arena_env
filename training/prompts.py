"""System prompts and observation formatting for LLM agent training."""
from __future__ import annotations

try:
    from ..models import DefenderObservation, AttackerObservation
except ImportError:
    from models import DefenderObservation, AttackerObservation


DEFENDER_SYSTEM_PROMPT: str = (
    "You are a blue-team SRE agent protecting an nginx+express web service.\n"
    "Each turn, choose exactly one action and output ONLY a JSON object — no preamble.\n\n"
    "Actions:\n"
    '1. read_log            : {"action_type": "read_log", "log_tail_lines": 50}\n'
    '2. append_nginx_rule   : {"action_type": "append_nginx_rule", "rule_text": "deny 1.2.3.4;"}\n'
    "3. write_express_middleware: {\"action_type\": \"write_express_middleware\", "
    "\"route\": \"/api/process\", \"middleware_js\": \"if(req.body.x){return res.status(403).end();}next();\"}\n\n"
    "Output ONLY valid JSON. No explanation."
)

ATTACKER_SYSTEM_PROMPT: str = (
    "You are a red-team attacker finding weaknesses in an nginx+express web service.\n"
    "Each turn, choose exactly one template and output ONLY a JSON object — no preamble.\n\n"
    "Templates: single_ip_flood, ip_spray, credential_stuffing, payload_injection, "
    "header_spoof, slow_drip, path_traversal, mixed_legit_cover\n\n"
    'Schema: {"template": "<name>", "count": 10, "target_path": "/login", '
    '"source_ips": [], "payload": {}}\n\n'
    "Output ONLY valid JSON. No explanation."
)


def build_defender_prompt(observation: DefenderObservation) -> str:
    """Build SYSTEM/USER/ASSISTANT prompt for the defender from its observation."""
    metrics = observation.last_step_metrics
    status_200 = metrics.get("200_count", 0)
    status_403 = metrics.get("403_count", 0)
    status_429 = metrics.get("429_count", 0)

    log_lines = (
        "\n".join(f"  {line}" for line in observation.log_tail.splitlines())
        if observation.log_tail
        else "  (empty)"
    )
    rules = ", ".join(observation.current_rules) if observation.current_rules else "(none)"
    middleware = (
        ", ".join(observation.current_middleware.keys())
        if observation.current_middleware
        else "(none)"
    )

    user_turn = (
        f"Episode step: {observation.episode_step}\n"
        f"Access log tail:\n{log_lines}\n"
        f"Status code summary: 200={status_200}, 403={status_403}, 429={status_429}\n"
        f"Current nginx rules: [{rules}]\n"
        f"Current middleware: [{middleware}]"
    )
    return f"SYSTEM: {DEFENDER_SYSTEM_PROMPT}\nUSER:\n{user_turn}\nASSISTANT:\n"


def build_attacker_prompt(observation: AttackerObservation) -> str:
    """Build SYSTEM/USER/ASSISTANT prompt for the attacker from its observation."""
    resp = observation.last_response_summary
    resp_str = (
        ", ".join(f"{k}={v}" for k, v in resp.items()) if resp else "(none)"
    )
    rules = ", ".join(observation.probed_rules) if observation.probed_rules else "(none)"

    user_turn = (
        f"Episode step: {observation.episode_step}\n"
        f"Steps remaining: {observation.steps_remaining}\n"
        f"Last response summary: {resp_str}\n"
        f"Probed rules: [{rules}]"
    )
    return f"SYSTEM: {ATTACKER_SYSTEM_PROMPT}\nUSER:\n{user_turn}\nASSISTANT:\n"
