"""Hackathon inference script — defender + attacker episodes.

Usage:
    HF_TOKEN=<tok> python inference.py

Env vars (all optional except HF_TOKEN / OPENAI_API_KEY):
    API_BASE_URL   default https://router.huggingface.co/v1
    MODEL_NAME     default Qwen/Qwen2.5-7B-Instruct
    HF_TOKEN       required (or OPENAI_API_KEY as fallback)
    OPENENV_URL    default http://localhost:8000
    EPISODE_SEED   default 42
    MAX_STEPS      default 8
"""
from __future__ import annotations

import json
import os
import sys
from typing import Any

from openai import OpenAI
from pydantic import ValidationError

try:
    from .client import SreArenaEnvClient
    from .models import AttackerAction, AttackerObservation, DefenderAction, DefenderObservation
except ImportError:
    from client import SreArenaEnvClient  # type: ignore[no-redef]
    from models import AttackerAction, AttackerObservation, DefenderAction, DefenderObservation  # type: ignore[no-redef]

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL: str = os.environ.get("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME: str = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-7B-Instruct")
OPENENV_URL: str = os.environ.get("OPENENV_URL", "http://localhost:8000")
EPISODE_SEED: int = int(os.environ.get("EPISODE_SEED", "42"))
MAX_STEPS: int = int(os.environ.get("MAX_STEPS", "8"))

_token = os.environ.get("HF_TOKEN") or os.environ.get("OPENAI_API_KEY", "")

# ---------------------------------------------------------------------------
# System prompts
# ---------------------------------------------------------------------------

DEFENDER_SYSTEM: str = """You are a blue-team SRE agent protecting an nginx+express web service.
Each turn choose exactly one action and output ONLY a JSON object — no markdown, no preamble.

Available actions:

1. read_log — inspect recent traffic
   {"action_type": "read_log", "log_tail_lines": 50}

2. append_nginx_rule — block an IP or add a rate-limit directive
   {"action_type": "append_nginx_rule", "rule_text": "deny 10.0.1.1;"}

3. write_express_middleware — install JS middleware on a route
   {"action_type": "write_express_middleware", "route": "/api/process", "middleware_js": "if(req.body.x){return res.status(403).end();}next();"}

Reward formula: (malicious_blocked / total_malicious) x (legit_allowed / total_legit) - middleware_penalty
Maximise blocking of malicious traffic while keeping legit traffic through.

Output ONLY valid JSON, no markdown, no preamble."""

ATTACKER_SYSTEM: str = """You are a red-team attacker trying to get malicious HTTP requests through an nginx+express web service.
Each turn choose exactly one template and output ONLY a JSON object — no markdown, no preamble.

Available templates:
  single_ip_flood     — flood from one IP (easy to block, high volume)
  ip_spray            — spread requests across many IPs (evades IP bans)
  credential_stuffing — POST /login with many user/pass combos
  payload_injection   — craft malicious request body payload
  header_spoof        — spoof X-Forwarded-For / User-Agent headers
  slow_drip           — low-rate trickle to avoid rate limiting
  path_traversal      — attempt directory traversal paths
  mixed_legit_cover   — hide malicious requests among legit-looking ones

Schema:
{"template": "<name>", "count": 10, "target_path": "/login", "source_ips": [], "payload": {}}

Reward formula: HTTP 200 responses count as successes + diversity bonus for using varied templates.
Maximise 200 responses and use varied templates.

Output ONLY valid JSON, no markdown, no preamble."""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers that models emit."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        # drop first line (```json or ```) and last line (```)
        inner = lines[1:] if lines[-1].strip() == "```" else lines[1:]
        if inner and inner[-1].strip() == "```":
            inner = inner[:-1]
        text = "\n".join(inner).strip()
    return text


def _call_llm(client: OpenAI, system: str, user_msg: str) -> str:
    """Call the LLM and return raw content string."""
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user_msg},
        ],
        max_tokens=256,
        temperature=0.0,
    )
    return response.choices[0].message.content or ""


def _format_defender_obs(obs: DefenderObservation) -> str:
    metrics = obs.last_step_metrics or {}
    s200 = metrics.get("200_count", 0)
    s403 = metrics.get("403_count", 0)
    s429 = metrics.get("429_count", 0)
    log_lines = (
        "\n".join(f"  {line}" for line in obs.log_tail.splitlines())
        if obs.log_tail
        else "  (empty)"
    )
    rules = ", ".join(obs.current_rules) if obs.current_rules else "(none)"
    mw = ", ".join(obs.current_middleware.keys()) if obs.current_middleware else "(none)"
    return (
        f"Episode step: {obs.episode_step}\n"
        f"Access log tail:\n{log_lines}\n"
        f"Status code summary: 200={s200}, 403={s403}, 429={s429}\n"
        f"Current nginx rules: [{rules}]\n"
        f"Current middleware routes: [{mw}]\n"
        f"Reward so far: {obs.reward}\n"
        "Choose your next action."
    )


def _format_attacker_obs(obs: AttackerObservation) -> str:
    resp = obs.last_response_summary or {}
    resp_str = ", ".join(f"{k}={v}" for k, v in resp.items()) if resp else "(none)"
    rules = ", ".join(obs.probed_rules) if obs.probed_rules else "(none)"
    return (
        f"Episode step: {obs.episode_step}\n"
        f"Steps remaining: {obs.steps_remaining}\n"
        f"Last response summary: {resp_str}\n"
        f"Defender rules inferred: [{rules}]\n"
        f"Reward so far: {obs.reward}\n"
        "Choose your next attack template."
    )


def _parse_defender_action(raw: str) -> tuple[DefenderAction, str | None]:
    """Return (action, error_msg). error_msg is None on success."""
    try:
        data = json.loads(_strip_fences(raw))
        return DefenderAction(**data), None
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        fallback = DefenderAction(action_type="read_log")
        return fallback, str(exc)


def _parse_attacker_action(raw: str) -> tuple[AttackerAction, str | None]:
    """Return (action, error_msg). error_msg is None on success."""
    try:
        data = json.loads(_strip_fences(raw))
        return AttackerAction(**data), None
    except (json.JSONDecodeError, ValidationError, TypeError) as exc:
        fallback = AttackerAction(template="single_ip_flood", count=10, target_path="/login")
        return fallback, str(exc)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------


def run_episode(llm: OpenAI, role: str, task_name: str, seed: int) -> None:
    """Run one complete episode, printing [START]/[STEP]/[END] lines."""
    print(f"[START] task={task_name} env=sre_arena_env model={MODEL_NAME}", flush=True)
    try:
        rewards: list[float] = []
        steps_taken = 0

        sync_env = SreArenaEnvClient(OPENENV_URL).sync()
        with sync_env:
            result = sync_env.reset(role=role, seed=seed)
            obs = result.observation

            for step in range(1, MAX_STEPS + 1):
                steps_taken = step

                # Build LLM prompt and call model
                if role == "defender":
                    assert isinstance(obs, DefenderObservation)
                    user_msg = _format_defender_obs(obs)
                    try:
                        raw = _call_llm(llm, DEFENDER_SYSTEM, user_msg)
                        action, parse_err = _parse_defender_action(raw)
                    except Exception as llm_exc:
                        action = DefenderAction(action_type="read_log")
                        parse_err = str(llm_exc)
                    action_label = action.action_type if parse_err is None else "parse_error"
                else:
                    assert isinstance(obs, AttackerObservation)
                    user_msg = _format_attacker_obs(obs)
                    try:
                        raw = _call_llm(llm, ATTACKER_SYSTEM, user_msg)
                        action, parse_err = _parse_attacker_action(raw)
                    except Exception as llm_exc:
                        action = AttackerAction(template="single_ip_flood", count=10, target_path="/login")
                        parse_err = str(llm_exc)
                    action_label = action.template if parse_err is None else "parse_error"

                # Step the environment
                result = sync_env.step(action)
                obs = result.observation
                reward = result.reward if result.reward is not None else 0.0
                done = result.done
                rewards.append(float(reward))

                error_str = parse_err if parse_err else "null"
                print(
                    f"[STEP] step={step} action={action_label} "
                    f"reward={reward:.4f} done={str(done).lower()} error={error_str}",
                    flush=True,
                )

                if done:
                    break

        score = max(rewards) if rewards else 0.0
        score = max(0.0, min(1.0, score))
        rewards_str = ",".join(f"{r:.4f}" for r in rewards)
        print(
            f"[END] success={str(score > 0).lower()} steps={steps_taken} "
            f"score={score:.4f} rewards={rewards_str}",
            flush=True,
        )

    except Exception as exc:
        print(f"[END] success=false steps=0 score=0.0000 rewards=0.0000", flush=True)
        print(f"WARNING: run_episode({role!r}) crashed: {exc}", file=sys.stderr, flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    if not _token:
        print("ERROR: set HF_TOKEN or OPENAI_API_KEY", file=sys.stderr)
        sys.exit(1)

    llm = OpenAI(api_key=_token, base_url=API_BASE_URL)

    run_episode(llm, role="defender", task_name="defender_blocking", seed=EPISODE_SEED)
    run_episode(llm, role="attacker", task_name="attacker_evasion", seed=EPISODE_SEED + 1)


if __name__ == "__main__":
    main()
