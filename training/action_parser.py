"""LLM output parser — converts raw string to validated Pydantic action."""
from __future__ import annotations

import json
import re

from pydantic import ValidationError

try:
    from ..models import DefenderAction, AttackerAction
except ImportError:
    from models import DefenderAction, AttackerAction


_FENCE_RE = re.compile(r'```(?:json)?\s*(\{.*?\})\s*```', re.DOTALL)

_VALID_DEFENDER_TYPES = frozenset({"read_log", "append_nginx_rule", "write_express_middleware"})

# Semantic required fields: Pydantic allows empty strings by default; enforce non-empty
_DEFENDER_SEMANTIC: dict[str, list[str]] = {
    "append_nginx_rule": ["rule_text"],
    "write_express_middleware": ["route", "middleware_js"],
}


def _extract_json_str(text: str) -> tuple[str | None, str]:
    """Return (json_str, error). Tries code fence first, then first balanced {…} block."""
    if not text or not text.strip():
        return None, "empty input"

    m = _FENCE_RE.search(text)
    if m:
        return m.group(1), ""

    start = text.find("{")
    if start == -1:
        return None, "no JSON object found in output"

    depth = 0
    for i, ch in enumerate(text[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1], ""

    return None, "unmatched braces in output"


def parse_defender_action(raw_output: str) -> tuple[DefenderAction | None, str]:
    """Parse raw LLM output into a DefenderAction.

    Returns (action, "") on success, (None, error_description) on failure.
    Never raises an exception.
    """
    json_str, err = _extract_json_str(raw_output)
    if err:
        return None, err

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    action_type = data.get("action_type")
    if action_type not in _VALID_DEFENDER_TYPES:
        return None, f"unknown action_type: {action_type!r}"

    for field in _DEFENDER_SEMANTIC.get(action_type, []):
        if not data.get(field, ""):
            return None, f"missing required field for {action_type!r}: {field!r}"

    try:
        action = DefenderAction(**data)
    except ValidationError as exc:
        return None, str(exc)

    return action, ""


def parse_attacker_action(raw_output: str) -> tuple[AttackerAction | None, str]:
    """Parse raw LLM output into an AttackerAction.

    Returns (action, "") on success, (None, error_description) on failure.
    Never raises an exception.
    """
    json_str, err = _extract_json_str(raw_output)
    if err:
        return None, err

    try:
        data = json.loads(json_str)
    except json.JSONDecodeError as exc:
        return None, f"JSON parse error: {exc}"

    try:
        action = AttackerAction(**data)
    except ValidationError as exc:
        return None, str(exc)

    return action, ""
