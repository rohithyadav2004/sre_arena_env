"""Tests for training.action_parser — LLM output → Pydantic action."""
from __future__ import annotations

import json

import pytest

from sre_arena_env.models import DefenderAction, AttackerAction
from sre_arena_env.training.action_parser import (
    parse_defender_action,
    parse_attacker_action,
)


# ── Defender tests ─────────────────────────────────────────────────────────────

class TestParseDefenderAction:
    # Happy path: valid JSON for each of 3 action types
    def test_read_log_valid(self):
        raw = '{"action_type": "read_log", "log_tail_lines": 50}'
        action, err = parse_defender_action(raw)
        assert err == ""
        assert isinstance(action, DefenderAction)
        assert action.action_type == "read_log"
        assert action.log_tail_lines == 50

    def test_append_nginx_rule_valid(self):
        raw = '{"action_type": "append_nginx_rule", "rule_text": "deny 10.0.0.1;"}'
        action, err = parse_defender_action(raw)
        assert err == ""
        assert isinstance(action, DefenderAction)
        assert action.action_type == "append_nginx_rule"
        assert action.rule_text == "deny 10.0.0.1;"

    def test_write_express_middleware_valid(self):
        raw = json.dumps({
            "action_type": "write_express_middleware",
            "route": "/api/process",
            "middleware_js": "next();"
        })
        action, err = parse_defender_action(raw)
        assert err == ""
        assert isinstance(action, DefenderAction)
        assert action.route == "/api/process"
        assert action.middleware_js == "next();"

    # Code fence stripping
    def test_json_in_code_fence_stripped(self):
        raw = '```json\n{"action_type": "read_log", "log_tail_lines": 20}\n```'
        action, err = parse_defender_action(raw)
        assert err == ""
        assert isinstance(action, DefenderAction)
        assert action.log_tail_lines == 20

    # Trailing text extraction
    def test_json_with_trailing_text_extracted(self):
        raw = '{"action_type": "read_log"} Some explanation from the LLM.'
        action, err = parse_defender_action(raw)
        assert err == ""
        assert isinstance(action, DefenderAction)

    # Leading text: bracket extractor finds first {
    def test_json_with_leading_text_extracted(self):
        raw = 'I will read the logs now. {"action_type": "read_log"}'
        action, err = parse_defender_action(raw)
        assert err == ""
        assert isinstance(action, DefenderAction)

    # Empty input
    def test_empty_string_returns_none(self):
        action, err = parse_defender_action("")
        assert action is None
        assert err != ""

    # Whitespace-only
    def test_whitespace_only_returns_none(self):
        action, err = parse_defender_action("   \n\t  ")
        assert action is None
        assert err != ""

    # No JSON at all
    def test_plain_text_no_json_returns_none(self):
        action, err = parse_defender_action("I read the logs and everything looks fine.")
        assert action is None
        assert err != ""

    # Malformed JSON (trailing comma — invalid per JSON spec)
    def test_malformed_json_trailing_comma_returns_none(self):
        raw = '{"action_type": "read_log",}'
        action, err = parse_defender_action(raw)
        assert action is None
        assert "JSON parse error" in err

    # Missing action_type field entirely
    def test_missing_action_type_returns_none(self):
        raw = '{"log_tail_lines": 50}'
        action, err = parse_defender_action(raw)
        assert action is None
        assert err != ""

    # Unknown action_type
    def test_unknown_action_type_returns_none(self):
        raw = '{"action_type": "delete_file", "path": "/etc/passwd"}'
        action, err = parse_defender_action(raw)
        assert action is None
        assert "unknown action_type" in err

    # append_nginx_rule without rule_text → semantic validation fails
    def test_append_nginx_rule_without_rule_text_returns_none(self):
        raw = '{"action_type": "append_nginx_rule"}'
        action, err = parse_defender_action(raw)
        assert action is None
        assert err != ""

    # Never raises no matter how bad the input
    def test_never_raises_on_garbage(self):
        garbage_inputs = [
            "}{}{",
            '{"key": ',
            "\x00\x01\x02",
            "not json at all !!!",
        ]
        for raw in garbage_inputs:
            result = parse_defender_action(raw)
            assert isinstance(result, tuple), f"Expected tuple for {raw!r}"
            assert len(result) == 2


# ── Attacker tests ─────────────────────────────────────────────────────────────

class TestParseAttackerAction:
    # Happy path: valid JSON for different templates
    def test_single_ip_flood_valid(self):
        raw = '{"template": "single_ip_flood", "count": 10}'
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)
        assert action.template == "single_ip_flood"

    def test_credential_stuffing_with_count(self):
        raw = '{"template": "credential_stuffing", "count": 5, "target_path": "/login"}'
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)
        assert action.count == 5

    def test_payload_injection_valid(self):
        raw = json.dumps({
            "template": "payload_injection",
            "target_path": "/api/process",
            "payload": {"command": "cat /etc/passwd"}
        })
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)
        assert action.template == "payload_injection"

    # Code fence
    def test_json_in_code_fence_stripped(self):
        raw = '```json\n{"template": "ip_spray"}\n```'
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)

    # Trailing text
    def test_json_with_trailing_text_extracted(self):
        raw = '{"template": "slow_drip"} Explanation follows.'
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)

    # Leading text
    def test_json_with_leading_text_extracted(self):
        raw = 'My attack plan: {"template": "header_spoof"}'
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)

    # Empty input
    def test_empty_string_returns_none(self):
        action, err = parse_attacker_action("")
        assert action is None
        assert err != ""

    # Whitespace-only
    def test_whitespace_only_returns_none(self):
        action, err = parse_attacker_action("  \n  ")
        assert action is None
        assert err != ""

    # No JSON at all
    def test_plain_text_no_json_returns_none(self):
        action, err = parse_attacker_action("I will flood the server with requests.")
        assert action is None
        assert err != ""

    # Malformed JSON
    def test_malformed_json_returns_none(self):
        raw = '{"template": "ip_spray",}'
        action, err = parse_attacker_action(raw)
        assert action is None
        assert "JSON parse error" in err

    # Missing required field (template has no default — Pydantic Literal required)
    def test_missing_template_field_returns_none(self):
        raw = '{"count": 10, "target_path": "/login"}'
        action, err = parse_attacker_action(raw)
        assert action is None
        assert err != ""

    # Unknown template (Pydantic Literal validation rejects it)
    def test_unknown_template_returns_none(self):
        raw = '{"template": "ddos_attack", "count": 100}'
        action, err = parse_attacker_action(raw)
        assert action is None
        assert err != ""

    # All optional fields specified — full payload
    def test_all_optional_fields_specified_valid(self):
        raw = json.dumps({
            "template": "mixed_legit_cover",
            "count": 20,
            "target_path": "/api/data",
            "source_ips": ["1.2.3.4"],
            "payload": {},
            "delay_ms": 0,
        })
        action, err = parse_attacker_action(raw)
        assert err == ""
        assert isinstance(action, AttackerAction)
        assert action.template == "mixed_legit_cover"
        assert action.count == 20

    # Never raises
    def test_never_raises_on_garbage(self):
        garbage_inputs = [
            "}{}{",
            '{"key": ',
            "\x00\x01\x02",
            "{'template': 'bad'}",  # single quotes — invalid JSON
        ]
        for raw in garbage_inputs:
            result = parse_attacker_action(raw)
            assert isinstance(result, tuple), f"Expected tuple for {raw!r}"
            assert len(result) == 2
