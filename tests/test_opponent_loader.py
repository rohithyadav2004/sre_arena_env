"""Tests for OpponentModel — CPU only, GPU model mocked via MagicMock."""
from __future__ import annotations

from unittest.mock import MagicMock

import torch
import pytest

from sre_arena_env.training.opponent_loader import OpponentModel
from sre_arena_env.training.scripted_attacker import ScriptedAttacker
from sre_arena_env.training.scripted_defender import ScriptedDefender


# ── Helpers ──────────────────────────────────────────────────────────────────


def _make_mock_pair(decode_output: str, input_len: int = 3) -> tuple[MagicMock, MagicMock]:
    """Return (mock_model, mock_tokenizer) whose generate() yields decode_output."""
    mock_tok = MagicMock()
    mock_tok.eos_token_id = 0
    # tokenizer(prompt, return_tensors='pt') → dict with real tensor so .to(device) works
    mock_tok.return_value = {"input_ids": torch.zeros(1, input_len, dtype=torch.long)}
    mock_tok.decode.return_value = decode_output

    mock_model = MagicMock()
    # generate() must return a tensor; shape (1, input_len + 5) simulates 5 new tokens
    mock_model.generate.return_value = torch.zeros(1, input_len + 5, dtype=torch.long)
    # parameters() used to detect device — return a list so iter() works repeatedly
    mock_param = MagicMock()
    mock_param.device = torch.device("cpu")
    mock_model.parameters.return_value = [mock_param]

    return mock_model, mock_tok


# ── Tests ─────────────────────────────────────────────────────────────────────


class TestOpponentModel:
    def test_generate_action_attacker_returns_dict_on_valid_output(self):
        valid_json = (
            '{"template": "single_ip_flood", "count": 10, "target_path": "/login"}'
        )
        model, tok = _make_mock_pair(valid_json)
        opponent = OpponentModel(model, tok, role="attacker")
        result = opponent.generate_action()
        assert isinstance(result, dict)
        assert result["template"] == "single_ip_flood"

    def test_generate_action_attacker_returns_fallback_on_parse_failure(self):
        model, tok = _make_mock_pair("this is not valid json at all ~~~")
        opponent = OpponentModel(model, tok, role="attacker")
        result = opponent.generate_action()
        expected = ScriptedAttacker(seed=42).act().model_dump(
            exclude={"delay_ms", "metadata"}
        )
        assert result == expected

    def test_generate_action_defender_returns_dict_on_valid_output(self):
        valid_json = '{"action_type": "read_log", "log_tail_lines": 20}'
        model, tok = _make_mock_pair(valid_json)
        opponent = OpponentModel(model, tok, role="defender")
        result = opponent.generate_action()
        assert isinstance(result, dict)
        assert result["action_type"] == "read_log"

    def test_generate_action_defender_returns_fallback_on_parse_failure(self):
        model, tok = _make_mock_pair("garbage garbage garbage")
        opponent = OpponentModel(model, tok, role="defender")
        result = opponent.generate_action()
        expected = ScriptedDefender(seed=42).act().model_dump()
        assert result == expected

    def test_model_set_to_eval_mode_on_init(self):
        model, tok = _make_mock_pair("")
        OpponentModel(model, tok, role="attacker")
        model.eval.assert_called_once()

    def test_no_grad_active_during_generate(self):
        """torch.no_grad() must be the active context when model.generate() runs."""
        model, tok = _make_mock_pair("")
        no_grad_snapshots: list[bool] = []

        def capture_no_grad(*args, **kwargs):
            # Inside torch.no_grad() grad is disabled, so is_grad_enabled() == False
            no_grad_snapshots.append(not torch.is_grad_enabled())
            return torch.zeros(1, 8, dtype=torch.long)

        model.generate.side_effect = capture_no_grad
        opponent = OpponentModel(model, tok, role="attacker")
        opponent.generate_action()

        assert no_grad_snapshots == [True], (
            "model.generate() must be called inside torch.no_grad()"
        )
