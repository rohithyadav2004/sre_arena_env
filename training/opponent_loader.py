"""OpponentModel — frozen PEFT checkpoint loader for alternating best-response training.

Loads a prior-generation LoRA checkpoint as a frozen opponent during training.
GPU imports (transformers, peft, bitsandbytes) are deferred inside
``from_checkpoint`` so this module is importable on CPU-only machines.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

try:
    from ..models import AttackerObservation, DefenderObservation
    from .prompts import build_attacker_prompt, build_defender_prompt
    from .action_parser import parse_attacker_action, parse_defender_action
    from .scripted_attacker import ScriptedAttacker
    from .scripted_defender import ScriptedDefender
except ImportError:
    from models import AttackerObservation, DefenderObservation
    from training.prompts import build_attacker_prompt, build_defender_prompt
    from training.action_parser import parse_attacker_action, parse_defender_action
    from training.scripted_attacker import ScriptedAttacker
    from training.scripted_defender import ScriptedDefender

_VALID_ROLES = frozenset({"attacker", "defender"})


class OpponentModel:
    """Frozen opponent backed by a (possibly PEFT-wrapped) HuggingFace model.

    Use ``from_checkpoint`` to load from a LoRA checkpoint directory.
    Use ``__init__`` directly when passing pre-loaded mocks (e.g. in tests).
    """

    def __init__(self, model, tokenizer, role: str) -> None:
        """Store model/tokenizer, call eval(), validate role.

        Args:
            model: A HuggingFace-compatible model object with ``.generate()``
                and ``.parameters()``.
            tokenizer: A HuggingFace-compatible tokenizer object.
            role: Either ``"attacker"`` or ``"defender"``.

        Raises:
            ValueError: If ``role`` is not ``"attacker"`` or ``"defender"``.
        """
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)!r}, got {role!r}"
            )
        self._model = model
        self._tokenizer = tokenizer
        self._role = role
        self._model.eval()
        logger.debug("OpponentModel initialised (role=%s)", role)

    @classmethod
    def from_checkpoint(
        cls,
        base_model_name: str,
        checkpoint_path: str,
        role: str,
        tokenizer,
    ) -> "OpponentModel":
        """Load a 4-bit quantised base model and wrap it with a PEFT LoRA adapter.

        GPU-specific imports are deferred here so the module stays importable
        on CPU-only machines (e.g. during unit tests or dashboard runs).

        Args:
            base_model_name: HuggingFace model ID, e.g. ``"Qwen/Qwen2.5-7B-Instruct"``.
            checkpoint_path: Local path or HF repo ID for the LoRA adapter.
            role: ``"attacker"`` or ``"defender"``.
            tokenizer: Pre-loaded tokenizer (avoids double-loading).

        Returns:
            An ``OpponentModel`` instance ready for inference.
        """
        if role not in _VALID_ROLES:
            raise ValueError(
                f"role must be one of {sorted(_VALID_ROLES)!r}, got {role!r}"
            )
        import torch  # noqa: F401 — deferred GPU import
        from transformers import AutoModelForCausalLM, BitsAndBytesConfig
        from peft import PeftModel

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        logger.info(
            "Loading base model %s with 4-bit quantisation…", base_model_name
        )
        base = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
        )
        logger.info("Wrapping with PEFT adapter from %s…", checkpoint_path)
        model = PeftModel.from_pretrained(base, checkpoint_path)
        return cls(model, tokenizer, role)

    # ── Inference ────────────────────────────────────────────────────────────

    def generate_action(self) -> dict:
        """Generate an action dict from a static, blind (no observation) prompt.

        Returns:
            A dict that is the ``model_dump()`` of the parsed action, or the
            scripted fallback if parsing fails.
        """
        import torch

        prompt = self._build_prompt()
        inputs = self._tokenizer(prompt, return_tensors="pt")
        device = next(iter(self._model.parameters())).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=self._tokenizer.eos_token_id,
            )

        new_tokens = outputs[0][input_len:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        logger.debug("OpponentModel raw output (role=%s): %r", self._role, text)
        return self._parse_or_fallback(text)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _build_prompt(self) -> str:
        """Build a minimal static observation prompt for blind generation."""
        if self._role == "attacker":
            obs = AttackerObservation(episode_step=0, steps_remaining=50)
            return build_attacker_prompt(obs)
        else:
            obs = DefenderObservation(episode_step=0)
            return build_defender_prompt(obs)

    def _parse_or_fallback(self, text: str) -> dict:
        """Parse LLM output into a dict; return scripted fallback on failure.

        Args:
            text: Raw decoded text from the model.

        Returns:
            Parsed action dict, or scripted fallback dict.
        """
        if self._role == "attacker":
            action, err = parse_attacker_action(text)
            if err or action is None:
                logger.debug("Attacker parse failed (%s); using scripted fallback", err)
                return ScriptedAttacker(seed=42).act().model_dump(
                    exclude={"delay_ms", "metadata"}
                )
            return action.model_dump(exclude={"delay_ms", "metadata"})
        else:
            action, err = parse_defender_action(text)
            if err or action is None:
                logger.debug("Defender parse failed (%s); using scripted fallback", err)
                return ScriptedDefender(seed=42).act().model_dump()
            return action.model_dump()
