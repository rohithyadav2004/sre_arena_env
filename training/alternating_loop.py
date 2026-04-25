"""Alternating best-response training orchestrator.

Usage:
    python -m training.alternating_loop --config configs/colab_demo.yaml [--dry-run]

Orchestrates N generations of alternating defender/attacker training.
Imports train_defender and train_attacker as regular functions (Option A:
in-process). GPU code is deferred inside each training function — this
module is importable and testable on CPU.

--dry-run logs what each generation would do without loading models.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

try:
    from .train_defender import train_defender
    from .train_attacker import train_attacker
except ImportError:
    from training.train_defender import train_defender
    from training.train_attacker import train_attacker


def load_config(config_path: str) -> dict:
    """Load and return a YAML training config.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed config dict.

    Raises:
        FileNotFoundError: If config_path does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(path) as f:
        return yaml.safe_load(f)


def get_opponent_checkpoint(
    gen_idx: int,
    role: str,
    output_dir: str,
) -> Optional[str]:
    """Return the checkpoint path from the previous generation, or None for gen 0.

    Args:
        gen_idx: Current generation index (0-based).
        role: The role being trained this generation ("defender" or "attacker").
        output_dir: Base output directory from config.

    Returns:
        Path string of the opponent's checkpoint, or None if gen_idx == 0.
    """
    if gen_idx == 0:
        return None
    prev_gen = gen_idx - 1
    opponent_role = "attacker" if role == "defender" else "defender"
    return f"{output_dir}_{opponent_role}_gen{prev_gen}"


def _role_for_gen(gen_idx: int, per_gen: list[dict]) -> str:
    """Return the role to train for generation gen_idx.

    Uses the per_generation config list if available; falls back to
    even=defender / odd=attacker alternation.

    Args:
        gen_idx: Generation index (0-based).
        per_gen: List of per-generation dicts from config (may be empty).

    Returns:
        "defender" or "attacker".
    """
    if gen_idx < len(per_gen):
        return per_gen[gen_idx]["role"]
    return "defender" if gen_idx % 2 == 0 else "attacker"


def run_alternating_loop(cfg: dict, dry_run: bool = False) -> None:
    """Execute the alternating best-response training loop.

    Args:
        cfg: Parsed YAML config dict. Must contain "num_generations" (int),
            "per_generation" (list[dict]), and "training.output_dir" (str).
        dry_run: If True, log what each generation would do without calling
            any training function or loading any model.
    """
    num_gens: int = cfg.get("num_generations", 3)
    per_gen: list[dict] = cfg.get("per_generation", [])
    output_dir: str = cfg["training"]["output_dir"]

    logger.info("Starting alternating loop: %d generations (dry_run=%s)", num_gens, dry_run)

    for gen_idx in range(num_gens):
        role = _role_for_gen(gen_idx, per_gen)
        opponent_ckpt = get_opponent_checkpoint(gen_idx, role, output_dir)

        logger.info(
            "Gen %d: training %s against %s",
            gen_idx, role, opponent_ckpt or "scripted baseline",
        )

        if dry_run:
            continue

        if role == "defender":
            ckpt = train_defender(cfg, opponent_checkpoint=opponent_ckpt, gen_idx=gen_idx)
        else:
            ckpt = train_attacker(cfg, opponent_checkpoint=opponent_ckpt, gen_idx=gen_idx)

        logger.info("Gen %d complete: checkpoint -> %s", gen_idx, ckpt)

    logger.info("Alternating loop complete.")


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    parser = argparse.ArgumentParser(
        description="Alternating best-response training loop"
    )
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log generation plan without running training",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    run_alternating_loop(cfg, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
