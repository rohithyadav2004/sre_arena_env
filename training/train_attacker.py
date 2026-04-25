"""Attacker GRPO training entrypoint.

Usage:
    python -m training.train_attacker --config configs/colab_demo.yaml

Mirror of train_defender.py for the red (attacker) role.
GPU is required to run training. All GPU-specific imports are deferred
inside train_attacker() so the module is importable on CPU-only machines.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

try:
    from ..models import AttackerObservation
    from ..server.sre_arena_env_environment import SreArenaEnvironment
    from .dataset_builder import build_attacker_rollout_dataset
    from .reward_function import (
        make_attacker_reward_function,
        make_attacker_reward_function_with_opponent,
    )
except ImportError:
    from models import AttackerObservation
    from server.sre_arena_env_environment import SreArenaEnvironment
    from training.dataset_builder import build_attacker_rollout_dataset
    from training.reward_function import (
        make_attacker_reward_function,
        make_attacker_reward_function_with_opponent,
    )


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


def _collect_attacker_observations(
    cfg: dict,
) -> tuple[list[AttackerObservation], list[int]]:
    """Collect one initial AttackerObservation per training episode.

    For each episode:
    1. Reset env as "attacker" with a deterministic seed.
    2. Collect the initial AttackerObservation as the training prompt.

    The env starts with no defender rules (fresh reset). This represents the
    state after the scripted defender's first read_log (which is a no-op on
    nginx state). The reward function replays each episode from its seed.

    Args:
        cfg: Parsed YAML config dict.

    Returns:
        Tuple of (observations, episode_seeds), both same length.
    """
    num_episodes: int = cfg["training"]["num_episodes"]
    base_seed: int = cfg["env"]["base_seed"]
    task_id: str = cfg["env"]["task_id"]

    env = SreArenaEnvironment()
    observations: list[AttackerObservation] = []
    episode_seeds: list[int] = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        initial_obs = env.reset(role="attacker", seed=seed, task_id=task_id)
        observations.append(initial_obs)
        episode_seeds.append(seed)

        if (ep + 1) % 10 == 0:
            logger.info("Collected %d / %d attacker episodes", ep + 1, num_episodes)

    return observations, episode_seeds


def train_attacker(
    cfg: dict,
    opponent_checkpoint: str | None = None,
    gen_idx: int = 0,
) -> str:
    """Run one generation of attacker GRPO training. Returns output dir path.

    All GPU imports are deferred inside this function so the module is
    importable on CPU-only machines.

    Args:
        cfg: Parsed YAML config dict.
        opponent_checkpoint: Path to a saved defender LoRA checkpoint directory.
            When provided, loads a frozen OpponentModel to apply defensive rules
            before evaluating each attacker rollout. When None, uses a fresh env
            with no defender rules (Gen-0 baseline).
        gen_idx: Generation index, appended to the output directory name.

    Returns:
        Path string of the saved checkpoint directory.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as exc:
        logger.error(
            "Training dependencies not installed. Run: pip install -e '.[training]'\n%s", exc
        )
        sys.exit(1)

    if not torch.cuda.is_available():
        logger.error("No CUDA GPU detected. Training requires a GPU.")
        sys.exit(1)

    model_name: str = cfg["model"]["name"]
    logger.info("Loading attacker model %s ...", model_name)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=cfg["model"]["load_in_4bit"],
        bnb_4bit_compute_dtype=getattr(torch, cfg["model"]["bnb_4bit_compute_dtype"]),
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    opponent = None
    if opponent_checkpoint is not None:
        try:
            from .opponent_loader import OpponentModel
        except ImportError:
            from training.opponent_loader import OpponentModel
        logger.info("Loading opponent defender checkpoint: %s", opponent_checkpoint)
        opponent = OpponentModel.from_checkpoint(
            model_name, opponent_checkpoint, "defender", tokenizer
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    model = prepare_model_for_kbit_training(model)

    lora_cfg = cfg["lora"]
    lora_config = LoraConfig(
        r=lora_cfg["r"],
        lora_alpha=lora_cfg["lora_alpha"],
        lora_dropout=lora_cfg["lora_dropout"],
        target_modules=lora_cfg["target_modules"],
        bias=lora_cfg["bias"],
        task_type=lora_cfg["task_type"],
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    logger.info("Collecting attacker rollout observations ...")
    observations, episode_seeds = _collect_attacker_observations(cfg)
    dataset = build_attacker_rollout_dataset(observations, episode_seeds)
    logger.info("Dataset built: %d prompts", len(dataset))

    tr = cfg["training"]
    output_dir = f"{tr['output_dir']}_attacker_gen{gen_idx}"
    grpo_config = GRPOConfig(
        output_dir=output_dir,
        learning_rate=tr["learning_rate"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        max_grad_norm=tr["max_grad_norm"],
        num_train_epochs=1,
        warmup_steps=tr["warmup_steps"],
        logging_steps=tr["logging_steps"],
        save_steps=tr["save_steps"],
        report_to=tr["report_to"],
        max_completion_length=tr["max_new_tokens"],
        num_generations=tr["rollouts_per_episode"],
        temperature=tr["temperature"],
        top_p=tr["top_p"],
        fp16=True,
    )

    if opponent is not None:
        reward_fn = make_attacker_reward_function_with_opponent(
            opponent=opponent, task_id=cfg["env"]["task_id"]
        )
    else:
        reward_fn = make_attacker_reward_function(task_id=cfg["env"]["task_id"])

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    logger.info("Starting attacker training (gen %d) ...", gen_idx)
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Model saved to %s", output_dir)
    return output_dir


def main() -> None:
    """CLI entrypoint for attacker GRPO training. Requires CUDA GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train SRE Arena attacker with GRPO")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--opponent-checkpoint",
        default=None,
        help="Path to trained defender checkpoint (Phase 7)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_attacker(cfg, opponent_checkpoint=args.opponent_checkpoint, gen_idx=0)


if __name__ == "__main__":
    main()
