"""Defender GRPO training entrypoint.

Usage:
    python -m training.train_defender --config configs/colab_demo.yaml

GPU is required to run training. Unit tests cover load_config and
_collect_observations without GPU by importing this module directly.
All GPU-specific imports (torch, transformers, peft, trl) are deferred
inside train_defender() so the module is importable on CPU-only machines.
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
    from ..models import DefenderAction, DefenderObservation
    from ..server.sre_arena_env_environment import SreArenaEnvironment
    from .dataset_builder import build_rollout_dataset
    from .reward_function import make_reward_function
    from .scripted_attacker import ScriptedAttacker
except ImportError:
    from models import DefenderAction, DefenderObservation
    from server.sre_arena_env_environment import SreArenaEnvironment
    from training.dataset_builder import build_rollout_dataset
    from training.reward_function import make_reward_function
    from training.scripted_attacker import ScriptedAttacker


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


def _collect_observations(
    cfg: dict,
    opponent=None,
) -> tuple[list[DefenderObservation], list[dict], list[int]]:
    """Collect one step-1 observation per training episode.

    For each episode:
    1. Reset env with a deterministic seed.
    2. Inject attacker template into env._last_attacker_action — either from
       a scripted baseline (opponent=None) or from a trained OpponentModel.
    3. Take a read_log action to generate traffic and populate metrics.
    4. Collect the resulting DefenderObservation as the training prompt.

    Note: obs.log_tail is empty at step 1 because sim_nginx logs traffic
    after read_log executes within the same step. The observation is still
    useful via last_step_metrics (requests_total, 200_count, etc.).

    Args:
        cfg: Parsed YAML config dict.
        opponent: Optional OpponentModel for the attacker role. When None,
            ScriptedAttacker is used (Gen-0 baseline).

    Returns:
        Tuple of (observations, attacker_dicts, episode_seeds), all same length.
    """
    num_episodes: int = cfg["training"]["num_episodes"]
    base_seed: int = cfg["env"]["base_seed"]
    task_id: str = cfg["env"]["task_id"]

    scripted = ScriptedAttacker(seed=base_seed) if opponent is None else None
    env = SreArenaEnvironment()

    observations: list[DefenderObservation] = []
    attacker_dicts: list[dict] = []
    episode_seeds: list[int] = []

    for ep in range(num_episodes):
        seed = base_seed + ep
        env.reset(role="defender", seed=seed, task_id=task_id)

        if opponent is not None:
            attacker_dict = opponent.generate_action()
        else:
            attacker_action = scripted.act()
            attacker_dict = attacker_action.model_dump(exclude={"delay_ms", "metadata"})
        env._last_attacker_action = attacker_dict

        # Warm-up step: triggers traffic generation (log not yet populated)
        env.step(DefenderAction(action_type="read_log", log_tail_lines=20))
        # Capture step: log_tail now reflects the warm-up step's traffic
        obs = env.step(DefenderAction(action_type="read_log", log_tail_lines=20))

        observations.append(obs)
        attacker_dicts.append(attacker_dict)
        episode_seeds.append(seed)

        if (ep + 1) % 10 == 0:
            logger.info("Collected %d / %d episodes", ep + 1, num_episodes)

    return observations, attacker_dicts, episode_seeds


def train_defender(
    cfg: dict,
    opponent_checkpoint: str | None = None,
    gen_idx: int = 0,
) -> str:
    """Run one generation of defender GRPO training. Returns output dir path.

    All GPU imports are deferred inside this function so the module is
    importable on CPU-only machines.

    Args:
        cfg: Parsed YAML config dict.
        opponent_checkpoint: Path to a saved attacker LoRA checkpoint directory.
            When provided, loads a frozen OpponentModel to generate attacker actions
            during rollout collection. When None, uses ScriptedAttacker (Gen-0 baseline).
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
    logger.info("Loading model %s ...", model_name)

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
        logger.info("Loading opponent attacker checkpoint: %s", opponent_checkpoint)
        opponent = OpponentModel.from_checkpoint(
            model_name, opponent_checkpoint, "attacker", tokenizer
        )

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        dtype=torch.float16,
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

    logger.info("Collecting rollout observations ...")
    observations, attacker_dicts, episode_seeds = _collect_observations(cfg, opponent=opponent)
    dataset = build_rollout_dataset(observations, attacker_dicts, episode_seeds)
    logger.info("Dataset built: %d prompts", len(dataset))

    tr = cfg["training"]
    output_dir = f"{tr['output_dir']}_defender_gen{gen_idx}"
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

    reward_fn = make_reward_function(task_id=cfg["env"]["task_id"])

    trainer = GRPOTrainer(
        model=model,
        args=grpo_config,
        train_dataset=dataset,
        reward_funcs=[reward_fn],
        processing_class=tokenizer,
    )

    logger.info("Starting defender training (gen %d) ...", gen_idx)
    trainer.train()
    trainer.save_model(output_dir)
    logger.info("Model saved to %s", output_dir)
    return output_dir


def main() -> None:
    """CLI entrypoint for defender GRPO training. Requires CUDA GPU."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Train SRE Arena defender with GRPO")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument(
        "--opponent-checkpoint",
        default=None,
        help="Path to trained attacker checkpoint (Phase 7)",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    train_defender(cfg, opponent_checkpoint=args.opponent_checkpoint, gen_idx=0)


if __name__ == "__main__":
    main()
