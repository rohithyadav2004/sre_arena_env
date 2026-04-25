"""Reward-curve plotting and cross-generation evaluation matrix.

plot_reward_curves:        reads TensorBoard logs, saves matplotlib figure.
plot_cross_gen_matrix_stub: generates a fake heatmap for pipeline testing.
                            Real implementation with checkpoint loading is Phase 7.
"""
from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def plot_reward_curves(checkpoint_dir: str, output_path: str) -> None:
    """Read TensorBoard event files from checkpoint_dir and save a reward plot.

    Returns without creating output_path if checkpoint_dir does not exist,
    if tensorboard is not installed, or if no reward scalar is found.

    Args:
        checkpoint_dir: Directory containing TFEvents files.
        output_path: Destination path for the saved PNG.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ckpt_path = Path(checkpoint_dir)
    if not ckpt_path.exists():
        logger.warning("checkpoint_dir does not exist: %s", checkpoint_dir)
        return

    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except ImportError:
        logger.warning("tensorboard not installed; skipping reward curve plot")
        return

    ea = EventAccumulator(str(ckpt_path))
    ea.Reload()

    reward_tag = next(
        (t for t in ea.Tags().get("scalars", []) if "reward" in t.lower()),
        None,
    )
    if reward_tag is None:
        logger.warning(
            "No reward scalar in %s; available: %s",
            checkpoint_dir,
            ea.Tags().get("scalars", []),
        )
        return

    events = ea.Scalars(reward_tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]

    fig, ax = plt.subplots()
    ax.plot(steps, values)
    ax.set_xlabel("Step")
    ax.set_ylabel("Reward")
    ax.set_title(f"Reward Curve: {ckpt_path.name}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Reward curve saved to %s", output_path)


def plot_cross_gen_matrix_stub(num_gens: int, output_path: str) -> None:
    """Generate a random heatmap for pipeline smoke-testing.

    Each cell represents the average reward for a (defender_gen, attacker_gen)
    matchup. Data is random — real evaluation with loaded checkpoints is Phase 7.

    Args:
        num_gens: Number of training generations (matrix is num_gens × num_gens).
        output_path: Destination path for the saved PNG.
    """
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    matrix = np.random.uniform(0.0, 1.0, (num_gens, num_gens))

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Avg Reward (stub)")

    ax.set_xlabel("Attacker Gen")
    ax.set_ylabel("Defender Gen")
    ax.set_title("Cross-Gen Evaluation Matrix (stub — Phase 7 will use real data)")
    ax.set_xticks(range(num_gens))
    ax.set_yticks(range(num_gens))
    ax.set_xticklabels([f"A{i}" for i in range(num_gens)])
    ax.set_yticklabels([f"D{i}" for i in range(num_gens)])

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out))
    plt.close(fig)
    logger.info("Cross-gen matrix stub saved to %s", output_path)
