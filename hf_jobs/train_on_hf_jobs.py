"""HF Jobs entrypoint for SRE Arena Env training.

Run with:
    hf jobs uv run \
        --flavor a100-large \
        --timeout 8h \
        --with "git+https://github.com/rohithyadav2004/sre_arena_env.git@main#egg=openenv-sre-arena-env[training]" \
        --with "tensorboard" \
        --secrets HF_TOKEN \
        hf_jobs/train_on_hf_jobs.py
"""
import os
import subprocess
import sys
from pathlib import Path

print("=" * 60)
print("HF Jobs: SRE Arena Env training")
print("=" * 60)

# Verify GPU
subprocess.run(["nvidia-smi"], check=True)

# Verify imports
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Now run alternating loop
from sre_arena_env.training.alternating_loop import run_alternating_loop
import yaml

# Load config (paths inside container — config bundled in package)
import importlib.resources as pkg_resources
import sre_arena_env

# Try to find configs/l4_training.yaml from the installed package
# Since configs/ may not be in the package, we hardcode the config dict here
cfg = {
    "num_generations": 3,
    "per_generation": [
        {"role": "defender", "episodes": 200},
        {"role": "attacker", "episodes": 200},
        {"role": "defender", "episodes": 200},
    ],
    "model": {
        "name": "Qwen/Qwen2.5-7B-Instruct",  # Same as L4 for consistency. Use 7B if A100.
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
    },
    "lora": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "training": {
        "num_episodes": 150,
        "rollouts_per_episode": 8,
        "learning_rate": 5.0e-5,
        "per_device_train_batch_size": 1,
        "gradient_accumulation_steps": 8,
        "warmup_steps": 20,
        "max_grad_norm": 0.5,
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9,
        "save_steps": 50,
        "output_dir": "./checkpoints/hf_jobs",
        "logging_steps": 5,
        "report_to": "tensorboard",
    },
    "env": {
        "task_id": "task1",
        "base_seed": 42,
    },
}

print("Starting alternating loop with config:")
print(yaml.dump(cfg, default_flow_style=False))

run_alternating_loop(cfg, dry_run=False)

# After training: push checkpoints to HF Hub
from huggingface_hub import HfApi
api = HfApi()

for gen_idx in range(3):
    role = "defender" if gen_idx % 2 == 0 else "attacker"
    ckpt_dir = Path(f"./checkpoints/hf_jobs_{role}_gen{gen_idx}")
    if not ckpt_dir.exists():
        print(f"Skipping gen {gen_idx}: checkpoint dir not found")
        continue
    
    repo_id = f"blitz1809/sre-arena-{role}-gen{gen_idx}"
    print(f"Uploading {ckpt_dir} to {repo_id}...")
    api.create_repo(repo_id, private=True, exist_ok=True)
    api.upload_folder(folder_path=str(ckpt_dir), repo_id=repo_id, repo_type="model")
    print(f"Uploaded gen {gen_idx} checkpoint")

print("=" * 60)
print("HF Jobs run complete")
print("=" * 60)
