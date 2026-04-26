"""Generate notebooks/main_training.ipynb from source.

Run from the sre_arena_env/ directory:
    python notebooks/_generate.py
"""
import nbformat as nbf
import os

# ── Cell contents ─────────────────────────────────────────────────────────────

MD_TITLE = """\
# SRE Arena Env: Self-Play Training Demo

This notebook demonstrates self-play reinforcement learning for SRE Layer 7 defense.
We'll train **3 generations**: a defender, an attacker against that defender, then
a new defender against the trained attacker.

**Total runtime:** ~15 minutes (`fast` mode) or ~21 minutes (`full` mode) on free-tier Colab T4.

- 🌐 [Live demo on HF Spaces](https://blitz1809-sre-arena.hf.space)
- 📦 [GitHub repo](https://github.com/rohithyadav2004/sre_arena_env)
"""

MD_STEP1 = """\
## Step 1: Install dependencies

Installs the package + training extras. Takes ~3 minutes.

> **After installation completes, restart the runtime once** (Runtime → Restart runtime),
> then re-run from this cell. This is required for `bitsandbytes` and `peft` to load correctly.
"""

CODE_INSTALL = """\
!pip install -q "git+https://github.com/rohithyadav2004/sre_arena_env.git@main#egg=openenv-sre-arena-env[training]"
!pip install -q "trl>=0.29.0" "tensorboard" "matplotlib"
"""

MD_STEP2 = """\
## Step 2: Verify GPU

Free-tier Colab provides a T4 (16GB VRAM). The 3B model + 4-bit QLoRA fits comfortably.
"""

CODE_GPU = """\
import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("\\u26a0 No GPU detected. Enable GPU runtime via Runtime \\u2192 Change runtime type.")
"""

MD_STEP3 = """\
## Step 3: Load training config

We load the smoke-test config and compress episode counts so the demo finishes within
Colab's free-tier time budget. The 3-generation structure is preserved.
"""

CODE_CONFIG = """\
import yaml
import urllib.request

# Fetch config directly from GitHub raw (most reliable on Colab)
config_url = "https://raw.githubusercontent.com/rohithyadav2004/sre_arena_env/main/configs/colab_demo.yaml"
cfg = yaml.safe_load(urllib.request.urlopen(config_url).read().decode())

# ── Demo mode ────────────────────────────────────────────────────────────────
# Use "fast" if your Colab session has limited time (~15 min). Curves may look
# flat at this scale — the README shows full-scale results from our A100 run.
DEMO_MODE = "full"   # "fast" (10/5/5, ~15 min) | "full" (12/6/6, ~21 min)

_episodes = {"fast": (10, 5, 5), "full": (12, 6, 6)}[DEMO_MODE]
cfg["per_generation"] = [
    {"role": "defender", "episodes": _episodes[0]},
    {"role": "attacker", "episodes": _episodes[1]},
    {"role": "defender", "episodes": _episodes[2]},
]
cfg["num_generations"] = 3
cfg["training"]["num_episodes"] = max(_episodes)  # cap for any single gen
cfg["training"]["logging_steps"] = 1
cfg["training"]["save_steps"] = 999
cfg["training"]["output_dir"] = "./checkpoints/colab_demo"

print(f"DEMO_MODE={DEMO_MODE!r}  episodes={_episodes}")
print(yaml.dump(cfg, default_flow_style=False))
"""

MD_STEP4 = """\
## Step 4: Train 3 generations

This runs the full alternating self-play loop:

| Gen | Role | Opponent |
|-----|------|----------|
| 0   | Defender | Scripted attacker (cycles through 8 templates) |
| 1   | Attacker | Trained Gen-0 defender (PEFT checkpoint) |
| 2   | Defender | Trained Gen-1 attacker (PEFT checkpoint) |

Each generation uses GRPO (Group Relative Policy Optimization) with QLoRA fine-tuning.
The reward shaping uses an **anti-exploit multiplicative formula** that prevents
"block everything" degenerate strategies.

**Expected runtime: ~15 min (`fast`) or ~21 min (`full`) on T4.**
"""

CODE_TRAIN = """\
from sre_arena_env.training.alternating_loop import run_alternating_loop

run_alternating_loop(cfg, dry_run=False)
print("\\n\\u2713 All 3 generations complete.")
"""

MD_STEP5 = """\
## Step 5: Visualize per-generation learning curves

We plot the reward curve from each generation. Watch for:
- **Gen 0 defender:** reward should climb as it learns to write effective nginx rules
- **Gen 1 attacker:** reward reflects evasion success against the trained defender
- **Gen 2 defender:** reward shows whether it can re-defend against the smarter attacker
"""

CODE_PLOT = """\
import json
import matplotlib.pyplot as plt
from pathlib import Path

base_output_dir = cfg["training"]["output_dir"]
generations = [
    (0, "defender", "Gen 0: Defender (vs scripted attacker)"),
    (1, "attacker", "Gen 1: Attacker (vs Gen-0 defender)"),
    (2, "defender", "Gen 2: Defender (vs Gen-1 attacker)"),
]

fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=110)

for i, (gen_idx, role, title) in enumerate(generations):
    ax = axes[i]
    ckpt_dir = Path(f"{base_output_dir}_{role}_gen{gen_idx}")
    state_path = ckpt_dir / "trainer_state.json"

    if not state_path.exists():
        candidates = list(ckpt_dir.glob("**/trainer_state.json"))
        if candidates:
            state_path = candidates[-1]

    if state_path.exists():
        state = json.loads(state_path.read_text())
        log_history = state.get("log_history", [])
        steps = []
        rewards = []
        for entry in log_history:
            if "reward" in entry:
                steps.append(entry.get("step", len(steps)))
                rewards.append(entry["reward"])

        if rewards:
            ax.plot(steps, rewards, marker="o", markersize=5, linewidth=1.5,
                    color="#2E86AB" if role == "defender" else "#E63946")
            ax.axhline(0, color="gray", ls="--", alpha=0.4)
            ax.set_xlabel("Training step")
            ax.set_ylabel("Reward")
            ax.grid(alpha=0.3)
            ax.set_title(title, fontsize=11)
        else:
            ax.text(0.5, 0.5, "No reward data logged", ha="center", va="center",
                    transform=ax.transAxes, fontsize=12)
            ax.set_title(title, fontsize=11)
    else:
        ax.text(0.5, 0.5, f"Checkpoint not found:\\n{ckpt_dir}", ha="center", va="center",
                transform=ax.transAxes, fontsize=10)
        ax.set_title(title, fontsize=11)

plt.suptitle("SRE Arena Env: 3-Generation Self-Play Reward Curves",
             fontsize=14, y=1.02)
plt.tight_layout()
plt.savefig("colab_self_play_curves.png", dpi=120, bbox_inches="tight")
plt.show()

print("\\n\\u2713 Reward curves saved to colab_self_play_curves.png")
"""

MD_STEP6 = """\
## Step 6: Inspect a trained agent

Quick inference example: load the Gen-2 defender (the most-trained model) and see
what action it produces given a sample observation.
"""

CODE_INFER = """\
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch

# Load base model in 4-bit
bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

base = AutoModelForCausalLM.from_pretrained(
    cfg["model"]["name"],
    quantization_config=bnb,
    device_map="auto",
)
tok = AutoTokenizer.from_pretrained(cfg["model"]["name"])

# Wrap with Gen-2 defender LoRA adapter
gen2_path = f"{cfg['training']['output_dir']}_defender_gen2"
if not Path(gen2_path).exists():
    raise FileNotFoundError(
        f"Gen-2 checkpoint not found at {gen2_path!r}. "
        "Did Step 4 (training) complete successfully?"
    )
model = PeftModel.from_pretrained(base, gen2_path)

# Sample defender prompt
prompt = (
    "You are an SRE defender. Recent log shows:\\n"
    '10.0.5.42 - - [12:00:01] "POST /login HTTP/1.1" 401 200\\n'
    '10.0.5.42 - - [12:00:01] "POST /login HTTP/1.1" 401 200\\n'
    '10.0.5.42 - - [12:00:01] "POST /login HTTP/1.1" 401 200\\n'
    "(...repeated 200 times from same IP)\\n\\n"
    "Output a JSON action.\\n"
)

inputs = tok(prompt, return_tensors="pt").to(model.device)
out = model.generate(**inputs, max_new_tokens=80, do_sample=False)
response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
print("Trained defender's action:")
print(response)
"""

MD_NEXTSTEPS = """\
## Next steps

This Colab demo runs at compressed scale. For real training:

1. **Scale up episodes:** Use the production config (`configs/l4_training.yaml`) with 150+ episodes per generation.
2. **Bigger model:** Qwen2.5-7B trains beautifully on A100. We trained Gen 0 to perfect reward (1.0) in ~74 minutes — see the README for the full reward curve.
3. **HF Jobs A100:**
```bash
hf jobs uv run --flavor a100-large --timeout 6h \\
  --with "git+https://github.com/rohithyadav2004/sre_arena_env.git@main#egg=openenv-sre-arena-env[training]" \\
  --with "trl>=0.29.0" --secrets HF_TOKEN \\
  hf_jobs/train_on_hf_jobs.py
```

To upload your checkpoint to HF Hub:

```python
# from huggingface_hub import HfApi, login
# login()  # use your HF token
# api = HfApi()
# api.create_repo("your-username/sre-defender-gen2", private=True, exist_ok=True)
# api.upload_folder(folder_path=gen2_path, repo_id="your-username/sre-defender-gen2")
```

Read the [project README](https://github.com/rohithyadav2004/sre_arena_env)
for architecture details and the anti-exploit reward formula.
"""

MD_FOOTER = """\
---

Built for **Meta x Scaler OpenEnv Hackathon 2026**.

- 🌐 [Live demo on HF Spaces](https://blitz1809-sre-arena.hf.space)
- 📦 [GitHub repo](https://github.com/rohithyadav2004/sre_arena_env)
"""

# ── Assemble notebook ─────────────────────────────────────────────────────────

def build_notebook() -> nbf.NotebookNode:
    nb = nbf.v4.new_notebook()
    nb["cells"] = [
        nbf.v4.new_markdown_cell(MD_TITLE),      # 1
        nbf.v4.new_markdown_cell(MD_STEP1),      # 2
        nbf.v4.new_code_cell(CODE_INSTALL),      # 3
        nbf.v4.new_markdown_cell(MD_STEP2),      # 4
        nbf.v4.new_code_cell(CODE_GPU),          # 5
        nbf.v4.new_markdown_cell(MD_STEP3),      # 6
        nbf.v4.new_code_cell(CODE_CONFIG),       # 7
        nbf.v4.new_markdown_cell(MD_STEP4),      # 8
        nbf.v4.new_code_cell(CODE_TRAIN),        # 9
        nbf.v4.new_markdown_cell(MD_STEP5),      # 10
        nbf.v4.new_code_cell(CODE_PLOT),         # 11
        nbf.v4.new_markdown_cell(MD_STEP6),      # 12
        nbf.v4.new_code_cell(CODE_INFER),        # 13
        nbf.v4.new_markdown_cell(MD_NEXTSTEPS),  # 14
        nbf.v4.new_markdown_cell(MD_FOOTER),     # 15
    ]
    nb.metadata = {
        "colab": {"provenance": [], "name": "SRE Arena Env: Self-Play Training Demo"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    }
    return nb


if __name__ == "__main__":
    os.makedirs("notebooks", exist_ok=True)
    nb = build_notebook()
    out_path = "notebooks/main_training.ipynb"
    with open(out_path, "w") as f:
        nbf.write(nb, f)
    total = len(nb["cells"])
    code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    md = total - code
    print(f"Written: {out_path}  ({total} cells: {code} code, {md} markdown)")
