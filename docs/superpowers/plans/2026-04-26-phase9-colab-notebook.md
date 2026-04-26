# Phase 9: Colab Self-Play Notebook — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `notebooks/_generate.py` and `notebooks/main_training.ipynb` — a judge-runnable 15-cell Colab notebook demonstrating 3-generation alternating self-play training on free-tier T4.

**Architecture:** A standalone Python generator script (`_generate.py`) uses `nbformat.v4` to construct all 15 cells programmatically and writes `main_training.ipynb`. The notebook fetches `colab_demo.yaml` from GitHub raw, applies a `DEMO_MODE` toggle (fast: 10/5/5, full: 12/6/6 episodes), then calls `run_alternating_loop(cfg)` for the training, and plots 3-panel reward curves from `trainer_state.json` files.

**Tech Stack:** `nbformat>=5`, `yaml`, `urllib.request` (stdlib), `matplotlib`, `peft`, `transformers`, `trl>=0.29.0`, `bitsandbytes`

---

## File Map

| File | Action | Purpose |
|------|--------|---------|
| `notebooks/_generate.py` | **Create** | Generates the notebook via nbformat — single source of truth |
| `notebooks/main_training.ipynb` | **Create (generated)** | The actual Colab notebook committed to the repo |

No existing files are modified. No new tests are written (the verification steps below serve as the test suite).

---

### Task 1: Create `notebooks/_generate.py`

**Files:**
- Create: `notebooks/_generate.py`

- [ ] **Step 1: Create the `notebooks/` directory and write `_generate.py`**

The entire file content is below. Run this step by writing the file exactly as shown — every cell content is a module-level variable to keep triple-quote escaping unambiguous.

```python
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
cfg["training"]["num_episodes"] = _episodes[0]   # max single-gen cap
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
out = model.generate(**inputs, max_new_tokens=80, do_sample=False, temperature=0.1)
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
```

- [ ] **Step 2: Verify the file was written cleanly**

```bash
python -c "import ast, sys; ast.parse(open('notebooks/_generate.py').read()); print('Syntax OK')"
```

Expected: `Syntax OK`

---

### Task 2: Run `_generate.py` to produce the notebook

**Files:**
- Create (generated): `notebooks/main_training.ipynb`

- [ ] **Step 1: Install nbformat if not already present, then run the generator**

```bash
cd /home/rohith/Rohith/Scalar/sre_arena_env
source ../.venv/bin/activate
pip install -q nbformat
python notebooks/_generate.py
```

Expected output:
```
Written: notebooks/main_training.ipynb  (15 cells: 6 code, 9 markdown)
```

- [ ] **Step 2: Verify JSON validity**

```bash
python -c "import json; json.load(open('notebooks/main_training.ipynb')); print('Valid JSON')"
```

Expected: `Valid JSON`

- [ ] **Step 3: Verify cell count and structure**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/main_training.ipynb', as_version=4)
total = len(nb.cells)
code  = sum(1 for c in nb.cells if c.cell_type == 'code')
md    = sum(1 for c in nb.cells if c.cell_type == 'markdown')
print(f'Total: {total}  Code: {code}  Markdown: {md}')
assert total == 15, f'Expected 15 cells, got {total}'
assert code  == 6,  f'Expected 6 code cells, got {code}'
assert md    == 9,  f'Expected 9 markdown cells, got {md}'
print('Cell count OK')
"
```

Expected:
```
Total: 15  Code: 6  Markdown: 9
Cell count OK
```

- [ ] **Step 4: Spot-check DEMO_MODE toggle is present in cell 7**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/main_training.ipynb', as_version=4)
cell7 = nb.cells[6]   # 0-indexed
assert cell7.cell_type == 'code', 'Cell 7 should be code'
assert 'DEMO_MODE' in cell7.source, 'DEMO_MODE toggle missing from cell 7'
assert '\"fast\"' in cell7.source or \"'fast'\" in cell7.source, 'fast mode missing'
assert '\"full\"' in cell7.source or \"'full'\" in cell7.source, 'full mode missing'
print('DEMO_MODE toggle OK')
"
```

Expected: `DEMO_MODE toggle OK`

- [ ] **Step 5: Spot-check Colab metadata is set**

```bash
python -c "
import nbformat
nb = nbformat.read('notebooks/main_training.ipynb', as_version=4)
assert nb.metadata.get('accelerator') == 'GPU', 'accelerator not set'
assert 'colab' in nb.metadata, 'colab metadata missing'
print('Metadata OK')
"
```

Expected: `Metadata OK`

- [ ] **Step 6: Re-run generator to confirm idempotency**

```bash
python notebooks/_generate.py
python -c "import json; json.load(open('notebooks/main_training.ipynb')); print('Idempotent OK')"
```

Expected: `Idempotent OK`

---

### Task 3: Commit and push

**Files:**
- Stage: `notebooks/_generate.py`, `notebooks/main_training.ipynb`, `docs/superpowers/specs/2026-04-26-colab-notebook-design.md`, `docs/superpowers/plans/2026-04-26-phase9-colab-notebook.md`

- [ ] **Step 1: Stage files**

```bash
cd /home/rohith/Rohith/Scalar/sre_arena_env
git add notebooks/_generate.py notebooks/main_training.ipynb \
        docs/superpowers/specs/2026-04-26-colab-notebook-design.md \
        docs/superpowers/plans/2026-04-26-phase9-colab-notebook.md
git status
```

Expected: 4 new files staged (all green `new file:`).

- [ ] **Step 2: Commit**

```bash
git commit -m "$(cat <<'EOF'
Phase 9: Colab self-play demo notebook (3-gen, T4-compatible)

- notebooks/_generate.py: nbformat generator for the 15-cell notebook
- notebooks/main_training.ipynb: judge-runnable Colab notebook
  - 3-gen alternating self-play (defender→attacker→defender)
  - DEMO_MODE toggle: fast (10/5/5, ~15 min) / full (12/6/6, ~21 min)
  - 3-panel reward curve plot + Gen-2 inference example

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Expected: `[main <hash>] Phase 9: Colab self-play demo notebook (3-gen, T4-compatible)`

- [ ] **Step 3: Push to origin**

```bash
git push origin main
```

Expected: `Branch 'main' set up to track remote branch 'main' from 'origin'.` or similar push confirmation.

- [ ] **Step 4: Final report**

```bash
python -c "
import os, nbformat
nb = nbformat.read('notebooks/main_training.ipynb', as_version=4)
size = os.path.getsize('notebooks/main_training.ipynb')
total = len(nb.cells)
code  = sum(1 for c in nb.cells if c.cell_type == 'code')
md    = total - code
# time estimate: full mode 12/6/6, defender=30s, attacker=120s
fast_t = 10*30 + 5*120 + 5*30   # fast mode
full_t = 12*30 + 6*120 + 6*30   # full mode
print(f'File size:    {size:,} bytes')
print(f'Total cells:  {total}  ({code} code, {md} markdown)')
print(f'fast mode:    ~{fast_t//60} min ({fast_t}s)')
print(f'full mode:    ~{full_t//60} min ({full_t}s)')
"
git log --oneline -1
```

---

## Self-Review Checklist

**Spec coverage:**
- [x] 15 cells in order (markdown/code alternation matches spec exactly)
- [x] Cell 1: title + HF Space link + GitHub link
- [x] Cell 3: installs `[training]` extra + `trl>=0.29.0` + `tensorboard` + `matplotlib`
- [x] Cell 5: GPU check with VRAM print
- [x] Cell 7: `colab_demo.yaml` from GitHub raw + DEMO_MODE toggle (fast 10/5/5, full 12/6/6) + explanation comment
- [x] Cell 9: `run_alternating_loop(cfg, dry_run=False)`
- [x] Cell 11: 3-panel matplotlib plot with fallback for missing checkpoints
- [x] Cell 13: PeftModel inference on Gen-2 defender checkpoint
- [x] Cell 14: next-steps with HF Hub upload as commented code
- [x] Colab GPU metadata set (`accelerator: GPU`)
- [x] No HF login required
- [x] Generator script is idempotent
- [x] Commit + push included

**No placeholders:** All cell content is complete literal code — no TBDs.

**Type consistency:** `cfg["training"]["output_dir"]` used consistently across cells 7, 11, 13. `run_alternating_loop` import path `sre_arena_env.training.alternating_loop` matches actual module layout.
