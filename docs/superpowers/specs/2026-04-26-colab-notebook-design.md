# Phase 9: Colab Self-Play Notebook — Design Spec

**Date:** 2026-04-26  
**Status:** Approved  
**Deadline:** 18:00 IST today (hackathon submission)

---

## Goal

Produce `notebooks/main_training.ipynb` — a judge-runnable Colab notebook demonstrating
the full 3-generation alternating self-play loop on free-tier T4 (16 GB VRAM).
Generated programmatically via `notebooks/_generate.py` using `nbformat`.

---

## Constraints

| Constraint | Value |
|-----------|-------|
| Target GPU | Colab free-tier T4 (16 GB VRAM) |
| Wall-clock budget | < 20 min (`full` mode), < 15 min (`fast` mode) |
| HF login required | No — base model only, no checkpoint upload |
| wandb | No |
| Python | 3.x (Colab default) |

---

## Notebook Structure — 15 Cells

| # | Type | Purpose |
|---|------|---------|
| 1 | Markdown | Title + links to HF Space and GitHub |
| 2 | Markdown | Step 1 header: Install |
| 3 | Code | `pip install` package + trl + tensorboard + matplotlib |
| 4 | Markdown | Step 2 header: Verify GPU |
| 5 | Code | GPU check (`torch.cuda`, device name, VRAM) |
| 6 | Markdown | Step 3 header: Load + customise config |
| 7 | Code | Fetch `colab_demo.yaml` + DEMO_MODE toggle + overrides |
| 8 | Markdown | Step 4 header: 3-gen training table + runtime note |
| 9 | Code | `run_alternating_loop(cfg, dry_run=False)` |
| 10 | Markdown | Step 5 header: Visualise learning curves |
| 11 | Code | 3-panel matplotlib reward-curve plot |
| 12 | Markdown | Step 6 header: Inspect trained Gen-2 defender |
| 13 | Code | Load Gen-2 LoRA + run sample inference |
| 14 | Markdown | Next steps (scale-up, HF Hub upload as commented code) |
| 15 | Markdown | Footer with links |

---

## Key Design Decisions

### DEMO_MODE toggle (Cell 7)
```python
DEMO_MODE = "full"   # "fast" (10/5/5, ~15 min) | "full" (12/6/6, ~21 min)
```
- `full`: 12 defender + 6 attacker + 6 defender = 24 episodes, ~21 min wall-clock
- `fast`: 10 defender + 5 attacker + 5 defender = 20 episodes, ~15 min wall-clock
- A comment above the toggle explains: *"Use 'fast' if your Colab session has limited time.
  Curves may look flat at this scale — the README shows full-scale results from our A100 run."*

Time estimate formula: `defender_steps × 30s + attacker_steps × 120s`
(attacker steps are slower due to opponent forward pass in Gen 1+)

### Config fetch strategy (Cell 7)
`urllib.request.urlopen` fetches `colab_demo.yaml` directly from GitHub raw.
Avoids path issues with Colab's working directory.

### `output_dir` override
Set to `./checkpoints/colab_demo` (explicit, relative to Colab `/content/`).
Avoids inheriting `smoketest` path from the committed YAML.

### Reward curve fallback (Cell 11)
If `trainer_state.json` not found in `{output_dir}_{role}_gen{N}`, the cell
searches `**/trainer_state.json` subdirectories. If still missing, shows a
"Checkpoint not found" placeholder text — never crashes.

### Inference cell (Cell 13)
Loads Gen-2 defender LoRA adapter via `PeftModel.from_pretrained` directly
(not via `OpponentModel` — judges see the raw HuggingFace API).
Uses `BitsAndBytesConfig` 4-bit for VRAM efficiency.

---

## Install Cell Details (Cell 3)

```python
!pip install -q "git+https://github.com/rohithyadav2004/sre_arena_env.git@main#egg=openenv-sre-arena-env[training]"
!pip install -q "trl>=0.29.0" "tensorboard" "matplotlib"
```

The package's `[training]` extra provides: `trl>=0.21.0,<0.30`, `peft`, `transformers`,
`bitsandbytes`, `matplotlib`. The explicit `trl>=0.29.0` pin ensures we get 0.29.x
(compatible with `<0.30` in pyproject.toml, guarantees `GRPOConfig.max_completion_length`
and `num_generations` parameters).

---

## Notebook Metadata

```json
{
  "colab": {"provenance": [], "name": "SRE Arena Env: Self-Play Training Demo"},
  "kernelspec": {"name": "python3", "display_name": "Python 3"},
  "language_info": {"name": "python"},
  "accelerator": "GPU"
}
```

---

## Generation Script

`notebooks/_generate.py` — standalone Python script.  
Running `python notebooks/_generate.py` regenerates `notebooks/main_training.ipynb`.  
Uses `nbformat.v4` API. No external dependencies beyond `nbformat`.

---

## Verification Steps

1. `python -c "import json; json.load(open('notebooks/main_training.ipynb')); print('Valid JSON')"`
2. Cell count check via `nbformat.read`  
3. `python notebooks/_generate.py` — confirm it re-runs cleanly
4. Commit both files + push to `origin main`

---

## Out of Scope

- wandb integration
- HF Hub auth / login (upload shown only as commented code)
- Running the notebook locally (no T4 available)
- Skipping the alternating loop

---

## Links

- HF Space: `https://blitz1809-sre-arena.hf.space`
- GitHub: `https://github.com/rohithyadav2004/sre_arena_env`
