---
title: SRE Arena
emoji: 🛡️
colorFrom: red
colorTo: blue
sdk: docker
pinned: false
app_port: 8000
tags:
  - openenv
---

# SRE Arena Env

**Self-play reinforcement learning for Layer 7 web defense.**

An OpenEnv environment where blue-team (defender) and red-team (attacker) LLM agents alternate training, learning to write surgical nginx rules and Express middleware against a simulated web service.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohithyadav2004/sre_arena_env/blob/main/notebooks/main_training.ipynb)
[![HF Space](https://img.shields.io/badge/🤗_Space-Live_Demo-yellow)](https://blitz1809-sre-arena.hf.space)
[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)

---

![3-generation self-play reward curves](multi_gen_overlay.png)

> 3-generation self-play on Qwen2.5-7B + QLoRA + GRPO (single A100). Gen 0 defender learns from -0.03 to 1.0 reward in 74 minutes. Gen 1 attacker collapses to zero gradient (documented in [BLOG.md](BLOG.md)). Gen 2 defender re-converges in 3 gradient steps.

---

## TL;DR

- **OpenEnv environment** simulating an nginx + Express.js stack with 8 attack templates and a multiplicative anti-exploit reward formula
- **Trained Qwen2.5-7B** with QLoRA + GRPO across 3 self-play generations on a single A100 — Gen 0 reaches perfect reward (1.0) in 74 minutes
- **Documented self-play failure mode** (Gen 1 zero-gradient collapse) using GRPO auxiliary metrics — gradient norm, frac_reward_zero_std, policy entropy
- **216 tests passing**, live HF Space demo, Colab notebook for reproducibility
- Full story + plot interpretations in **[BLOG.md](BLOG.md)**

---

## Quick Links

- 🌐 **Live demo:** [https://blitz1809-sre-arena.hf.space](https://blitz1809-sre-arena.hf.space) — watch traffic and defenses in real time
- 📓 **Colab notebook:** [`notebooks/main_training.ipynb`](notebooks/main_training.ipynb) — full 3-gen self-play in 20 min on free T4
- 📖 **Full write-up:** [`BLOG.md`](BLOG.md) — the engineering story, GRPO explanation, all 5 plots
- 🤗 **Trained models:** [`blitz1809/sre-arena-defender-gen0`](https://huggingface.co/blitz1809/sre-arena-defender-gen0) · [`-attacker-gen1`](https://huggingface.co/blitz1809/sre-arena-attacker-gen1) · [`-defender-gen2`](https://huggingface.co/blitz1809/sre-arena-defender-gen2)
- 🎯 **Inference baseline:** [`inference.py`](inference.py) — hackathon-mandatory script

---

## The Problem

SRE engineers face Layer 7 attacks daily — credential stuffing, distributed floods, payload injection. The defenses are hand-written nginx rules and middleware. Block too aggressively → kill legitimate traffic. Block too narrowly → attacks succeed. The tradeoff is fundamentally a sequential decision problem with a verifiable reward signal.

We built **SRE Arena Env**: an RL environment where an LLM agent learns to write surgical defenses by playing against an adversary that's also learning. The environment, simulator, training pipeline, and reward formula are described below.

---

## The Anti-Exploit Reward (Key Contribution)

```
score = (malicious_blocked / total_malicious) × (legit_allowed / total_legit) - middleware_penalty
```

Why multiplicative, not additive?

| Strategy | mal_blocked | legit_allowed | Sum | Multiplicative |
|----------|-------------|---------------|-----|----------------|
| Block everything | 100% | 0% | 1.0 | **0.0** |
| Allow everything | 0% | 100% | 1.0 | **0.0** |
| Surgical defense | 95% | 98% | 1.93 | **0.93** |

The multiplicative form forces the agent to balance both objectives simultaneously — same insight as F1 score vs precision+recall. Early experiments with sum rewards led to "block everything" degenerate policies.

Full discussion in [BLOG.md § 4](BLOG.md).

---

## Architecture

```
Blue agent (LLM) ──┐                            ┌── Red agent (LLM)
                   ↓                            ↓
              SreArenaEnvClient (sync wrapper over async)
                              ↓
                  FastAPI server (HF Space)
                              ↓
              ┌──────── SreArenaEnvironment ────────┐
              │                                      │
              │  sim_nginx ──► routing rules         │
              │  sim_express ──► middleware          │
              │  Rubric ──► (mal_blocked / total)    │
              │                × (legit / total)     │
              └──────────────────────────────────────┘
                              ↓
                       Reward [0, 1]
```

- **Pure-Python simulator** (1000× faster than real Docker)
- **Eight attack templates** for the red team (`single_ip_flood`, `ip_spray`, `credential_stuffing`, `payload_injection`, `header_spoof`, `slow_drip`, `path_traversal`, `mixed_legit_cover`)
- **Three defender actions** (`read_log`, `append_nginx_rule`, `write_express_middleware`)
- **Live SSE dashboard** on HF Space for visual debugging

---

## Quickstart

### Run the environment locally

```bash
git clone https://github.com/rohithyadav2004/sre_arena_env.git
cd sre_arena_env
pip install -e .
python -m sre_arena_env.server.app  # FastAPI on :8000
```

### Use the env from Python

```python
from sre_arena_env.client import SreArenaEnvClient
from sre_arena_env.models import DefenderAction

env = SreArenaEnvClient("http://localhost:8000")
with env.sync() as e:
    result = e.reset(role="defender", seed=42)
    obs = result.observation
    action = DefenderAction(action_type="append_nginx_rule", rule_text="deny 10.0.1.1;")
    result = e.step(action)
    print(result.observation.reward)
```

### Run the hackathon baseline (inference.py)

```bash
export API_BASE_URL="https://api.groq.com/openai/v1"
export MODEL_NAME="llama-3.3-70b-versatile"
export HF_TOKEN="<your_groq_or_hf_key>"
python inference.py
```

---

## Train Your Own

### Free-tier Colab T4 (~15 min, 3-gen self-play demo)

Open [`notebooks/main_training.ipynb`](notebooks/main_training.ipynb) — the notebook fetches the package from GitHub, applies a compressed config, and runs the full alternating loop. Visualizes per-generation reward curves at the end.

### Production scale on HF Jobs A100 (~5 hours per gen, 3 gens total)

```bash
hf jobs uv run --flavor a100-large --timeout 18h \
  --with "git+https://github.com/rohithyadav2004/sre_arena_env.git@main#egg=openenv-sre-arena-env[training]" \
  --with "trl>=0.29.0" \
  --secrets HF_TOKEN \
  hf_jobs/train_on_hf_jobs.py
```

Trained checkpoints upload automatically to HF Hub at `<your-namespace>/sre-arena-{role}-gen{N}`.

### Inference on a trained checkpoint

```bash
hf jobs uv run --flavor t4-small --timeout 30m \
  --with "transformers" --with "peft" --with "bitsandbytes" \
  --with "torch" --with "accelerate" --with "huggingface_hub" \
  --secrets HF_TOKEN \
  hf_jobs/run_inference_demo.py
```

This loads our Gen 2 defender from HF Hub and runs it on 3 hand-crafted attack scenarios. Outputs at [`hf_jobs/inference_demo.json`](hf_jobs/inference_demo.json).

---

## Results

| Generation | Role | Wall clock | First reward | Final reward | Gradient updates |
|-----------|------|-----------|--------------|--------------|------------------|
| Gen 0 | Defender | 74 min | -0.03 | 1.00 | 3 burst phases |
| Gen 1 | Attacker | 5h 5min | 1.00 | 1.00 | **0 (gradient collapsed)** |
| Gen 2 | Defender | 5h 5min | 0.21 | 1.00 | 3 steps (rapid convergence) |

Gen 0 demonstrates the env produces real learning curves. Gen 1 documents a self-play failure mode (blind-context opponent → zero reward variance → zero gradient) — diagnosed cleanly using GRPO's auxiliary metrics. Gen 2 validates that the alternating-loop infrastructure correctly hands off opponent checkpoints, even when an intermediate generation fails to learn.

The Gen 2 defender was tested on 3 hand-crafted scenarios — it correctly chose Express middleware for payload injection, applied rate limiting for distributed floods, but defaulted to rate limiting when surgical IP-blocking would have been more precise. Full discussion in [BLOG.md](BLOG.md).

---

## Project Structure

```
sre_arena_env/
├── client.py              ← async/sync env client
├── models.py              ← DefenderAction, AttackerAction, Observation schemas
├── inference.py           ← hackathon-mandatory baseline (defender + attacker)
├── BLOG.md                ← full submission write-up
├── server/
│   ├── app.py            ← FastAPI server
│   ├── sre_arena_env_environment.py
│   └── simulator/        ← sim_nginx, sim_express, rubric scoring
├── training/
│   ├── train_defender.py / train_attacker.py
│   ├── alternating_loop.py
│   ├── opponent_loader.py     ← Phase 7a PEFT cross-gen loading
│   ├── prompts.py / action_parser.py / dataset_builder.py / reward_function.py
├── hf_jobs/
│   ├── train_on_hf_jobs.py
│   ├── run_inference_demo.py
│   └── inference_demo.json
├── notebooks/
│   └── main_training.ipynb    ← Colab T4 demo
├── configs/
│   ├── colab_demo.yaml
│   └── l4_training.yaml
└── tests/                     ← 216 passing tests
```

---

## Tech Stack

- **Python 3.11**, **PyTorch 2.10**, **CUDA 12.x**
- **OpenEnv** ≥ 0.2.3 — environment framework
- **TRL** ≥ 0.29 — GRPO algorithm
- **PEFT** + **bitsandbytes** — QLoRA quantized fine-tuning
- **transformers**, **Qwen2.5-7B-Instruct** base model
- **FastAPI**, **WebSockets**, **Gradio** — server + dashboard
- **OpenAI client** — for inference.py (any OpenAI-compatible LLM endpoint)

---

## Future Work

- **Approach 3 (observation-aware opponents):** Eliminate Gen 1's gradient collapse by sharing observations between defender and attacker during reward eval
- **Reward shaping for surgical defenses:** Bonus for IP-specific blocks vs. blanket rate limits — teach finer-grained policy selection
- **Real-stack Docker validation:** Verify trained-on-sim policies transfer to actual nginx + Express
- **Expanded attack templates:** GraphQL injection, JWT forgery, race conditions, prototype pollution
- **Production WAF rule export:** Translate to ModSecurity / Cloudflare formats

---

## Acknowledgements

Built for **Meta × Scaler PyTorch OpenEnv Hackathon 2026**. Thank you to:

- The **OpenEnv team** at Meta + Hugging Face for the framework
- **Hugging Face** for hosting (Spaces, Inference Providers, Jobs)
