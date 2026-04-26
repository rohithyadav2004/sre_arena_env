# Building an SRE Arena: Self-Play RL for Layer 7 Web Defense

*How we taught a 7B Qwen model to defend a web service against itself — and what happened when self-play collapsed.*

---

## The Hook

Site reliability engineers face a relentless category of threat: Layer 7 attacks that look like normal traffic. Credential stuffing submits valid-looking login payloads at scale. HTTP floods are indistinguishable from viral traffic spikes. Payload injection hides SQL or shell in JSON fields. Today's defenses are mostly hand-written nginx rules and middleware — brittle, require expert authorship, and break silently when attackers change their patterns.

The question we wanted to answer: *can an LLM learn to write surgical web defenses by playing against an adversary that's also learning?*

We built SRE Arena — an OpenEnv environment where a blue-team defender and red-team attacker alternate training generations, each trying to beat the other's latest checkpoint. We trained Qwen2.5-7B with QLoRA + GRPO across 3 self-play generations on a single A100. The first generation learned beautifully (reward -0.03 → 1.0). The second collapsed to zero gradient. The third converged in 3 gradient steps. Here's why.

---

## The Environment: SRE Arena

SRE Arena is an [OpenEnv](https://github.com/meta-pytorch/openenv) environment where two LLM agents fight on simulated web infrastructure. The blue-team (defender) agent writes nginx rules and Express middleware to block incoming attacks. The red-team (attacker) agent chooses from 8 attack templates — credential stuffing, HTTP flood, SQL injection, path traversal, header injection, rate-limit evasion, bot spoofing, and payload fuzzing — to bypass the defender's rules.

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

The core insight behind the simulator: running RL against a real nginx + Node.js stack would mean Docker cold-starts on every rollout — roughly 2–3 seconds per step. With 150 training steps × 8 rollouts = 1,200 trajectories per generation, that's over an hour of pure I/O overhead. We built `sim_nginx` and `sim_express` as pure-Python simulators that reproduce nginx's routing semantics and Express middleware chains with ~1 ms per step — roughly a 1000x speedup that makes GRPO feasible.

The environment ships with 216 tests. Sim-to-real fidelity is validated at the unit level: our leaky-bucket rate-limiter matches nginx's `limit_req` behavior under burst traffic to within rounding error.

---

## The Anti-Exploit Reward (Key Contribution)

Most RL reward functions for security tasks use a simple sum: "block malicious traffic" + "allow legit traffic." We discovered this leads immediately to degenerate policies.

Our reward formula is multiplicative:

```
score = (malicious_blocked / total_malicious) × (legit_allowed / total_legit) - middleware_penalty
```

The difference between additive and multiplicative becomes clear with extreme strategies:

| Strategy | mal_blocked | legit_allowed | Additive reward | Multiplicative |
|----------|-------------|---------------|-----------------|----------------|
| Block everything | 100% | 0% | 1.0 | **0.0** |
| Allow everything | 0% | 100% | 1.0 | **0.0** |
| Surgical defense | 95% | 98% | 1.93 | **0.93** |

The multiplicative form forces the agent to balance both objectives — it can't reward-hack by going extreme on either dimension. This is the same mathematical insight behind F1 score vs. precision+recall: you can inflate either metric individually, but their harmonic mean (or product) only stays high when both are high.

Early experiments with an additive reward function produced exactly the degenerate case: within 20 steps, the defender learned to write `deny all;` as the first nginx rule, achieving 100% malicious blocking and 0% legitimate traffic throughput — reward 1.0, completely useless policy. The multiplicative form eliminated this within one smoke-test run.

The `-0.05 middleware_penalty` is applied for each middleware function added beyond the first. This discourages the agent from writing redundant middleware chains that would cause latency and maintenance overhead in production.

---

## The Algorithm: Why GRPO

Most RL algorithms (PPO, A2C) need a "critic" — a separate neural network that estimates how good each state is. The critic takes memory, takes wall-clock time to train, and introduces a second optimization target that can conflict with the policy. GRPO (Group Relative Policy Optimization, introduced in DeepSeek-R1) skips the critic entirely.

Here's how it works:

1. For each prompt, generate K rollouts — we used K=8
2. Compute the deterministic [0,1] reward for each rollout
3. Advantage of rollout i = `(reward_i - group_mean) / group_std`
4. Update the policy toward rollouts with positive advantage, away from negative

Why this matters for SRE Arena specifically:

- **Memory:** no critic ≈ 30% VRAM saved → fits Qwen2.5-7B QLoRA on a single A100 80GB
- **Stability:** group-relative advantages have lower variance than absolute returns
- **Sample efficiency:** 150 steps × 8 rollouts = 1,200 trajectories total; PPO typically needs ~10x more for equivalent convergence on verifiable-reward tasks
- **Designed for verifiable rewards:** our env produces deterministic [0,1] scores — the ideal GRPO setup where the reward oracle is cheap and exact

GRPO's design has one critical implication that we didn't fully appreciate until Gen 1: when `group_std = 0` (all 8 rollouts return identical reward), the advantage becomes `0/0` — undefined, treated as zero — and the gradient vanishes. We'll see this exact failure mode in Section 7.

This algorithm is what enabled the 74-minute Gen 0 training run on a single A100.

---

## Training Setup

Quick specs for reproducibility:

- **Base model:** Qwen2.5-7B-Instruct
- **PEFT:** QLoRA — 10.1M trainable parameters (0.13% of 7.6B total)
- **Precision:** bf16, A100-80GB
- **Quantization:** 4-bit base weights; LoRA applied to q/k/v/o projections, rank 16
- **Trainer:** TRL 0.29 GRPOTrainer
- **Per generation:** 150 training steps, 8 rollouts per step
- **Reward:** anti-exploit multiplicative formula with `-0.05` middleware penalty
- **Generations:** 3 (defender → attacker vs Gen 0 def → defender vs Gen 1 atk)

One non-obvious setup detail: **bf16 vs fp16.** The newer HF Jobs A100 environment ships with a transformers + bitsandbytes stack that expects bf16 throughout, with no GradScaler. Forcing fp16 on this stack throws `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16` — a cryptic error that took 4 job attempts to trace. The fix is to trust the model config's native dtype (`torch_dtype=bfloat16`) and remove `fp16=True` from the TrainingArguments. If you're adapting this to GCP Vertex with an older transformers stack, you'll need the opposite: fp16 + GradScaler.

---

## Results: 3-Generation Self-Play

![All 3 generations overlaid. Gen 0 (blue) climbs from -0.03 to 1.0 over 30 logging steps. Gen 1 (red) appears saturated at 1.0 from step 1 — but the gradient norm reveals no learning occurred. Gen 2 (green) starts at 0.21 and hits 1.0 by step 7.](multi_gen_overlay.png)

Three generations. Three distinct stories. One is a clean success, one is a mathematically provable failure, and one validates that the infrastructure survived the failure.

### Gen 0 — The Defender Learns from Scratch

Wall clock: **74 minutes** on A100.

The Gen 0 defender starts with zero knowledge of nginx syntax or the attack patterns it's facing. For the first ~10 logging steps, reward hovers near zero — the model is exploring its action space, generating malformed rules that the simulator rejects or that fail to match any traffic. Around step 11, something clicks: the first valid `deny <ip>;` rule lands and the reward curve begins its takeoff. By step 21, reward reaches 1.0 — perfect malicious blocking with perfect legitimate traffic throughput.

Final trajectory: **-0.0325 → 1.0** over 30 logging steps.

![Gen 0 policy entropy (blue) drops smoothly from 0.13 to 0.02 — the textbook exploration→exploitation arc. Gen 1 (red) stays high and chaotic, oscillating 0.02–0.045 with no convergence signal. Gen 2 (green) crashes to 0.01 by step 5 — the most decisive convergence of all three.](entropy_curves.png)

The entropy curve tells the behavioral story clearly. Gen 0 starts with high entropy (0.13) — the model is uncertain, sampling widely from its action distribution. As the reward signal strengthens after step 11, entropy drops monotonically to 0.02 by step 30. This is the exploitation phase: the model has learned which action pattern works and converges on it.

![Gen 0 gradient norm (blue) shows punctuated bursts at steps ~3, ~11, and ~19 (values 0.3–0.6), with zero gradient between. Gen 1 (red) is identically zero throughout — axis range -0.04 to +0.04 confirms no updates occurred. Gen 2 (green) spikes at steps 1–3 (~0.4 peak) then drops to zero by step 4.](grad_norm_curves.png)

The gradient norm tells a more nuanced story. Gen 0 shows three punctuated bursts at steps ~3, ~11, ~19 — reaching 0.3–0.6 — with zero gradient between. This is characteristic of GRPO with sparse rewards: the model only updates when it discovers something new (a valid rule that changes the reward distribution across the rollout group). Between discoveries, all 8 rollouts produce identical rewards, `group_std = 0`, advantage = 0, gradient = 0. The 5x sample efficiency over PPO comes at the cost of bursty, non-smooth learning dynamics.

### Gen 1 — The Attacker's Gradient Signal Collapses

Wall clock: **5 hours 5 minutes** on A100.

The reward plot for Gen 1 looks like the attacker won: flat at 1.0 from step 1. The tempting interpretation is "saturation" — the attacker quickly found attacks that always succeed. **This interpretation is wrong.** The gradient norm shows the truth.

**Gen 1's gradient norm is identically zero across all 30 steps.** The y-axis range of -0.04 to +0.04 confirms this is not a scaling artifact — it is literal zero. The attacker never changed its weights once during a 5-hour training run.

Here is the mechanistic explanation. Our Phase 7a opponent loading uses "blind generation" (Approach 1): when the attacker's reward function evaluates a candidate action, it loads the Gen-0 defender as a frozen PEFT model and queries it for a defense — but without passing the actual environment observation. The defender, given no context, outputs useless or invalid actions. Every attack succeeds trivially. Every reward is 1.0.

With every rollout returning 1.0, `group_std = 0`. GRPO advantage = `0/0 = 0`. Gradient = 0. No policy update.

![frac_reward_zero_std across generations. Gen 0 (blue) bounces between 0.4–1.0 — variance is present, learning signal exists. Gen 1 (red) is flat at 1.0 from step 1 — every group of 8 rollouts returned identical reward, making GRPO advantage undefined (zero). Gen 2 (green) starts at 0.0 (maximum variance, maximum signal), jumps to 1.0 by step 6 (model locked in).](grpo_learning_signal.png)

The GRPO learning signal metric (`frac_reward_zero_std`) confirms this quantitatively: Gen 1 is flat at 1.0 from step 1 — meaning 100% of rollout groups had zero reward variance, meaning 100% of gradient steps were null operations.

Meanwhile, Gen 1's entropy stays high and chaotic (oscillating 0.02–0.045) because there's no signal to drive convergence. The model is sampling randomly, getting 1.0 reward regardless of what it outputs, and never moving.

This is a documented failure mode of self-play with blind-context opponents. We weren't surprised it happened — we were prepared for it — but we have the diagnostic metrics to prove it cleanly. Most self-play tutorials skip over this failure mode; we made it the centerpiece of our Gen 1 analysis.

### Gen 2 — The Defender Re-Learns in 3 Gradient Steps

Wall clock: **5 hours 5 minutes** on A100 (similar to Gen 1; the cost is in the 200-token rollouts, not the optimizer steps).

Gen 2 trains a fresh defender against the frozen Gen 1 attacker checkpoint. Starting reward: **0.21** — meaningfully higher than Gen 0's -0.03, because the broken Gen 1 attacker still produces varied (if random) attacks that give the defender a non-trivial learning signal.

The entropy curve for Gen 2 crashes from 0.08 to 0.01 by step 5 — the most decisive convergence of all three generations.

The gradient norm tells the sharpest story: **a single spike at steps 1–3 (~0.4 peak), then zero by step 4.** Three gradient updates and the model is done. The GRPO learning signal starts at 0.0 (maximum variance across rollout groups — strong signal) and jumps to 1.0 by step 6 (all groups identical — model locked in). Reward reaches 1.0 by step 7.

The Gen 2 defender was trained from base Qwen weights with a freshly initialized LoRA — nothing carried over from Gen 0 in the model parameters. What "transferred" was the alternating-loop infrastructure correctly handing off the Gen 1 attacker checkpoint as the frozen opponent. With strong reward variance present at the start, GRPO converged in 3 gradient steps. This validates that the cross-generation training pipeline is functionally correct, even when an intermediate generation fails.

**The combined story:** Gen 0 proved the env produces real learning curves. Gen 1 exposed the blind-context opponent collapse — with mathematical proof via gradient norm, reward variance, and entropy metrics. Gen 2 proved the infrastructure correctly handles the next iteration. Future work (Approach 3, observation-aware opponents) addresses Gen 1's failure mode directly.

---

## Challenges & What Broke

Honest engineering disclosure from a project that went wrong in interesting ways.

**Mergekit dependency hell.** TRL 0.24–0.28 imported `mergekit` at module load. On HF Jobs, `mergekit` resolved to a version that conflicted with our pinned `transformers`. The symptom was a clean import failure with no useful traceback. Solution: pin `trl>=0.29` — the mergekit dependency was removed in the 0.29 release. Lesson: always pin to the release that dropped the transitive dep, not the one that has a compatible version of it.

**bf16 vs fp16 mismatch.** Older transformers on GCP Vertex wants `fp16=True` + GradScaler. Newer transformers on HF Jobs wants `bf16=True` + no GradScaler. Forcing fp16 on the newer stack threw `_amp_foreach_non_finite_check_and_unscale_cuda not implemented for BFloat16` — a CUDA kernel error that reveals nothing about its own root cause. Took 4 HF Jobs attempts to isolate (each attempt costs ~20 minutes of queue + startup time). The fix: trust the model config's native dtype (`bfloat16`), remove `fp16=True` from TrainingArguments, remove the GradScaler. Lesson: when switching compute hosts, assume dtype defaults have changed.

**Reward signal collapse in early smoke tests.** 100% of completions hit `max_tokens` with reward=0. Root cause: the traffic generator hadn't seeded the log buffer before the first defender step. With an empty log, the defender had nothing to pattern-match, so every generated rule was a no-op against zero traffic. Fix: add a warm-up step in `env.reset()` that runs the attacker for one cycle before the defender's first observation. This is now covered in the integration test suite.

**Phase 7a opponent loading complexity.** Cross-generation training requires the previous generation's PEFT checkpoint loaded as a frozen opponent during rollout evaluation. Implementing this without breaking TRL's GRPOTrainer required a separate `OpponentModel` wrapper class with isolated forward passes and explicit `torch.no_grad()` contexts. We chose Approach 1 (blind generation — no environment observation passed to the opponent) for implementation simplicity, accepting the known limitation that it would collapse to zero gradient when the opponent always succeeds trivially. The training logs proved this prediction exactly.

**The Gen 1 zero-gradient discovery.** When Gen 1's reward stayed at 1.0 throughout, the initial read was that the attacker had saturated and won. The `grad_norm` plot corrected that interpretation immediately: not winning — not moving. This is the kind of failure mode you only catch by logging GRPO's auxiliary metrics (`frac_reward_zero_std`, `grad_norm`) alongside the headline reward. Headline metrics lie. Mechanistic metrics don't.

---

## What We'd Do Differently / Future Work

- **Approach 3 (observation-aware opponents):** Instead of blind generation, run the frozen opponent with shared environment observations — it sees the actual traffic log before generating a response. This eliminates the `group_std = 0` failure by coupling the opponent's output quality to real environment state, ensuring reward variance is non-zero.

- **Real-stack Docker validation:** The simulator gives ~1000x speedup but introduces sim-to-real risk. The next step is a Docker container with rootless nginx + Express that the trained agent defends in real time, verifying that policies learned in simulation transfer to the real request pipeline without degradation.

- **Expanded attack templates:** The current 8 templates cover the most common L7 patterns. Adding GraphQL injection, JWT forgery, race conditions, and prototype pollution would test whether a defender trained on 8 templates generalizes to a 9th unseen one — the core transfer question for production deployment.

- **Production WAF rule export:** Translate the trained agent's nginx rules to ModSecurity or Cloudflare ruleset format. The gap between "nginx simulator rule" and "production WAF rule" is narrower than it looks — the main additions are rule IDs, severity tags, and phase declarations. A trained agent that can emit production-ready ModSecurity rules would be immediately useful.

---

## Try It Yourself

**Live demo:** [HF Space](https://blitz1809-sre-arena.hf.space) — watch traffic flow and defenses applied in real time via the SSE dashboard.

**Train your own (Colab T4, free):** [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rohithyadav2004/sre_arena_env/blob/main/notebooks/main_training.ipynb) — 3-gen self-play in ~20 minutes on free hardware (1.5B model, 200 episodes).

**Production scale (HF Jobs A100):** See `hf_jobs/train_on_hf_jobs.py` for full 150-step runs. Trained checkpoints available on the Hub:
- Defender Gen 0: [blitz1809/sre-arena-defender-gen0](https://huggingface.co/blitz1809/sre-arena-defender-gen0)
- Attacker Gen 1: [blitz1809/sre-arena-attacker-gen1](https://huggingface.co/blitz1809/sre-arena-attacker-gen1)
- Defender Gen 2: [blitz1809/sre-arena-defender-gen2](https://huggingface.co/blitz1809/sre-arena-defender-gen2)

---

## Acknowledgements

Thank you to Meta, Scaler, and PyTorch for organizing this hackathon and providing the OpenEnv framework that made a clean self-play environment possible. Thank you to Hugging Face for hosting, Jobs compute, and the model Hub. And thank you to Anthropic Claude for pair-debugging through bf16 hell at 2am — the engineer experience of this project was genuinely collaborative.
