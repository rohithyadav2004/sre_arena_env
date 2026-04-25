"""Quick HF Jobs smoke test — verifies repo install and GPU access."""
import subprocess
subprocess.run(["nvidia-smi"], check=True)

import torch
print(f"PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Try importing our package
from sre_arena_env.training.alternating_loop import run_alternating_loop
from sre_arena_env.training.opponent_loader import OpponentModel
print("Package imports OK")

# Quick model load test
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
print(f"Tokenizer loaded: vocab_size={tok.vocab_size}")

print("SMOKE TEST PASSED")
