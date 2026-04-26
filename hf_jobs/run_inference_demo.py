#!/usr/bin/env python3
"""Run inference on Gen 2 defender — capture sample actions for BLOG."""
import json
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi, login

# Auth
hf_token = os.environ.get("HF_TOKEN")
if hf_token:
    login(token=hf_token)

print("=" * 70)
print("Gen 2 Defender Inference Demo")
print("=" * 70)

base_name = "Qwen/Qwen2.5-7B-Instruct"
adapter_name = "blitz1809/sre-arena-defender-gen2"

bnb = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_quant_type="nf4",
)

print(f"\nLoading {base_name}...")
base = AutoModelForCausalLM.from_pretrained(
    base_name,
    quantization_config=bnb,
    device_map="auto",
    dtype=torch.bfloat16,
)
tok = AutoTokenizer.from_pretrained(base_name)

print(f"Loading PEFT adapter from {adapter_name}...")
model = PeftModel.from_pretrained(base, adapter_name)
model.eval()

print("\nModel loaded. Running 3 scenarios.\n")

SYSTEM = """You are a blue-team SRE agent protecting an nginx + Express.js web service from Layer 7 attacks.

Each turn, you receive an observation containing a recent log tail, your current rules, current middleware, and last-step traffic metrics. You must respond with EXACTLY ONE JSON object — no markdown, no explanation, no preamble.

Three actions are available:

1. Inspect the log to discover attacker IPs/patterns:
   {"action_type": "read_log", "log_tail_lines": 50}

2. Append an nginx rule (block IPs or apply rate limiting):
   {"action_type": "append_nginx_rule", "rule_text": "deny 10.0.1.1;"}
   {"action_type": "append_nginx_rule", "rule_text": "limit_req zone=per_ip burst=5 nodelay;"}

3. Write Express middleware (for payload-based attacks):
   {"action_type": "write_express_middleware", "route": "/api/process", "middleware_js": "if (req.body.command) return res.status(403).end();"}

Reward: (malicious_blocked / total_malicious) × (legit_allowed / total_legit).
Output ONLY valid JSON, no other text.
"""

scenarios = [
    {
        "name": "Single IP flood attack on /login",
        "description": "Brute-force credential stuffing from one IP.",
        "obs": (
            "Last step metrics: {\"malicious_blocked\": 0, \"total_malicious\": 50, \"legit_allowed\": 12, \"total_legit\": 12}\n"
            "Current rules: []\n"
            "Recent access log:\n"
            "10.0.5.42 - - [12:00:01] \"POST /login HTTP/1.1\" 401\n"
            "10.0.5.42 - - [12:00:01] \"POST /login HTTP/1.1\" 401\n"
            "10.0.5.42 - - [12:00:01] \"POST /login HTTP/1.1\" 401\n"
            "10.0.5.42 - - [12:00:02] \"POST /login HTTP/1.1\" 401\n"
            "10.0.5.42 - - [12:00:02] \"POST /login HTTP/1.1\" 401\n"
            "192.168.1.10 - - [12:00:03] \"GET /api/data HTTP/1.1\" 200\n"
            "10.0.5.42 - - [12:00:03] \"POST /login HTTP/1.1\" 401\n"
            "192.168.1.11 - - [12:00:03] \"GET /api/data HTTP/1.1\" 200\n"
            "10.0.5.42 - - [12:00:04] \"POST /login HTTP/1.1\" 401\n"
            "10.0.5.42 - - [12:00:04] \"POST /login HTTP/1.1\" 401\n"
        ),
        "expected_action_pattern": "deny 10.0.5.42",
    },
    {
        "name": "Distributed flood (rotating IPs in 172.16.0.*/16)",
        "description": "DDoS from a botnet — needs rate limiting, not single-IP blocking.",
        "obs": (
            "Last step metrics: {\"malicious_blocked\": 0, \"total_malicious\": 80, \"legit_allowed\": 15, \"total_legit\": 15}\n"
            "Current rules: []\n"
            "Recent access log:\n"
            "172.16.0.5 - - [12:01:00] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.42 - - [12:01:00] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.91 - - [12:01:00] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.18 - - [12:01:00] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.66 - - [12:01:01] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.123 - - [12:01:01] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.201 - - [12:01:01] \"GET /api/data HTTP/1.1\" 200\n"
            "192.168.1.10 - - [12:01:02] \"GET /api/data HTTP/1.1\" 200\n"
            "172.16.0.55 - - [12:01:02] \"GET /api/data HTTP/1.1\" 200\n"
            "(...rotating 172.16.0.* IPs continue, ~80 reqs/sec...)\n"
        ),
        "expected_action_pattern": "limit_req or 172.16.0",
    },
    {
        "name": "Payload injection on /api/process",
        "description": "Command injection requires Express middleware, not nginx rules.",
        "obs": (
            "Last step metrics: {\"malicious_blocked\": 0, \"total_malicious\": 30, \"legit_allowed\": 8, \"total_legit\": 8}\n"
            "Current rules: []\n"
            "Current middleware: []\n"
            "Recent access log:\n"
            "10.0.10.5 - - [12:02:00] \"POST /api/process HTTP/1.1\" 200 body={\"command\": \"cat /etc/passwd\"}\n"
            "10.0.10.6 - - [12:02:00] \"POST /api/process HTTP/1.1\" 200 body={\"command\": \"rm -rf /\"}\n"
            "10.0.10.7 - - [12:02:01] \"POST /api/process HTTP/1.1\" 200 body={\"command\": \"whoami\"}\n"
            "192.168.1.10 - - [12:02:01] \"POST /api/process HTTP/1.1\" 200 body={\"data\": \"normal_payload\"}\n"
            "10.0.10.8 - - [12:02:02] \"POST /api/process HTTP/1.1\" 200 body={\"command\": \"ls -la /var\"}\n"
            "192.168.1.11 - - [12:02:02] \"POST /api/process HTTP/1.1\" 200 body={\"data\": \"another_normal\"}\n"
            "10.0.10.9 - - [12:02:03] \"POST /api/process HTTP/1.1\" 200 body={\"command\": \"id\"}\n"
        ),
        "expected_action_pattern": "write_express_middleware or req.body.command",
    },
]

outputs = []
for i, scenario in enumerate(scenarios, 1):
    print(f"\n--- Scenario {i}: {scenario['name']} ---")
    print(f"Context: {scenario['description']}")

    messages = [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": scenario['obs'] + "\n\nOutput one JSON action."}
    ]
    prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tok(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=200,
            do_sample=False,
            temperature=0.1,
            pad_token_id=tok.eos_token_id,
        )

    response = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).strip()
    print(f"\nDefender action:\n{response}\n")

    outputs.append({
        "scenario_num": i,
        "name": scenario['name'],
        "description": scenario['description'],
        "observation": scenario['obs'],
        "defender_action": response,
        "expected_pattern": scenario['expected_action_pattern'],
    })

# Save outputs as JSON for the BLOG
with open("/tmp/inference_demo.json", "w") as f:
    json.dump(outputs, f, indent=2)

print("\n" + "=" * 70)
print("Inference complete. Uploading to HF Hub...")
print("=" * 70)

# Upload outputs to HF Hub so we can retrieve them locally
api = HfApi()
upload_repo = "blitz1809/sre-arena-defender-gen2"
api.upload_file(
    path_or_fileobj="/tmp/inference_demo.json",
    path_in_repo="inference_demo.json",
    repo_id=upload_repo,
    repo_type="model",
)
print(f"\nUploaded inference_demo.json to {upload_repo}")
print("To retrieve locally: hf hub download blitz1809/sre-arena-defender-gen2 inference_demo.json")
