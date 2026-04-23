"""Episode traffic generator.

Each episode step consists of 80 legitimate requests plus the attacker's
output. Legitimate traffic uses a private IP pool and route weights that
mirror realistic production distributions.
"""
from __future__ import annotations

import random as _random_module

from .attack_templates import generate as _generate_attacks

_ROUTES = ["/login", "/api/data", "/api/process", "/api/admin", "/health"]
_ROUTE_WEIGHTS = [0.20, 0.30, 0.20, 0.10, 0.20]
_LEGIT_IP_POOL = [f"192.168.1.{i}" for i in range(1, 51)]
_LEGIT_METHODS = ["GET", "POST"]
_LEGIT_METHOD_WEIGHTS = [0.7, 0.3]


def generate_legit_batch(
    task_id: str,
    count: int,
    seed: int | None = None,
) -> list[dict]:
    """Generate a batch of legitimate (non-malicious) requests.

    Args:
        task_id: Scenario task identifier (reserved for future per-task profiles).
        count: Number of requests to generate.
        seed: Optional RNG seed for reproducibility.

    Returns:
        List of request dicts tagged ``is_malicious=False``.
    """
    rng = _random_module.Random(seed)
    requests = []
    for _ in range(count):
        ip = rng.choice(_LEGIT_IP_POOL)
        path = rng.choices(_ROUTES, weights=_ROUTE_WEIGHTS, k=1)[0]
        method = rng.choices(_LEGIT_METHODS, weights=_LEGIT_METHOD_WEIGHTS, k=1)[0]
        body: dict = {}
        if method == "POST" and path == "/login":
            body = {"username": f"user{rng.randint(1, 100)}", "password": "correct"}
        elif method == "POST" and path == "/api/process":
            body = {"data": f"value{rng.randint(1, 50)}"}
        requests.append(
            {
                "ip": ip,
                "method": method,
                "path": path,
                "headers": {},
                "body": body,
                "is_malicious": False,
            }
        )
    return requests


def generate_episode_traffic(
    task_id: str,
    attacker_action: dict | None,
    seed: int | None = None,
) -> list[dict]:
    """Generate one episode step's full traffic: 80 legit + attacker output.

    Args:
        task_id: Scenario task identifier.
        attacker_action: Dict matching AttackerAction fields (``template``,
            ``count``, ``target_path``, ``source_ips``, ``payload``), or
            ``None`` to generate only legitimate traffic.
        seed: Optional RNG seed applied to the legit batch only.

    Returns:
        Combined list of legit + malicious request dicts.
    """
    legit = generate_legit_batch(task_id, 80, seed=seed)
    if attacker_action is None:
        return legit
    template = attacker_action.get("template", "single_ip_flood")
    attacks = _generate_attacks(template, attacker_action)
    return legit + attacks
