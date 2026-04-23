"""Pure-Python nginx rule engine.

Supports deny/allow rules (IP + CIDR) and leaky-bucket rate limiting.
No real processes, no subprocesses — ~30µs per request.
"""
from __future__ import annotations

import ipaddress
import re
from datetime import datetime

_DENY_RE = re.compile(r"^deny\s+([\d./]+);$")
_ALLOW_RE = re.compile(r"^allow\s+([\d./]+);$")
_LIMIT_REQ_RE = re.compile(r"^limit_req\s+zone=(\w+)\s+burst=(\d+)\s+nodelay;$")
_LIMIT_REQ_ZONE_RE = re.compile(
    r"^limit_req_zone\s+\$binary_remote_addr\s+zone=(\w+):10m\s+rate=(\d+)r/s;$"
)


class SimulatedNginx:
    """Stateful nginx rule engine for the SRE Arena simulator.

    Supported rules:
    - ``deny <ip_or_cidr>;``
    - ``allow <ip_or_cidr>;``
    - ``limit_req zone=<name> burst=<n> nodelay;``
    - ``limit_req_zone $binary_remote_addr zone=<name>:10m rate=<n>r/s;``

    Rate limiting uses a per-step leaky-bucket: within each episode step
    each (zone, IP) pair is allowed ``burst`` requests. Call
    ``reset_step_counters()`` between steps to refill buckets.
    """

    def __init__(self) -> None:
        self._rules: list[str] = []
        self._log: list[str] = []
        self._zones: dict[str, dict] = {}         # zone_name -> {"rate": int}
        self._step_counts: dict[tuple, int] = {}  # (zone, ip) -> count this step

    # ── public API ───────────────────────────────────────────────────────────

    def add_rule(self, rule_text: str) -> bool:
        """Parse and store a rule. Returns False and ignores if unrecognised.

        Args:
            rule_text: Nginx directive, e.g. ``"deny 1.2.3.4;"``

        Returns:
            True if accepted, False if rejected as invalid.
        """
        rule = rule_text.strip()
        if _DENY_RE.match(rule) or _ALLOW_RE.match(rule) or _LIMIT_REQ_RE.match(rule):
            self._rules.append(rule)
            return True
        m = _LIMIT_REQ_ZONE_RE.match(rule)
        if m:
            self._zones[m.group(1)] = {"rate": int(m.group(2))}
            self._rules.append(rule)
            return True
        return False

    def process_request(self, req: dict) -> int:
        """Evaluate one request against current rules.

        Processing order mirrors nginx:
        1. Rate-limiting (``limit_req``) — returns 429 on exceeded.
        2. Allow/deny in declaration order — first match wins; returns 403 on deny.
        3. Default allow — returns 200.

        Args:
            req: Request dict; must contain at least ``"ip"``.

        Returns:
            HTTP status code: 200, 403, or 429.
        """
        ip = req.get("ip", "0.0.0.0")

        # Phase 1: rate limiting
        for rule in self._rules:
            m = _LIMIT_REQ_RE.match(rule)
            if m:
                zone, burst = m.group(1), int(m.group(2))
                key = (zone, ip)
                count = self._step_counts.get(key, 0)
                if count >= burst:
                    self._log_entry(req, 429)
                    return 429
                self._step_counts[key] = count + 1

        # Phase 2: allow/deny (first match wins)
        for rule in self._rules:
            dm = _DENY_RE.match(rule)
            if dm and self._ip_matches(ip, dm.group(1)):
                self._log_entry(req, 403)
                return 403
            am = _ALLOW_RE.match(rule)
            if am and self._ip_matches(ip, am.group(1)):
                break  # explicit allow — skip remaining allow/deny

        self._log_entry(req, 200)
        return 200

    def get_rules(self) -> list[str]:
        """Return ordered list of accepted rules."""
        return list(self._rules)

    def get_log_tail(self, n: int) -> str:
        """Return the last *n* log lines joined by newlines.

        Args:
            n: Number of trailing lines to return.

        Returns:
            Newline-joined log lines, or empty string if log is empty.
        """
        if not self._log:
            return ""
        return "\n".join(self._log[-n:])

    def reset_step_counters(self) -> None:
        """Refill leaky-bucket counters. Call once per episode step."""
        self._step_counts.clear()

    # ── internal helpers ─────────────────────────────────────────────────────

    def _ip_matches(self, ip: str, network_str: str) -> bool:
        try:
            addr = ipaddress.ip_address(ip)
            if "/" in network_str:
                return addr in ipaddress.ip_network(network_str, strict=False)
            return ip == network_str
        except ValueError:
            return False

    def _log_entry(self, req: dict, status: int) -> None:
        ts = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%S")
        self._log.append(
            f"{ts} {req.get('ip', '-')} "
            f'"{req.get("method", "GET")} {req.get("path", "/")}" {status}'
        )
