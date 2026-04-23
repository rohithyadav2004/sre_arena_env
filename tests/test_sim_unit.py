"""Unit tests for SimulatedNginx and SimulatedExpress."""
from __future__ import annotations

import pytest

from sre_arena_env.server.simulator.sim_nginx import SimulatedNginx


# ── helpers ───────────────────────────────────────────────────────────────────


def _req(
    ip: str,
    path: str = "/",
    method: str = "GET",
    body: dict | None = None,
    headers: dict | None = None,
) -> dict:
    return {
        "ip": ip,
        "path": path,
        "method": method,
        "body": body or {},
        "headers": headers or {},
        "is_malicious": False,
    }


# ── SimulatedNginx ────────────────────────────────────────────────────────────


class TestDenyAllow:
    def test_deny_blocks_matching_ip(self):
        nginx = SimulatedNginx()
        assert nginx.add_rule("deny 10.0.0.1;") is True
        assert nginx.process_request(_req("10.0.0.1")) == 403

    def test_deny_does_not_block_other_ip(self):
        nginx = SimulatedNginx()
        nginx.add_rule("deny 10.0.0.1;")
        assert nginx.process_request(_req("10.0.0.2")) == 200

    def test_deny_cidr_blocks_matching(self):
        nginx = SimulatedNginx()
        nginx.add_rule("deny 10.0.0.0/24;")
        assert nginx.process_request(_req("10.0.0.55")) == 403

    def test_deny_cidr_does_not_block_outside(self):
        nginx = SimulatedNginx()
        nginx.add_rule("deny 10.0.0.0/24;")
        assert nginx.process_request(_req("10.0.1.1")) == 200

    def test_allow_before_deny_permits_specific_ip(self):
        nginx = SimulatedNginx()
        nginx.add_rule("allow 10.0.0.5;")
        nginx.add_rule("deny 10.0.0.0/24;")
        assert nginx.process_request(_req("10.0.0.5")) == 200
        assert nginx.process_request(_req("10.0.0.9")) == 403

    def test_no_rules_allows_all(self):
        nginx = SimulatedNginx()
        assert nginx.process_request(_req("1.2.3.4")) == 200

    def test_invalid_rule_rejected_returns_false(self):
        nginx = SimulatedNginx()
        assert nginx.add_rule("block all traffic please;") is False

    def test_invalid_rule_not_added_to_rules_list(self):
        nginx = SimulatedNginx()
        nginx.add_rule("garbage;")
        assert nginx.get_rules() == []


class TestRateLimit:
    def _nginx_with_zone(
        self, zone: str = "flood", rate: int = 10, burst: int = 5
    ) -> SimulatedNginx:
        nginx = SimulatedNginx()
        nginx.add_rule(
            f"limit_req_zone $binary_remote_addr zone={zone}:10m rate={rate}r/s;"
        )
        nginx.add_rule(f"limit_req zone={zone} burst={burst} nodelay;")
        return nginx

    def test_burst_allows_up_to_burst_requests(self):
        nginx = self._nginx_with_zone(burst=5)
        for i in range(5):
            assert nginx.process_request(_req("10.0.0.1")) == 200, (
                f"request {i + 1} should be 200"
            )

    def test_burst_blocks_on_burst_plus_one(self):
        nginx = self._nginx_with_zone(burst=5)
        for _ in range(5):
            nginx.process_request(_req("10.0.0.1"))
        assert nginx.process_request(_req("10.0.0.1")) == 429

    def test_rate_limit_is_per_ip_not_global(self):
        nginx = self._nginx_with_zone(burst=3)
        for _ in range(3):
            nginx.process_request(_req("10.0.0.1"))
        assert nginx.process_request(_req("10.0.0.2")) == 200

    def test_reset_step_counters_refills_bucket(self):
        nginx = self._nginx_with_zone(burst=2)
        for _ in range(2):
            nginx.process_request(_req("10.0.0.1"))
        assert nginx.process_request(_req("10.0.0.1")) == 429
        nginx.reset_step_counters()
        assert nginx.process_request(_req("10.0.0.1")) == 200


class TestLog:
    def test_get_log_tail_returns_n_lines(self):
        nginx = SimulatedNginx()
        for i in range(10):
            nginx.process_request(_req(f"1.2.3.{i}"))
        tail = nginx.get_log_tail(3)
        assert len(tail.strip().split("\n")) == 3

    def test_get_log_tail_empty_when_no_requests(self):
        nginx = SimulatedNginx()
        assert nginx.get_log_tail(5) == ""


# ── SimulatedExpress ──────────────────────────────────────────────────────────

from sre_arena_env.server.simulator.sim_express import SimulatedExpress  # noqa: E402


class TestExpressMiddleware:
    def _req(
        self,
        path: str = "/api/process",
        ip: str = "1.2.3.4",
        body: dict | None = None,
        headers: dict | None = None,
    ) -> dict:
        return {
            "ip": ip,
            "path": path,
            "method": "POST",
            "body": body or {},
            "headers": headers or {},
            "is_malicious": False,
        }

    # ── body-field check ──────────────────────────────────────────────────────

    def test_body_field_check_blocks_matching_value(self):
        ex = SimulatedExpress()
        ok = ex.add_middleware(
            "/api/process",
            "if (req.body.command === 'rm') return res.status(403)",
        )
        assert ok is True
        assert ex.process_request(self._req(body={"command": "rm"})) == 403

    def test_body_field_check_allows_different_value(self):
        ex = SimulatedExpress()
        ex.add_middleware(
            "/api/process",
            "if (req.body.command === 'rm') return res.status(403)",
        )
        assert ex.process_request(self._req(body={"command": "echo"})) == 200

    def test_body_field_check_allows_missing_field(self):
        ex = SimulatedExpress()
        ex.add_middleware(
            "/api/process",
            "if (req.body.command === 'rm') return res.status(403)",
        )
        assert ex.process_request(self._req(body={})) == 200

    # ── header check ─────────────────────────────────────────────────────────

    def test_header_check_blocks_when_header_present(self):
        ex = SimulatedExpress()
        ok = ex.add_middleware(
            "/login",
            "if (req.headers['X-Forwarded-For']) return res.status(403)",
        )
        assert ok is True
        assert (
            ex.process_request(
                self._req(path="/login", headers={"X-Forwarded-For": "1.2.3.4"})
            )
            == 403
        )

    def test_header_check_allows_when_header_absent(self):
        ex = SimulatedExpress()
        ex.add_middleware(
            "/login",
            "if (req.headers['X-Forwarded-For']) return res.status(403)",
        )
        assert ex.process_request(self._req(path="/login", headers={})) == 200

    # ── IP check ─────────────────────────────────────────────────────────────

    def test_ip_check_blocks_matching_ip(self):
        ex = SimulatedExpress()
        ok = ex.add_middleware(
            "/api/admin",
            "if (req.ip === '10.0.0.1') return res.status(403)",
        )
        assert ok is True
        assert ex.process_request(self._req(path="/api/admin", ip="10.0.0.1")) == 403

    def test_ip_check_allows_different_ip(self):
        ex = SimulatedExpress()
        ex.add_middleware(
            "/api/admin",
            "if (req.ip === '10.0.0.1') return res.status(403)",
        )
        assert ex.process_request(self._req(path="/api/admin", ip="10.0.0.2")) == 200

    # ── unrecognised middleware ───────────────────────────────────────────────

    def test_unrecognized_middleware_returns_false(self):
        ex = SimulatedExpress()
        assert ex.add_middleware("/login", "console.log('hello');") is False

    def test_unrecognized_middleware_has_no_effect_on_requests(self):
        ex = SimulatedExpress()
        ex.add_middleware("/login", "console.log('hello');")
        assert ex.process_request(self._req(path="/login")) == 200

    def test_unrecognized_count_increments(self):
        ex = SimulatedExpress()
        ex.add_middleware("/login", "bad js 1;")
        ex.add_middleware("/login", "bad js 2;")
        assert ex.unrecognized_count == 2

    # ── route scoping ─────────────────────────────────────────────────────────

    def test_middleware_only_applies_to_its_route(self):
        ex = SimulatedExpress()
        ex.add_middleware(
            "/login",
            "if (req.body.command === 'rm') return res.status(403)",
        )
        assert (
            ex.process_request(
                {
                    "ip": "1.2.3.4",
                    "path": "/api/data",
                    "body": {"command": "rm"},
                    "headers": {},
                    "is_malicious": False,
                }
            )
            == 200
        )

    def test_invalid_route_rejected(self):
        ex = SimulatedExpress()
        assert (
            ex.add_middleware(
                "/nonexistent",
                "if (req.body.x === 'y') return res.status(403)",
            )
            is False
        )

    def test_get_middleware_summary_returns_dict(self):
        ex = SimulatedExpress()
        js = "if (req.body.command === 'rm') return res.status(403)"
        ex.add_middleware("/api/process", js)
        summary = ex.get_middleware_summary()
        assert "/api/process" in summary
        assert summary["/api/process"] == js
