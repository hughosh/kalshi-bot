"""Tests for risk management."""
import pytest

from integration.risk import RiskManager


class TestRiskManager:
    def test_allows_within_limits(self):
        rm = RiskManager(max_contracts_per_match=50, max_total_exposure=500.0)
        allowed = rm.check_and_reduce("TICKER1", "yes", 10, 50)
        assert allowed == 10

    def test_caps_at_match_limit(self):
        rm = RiskManager(max_contracts_per_match=5, max_total_exposure=500.0)
        rm.confirm_fill("TICKER1", "yes", 3, 50)
        allowed = rm.check_and_reduce("TICKER1", "yes", 10, 50)
        assert allowed == 2

    def test_caps_at_match_limit_exact(self):
        rm = RiskManager(max_contracts_per_match=5, max_total_exposure=500.0)
        rm.confirm_fill("TICKER1", "yes", 5, 50)
        allowed = rm.check_and_reduce("TICKER1", "yes", 10, 50)
        assert allowed == 0

    def test_caps_at_exposure_limit(self):
        rm = RiskManager(max_contracts_per_match=100, max_total_exposure=10.0, fee_per_contract=0.07)
        # Each contract costs 0.50 + 0.07 = 0.57 → max ~17 contracts for $10
        allowed = rm.check_and_reduce("TICKER1", "yes", 50, 50)
        assert allowed == 17

    def test_different_matches_independent(self):
        rm = RiskManager(max_contracts_per_match=5, max_total_exposure=500.0)
        rm.confirm_fill("TICKER1", "yes", 5, 50)
        allowed = rm.check_and_reduce("TICKER2", "yes", 5, 50)
        assert allowed == 5

    def test_release_match(self):
        rm = RiskManager(max_contracts_per_match=5, max_total_exposure=500.0)
        rm.confirm_fill("TICKER1", "yes", 5, 50)
        assert rm.active_matches == 1
        rm.release_match("TICKER1")
        assert rm.active_matches == 0

    def test_summary(self):
        rm = RiskManager()
        rm.confirm_fill("T1", "yes", 3, 40)
        rm.confirm_fill("T1", "no", 2, 60)
        s = rm.summary()
        assert s["active_matches"] == 1
        assert s["matches"]["T1"]["total"] == 5


class TestRiskManagerExposure:
    def test_zero_contracts_noop(self):
        rm = RiskManager()
        allowed = rm.check_and_reduce("T", "yes", 0, 50)
        assert allowed == 0

    def test_exposure_tracking(self):
        rm = RiskManager(fee_per_contract=0.07)
        rm.confirm_fill("T", "yes", 10, 50)
        # 10 * (0.50 + 0.07) = 5.70
        assert abs(rm.total_exposure - 5.70) < 0.01
