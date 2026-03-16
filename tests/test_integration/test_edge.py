"""Tests for edge computation and position sizing."""
import pytest

from integration.edge import compute_edge, evaluate_signal, quarter_kelly_size


class TestComputeEdge:
    def test_yes_edge(self):
        side, edge, price = compute_edge(0.70, best_ask_cents=60, best_bid_cents=55)
        assert side == "yes"
        assert abs(edge - 0.10) < 1e-9
        assert price == 60

    def test_no_edge(self):
        side, edge, price = compute_edge(0.30, best_ask_cents=40, best_bid_cents=35)
        # model says 30% YES → 70% NO; market NO ask = 100-35 = 65c
        assert side == "no"
        assert abs(edge - 0.05) < 1e-9
        assert price == 65

    def test_no_edge_when_market_aligned(self):
        side, edge, price = compute_edge(0.50, best_ask_cents=50, best_bid_cents=50)
        assert side is None
        assert edge == 0.0

    def test_no_bids(self):
        side, edge, price = compute_edge(0.80, best_ask_cents=60, best_bid_cents=None)
        assert side == "yes"
        assert abs(edge - 0.20) < 1e-9

    def test_no_asks(self):
        side, edge, price = compute_edge(0.20, best_ask_cents=None, best_bid_cents=30)
        assert side == "no"
        assert abs(edge - 0.10) < 1e-9


class TestQuarterKelly:
    def test_basic_sizing(self):
        # edge=0.10, price=50c → payout_ratio=1.0 → kelly=0.10 → quarter=0.025
        # bankroll=$500, cost_per=0.50+0.07=0.57 → floor(0.025*500/0.57) = floor(21.9) = 21
        # But capped at max_contracts=20
        contracts = quarter_kelly_size(0.10, 50, 500.0, 20, 0.07)
        assert contracts == 20

    def test_zero_edge(self):
        assert quarter_kelly_size(0.0, 50, 500.0, 20, 0.07) == 0

    def test_zero_bankroll(self):
        assert quarter_kelly_size(0.10, 50, 0.0, 20, 0.07) == 0

    def test_extreme_price_high(self):
        # At 99c, payout_ratio = 1/99 → kelly very large → capped at max
        assert quarter_kelly_size(0.10, 99, 500.0, 20, 0.07) == 20

    def test_price_at_boundary(self):
        assert quarter_kelly_size(0.10, 100, 500.0, 20, 0.07) == 0

    def test_small_edge_small_size(self):
        contracts = quarter_kelly_size(0.02, 50, 100.0, 20, 0.07)
        # kelly=0.02/1.0=0.02, quarter=0.005, 0.005*100/0.57=0.87 → 0
        assert contracts == 0


class TestEvaluateSignal:
    def test_signal_emitted(self):
        signal = evaluate_signal(
            ticker="KXATPMATCH-TEST",
            model_prob=0.75,
            confidence=0.80,
            points_observed=20,
            best_ask_cents=60,
            best_bid_cents=55,
            bankroll=500.0,
        )
        assert signal is not None
        assert signal.side == "yes"
        assert signal.edge > 0.05
        assert signal.contracts > 0

    def test_low_confidence_rejected(self):
        signal = evaluate_signal(
            ticker="TEST",
            model_prob=0.75,
            confidence=0.40,
            points_observed=20,
            best_ask_cents=60,
            best_bid_cents=55,
            bankroll=500.0,
        )
        assert signal is None

    def test_too_few_points_rejected(self):
        signal = evaluate_signal(
            ticker="TEST",
            model_prob=0.75,
            confidence=0.80,
            points_observed=3,
            best_ask_cents=60,
            best_bid_cents=55,
            bankroll=500.0,
        )
        assert signal is None

    def test_small_edge_rejected(self):
        signal = evaluate_signal(
            ticker="TEST",
            model_prob=0.62,
            confidence=0.80,
            points_observed=20,
            best_ask_cents=60,
            best_bid_cents=55,
            bankroll=500.0,
        )
        assert signal is None
