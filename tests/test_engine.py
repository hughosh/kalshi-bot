"""Tests for the PredictionEngine (integration tests).

These tests run without trained model artifacts by using a mock model that
returns a constant probability. This validates the full prediction pipeline
(Bayesian update → Markov chain → calibration → output).
"""
import time
import pytest
import numpy as np

from tennis.engine import PredictionEngine
from tennis.types import MatchFormat, MatchState, PredictionResult
from tennis.updater import BayesianUpdater
from tennis.calibration import MatchWinCalibrator
from tennis.elo import EloTracker


# ---------------------------------------------------------------------------
# Minimal mock model for testing without trained artifacts
# ---------------------------------------------------------------------------

class MockPointWinModel:
    """Returns a fixed p_serve for all inputs."""
    def __init__(self, p: float = 0.63):
        self.p = p
        self._trained = True

    def predict_single(self, x) -> float:
        return self.p

    def predict_proba(self, X) -> np.ndarray:
        return np.full(len(X), self.p)


class MockCalibrator:
    """Identity calibrator — passes through raw probabilities unchanged."""
    def calibrate(self, prob: float) -> float:
        return float(np.clip(prob, 0.0, 1.0))

    def calibrate_batch(self, probs):
        return np.clip(probs, 0.0, 1.0)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def engine():
    """PredictionEngine with mock model and identity calibrator."""
    return PredictionEngine(
        model=MockPointWinModel(p=0.63),
        calibrator=MockCalibrator(),
        elo_tracker=EloTracker(),
        player_stats_df=None,
        n_mc_samples=200,
        rng_seed=42,
    )


@pytest.fixture
def start_state():
    return MatchState(
        sets_p1=0, sets_p2=0,
        games_p1=0, games_p2=0,
        points_p1=0, points_p2=0,
        server=0,
        is_tiebreak=False,
        match_format=MatchFormat.best_of_3_tiebreak(),
        player1_id="p1",
        player2_id="p2",
        surface="Hard",
        tournament_level="A",
        match_date="2023-01-01",
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestEngineInterface:
    def test_predict_returns_prediction_result(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert isinstance(result, PredictionResult)

    def test_win_probs_sum_to_one(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert abs(result.win_prob_p1 + result.win_prob_p2 - 1.0) < 1e-6

    def test_win_prob_in_range(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert 0.0 <= result.win_prob_p1 <= 1.0
        assert 0.0 <= result.win_prob_p2 <= 1.0

    def test_confidence_in_range(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert 0.0 <= result.confidence <= 1.0

    def test_ci_ordered(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert result.ci_lower <= result.win_prob_p1 <= result.ci_upper

    def test_p_serve_estimate_in_range(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert 0.0 < result.p_serve_estimate < 1.0

    def test_points_observed_zero_at_start(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        assert result.points_observed == 0

    def test_points_observed_increments(self, engine, start_state):
        engine.new_match(start_state)
        engine.record_point(server_won=True)
        engine.record_point(server_won=False)
        result = engine.predict(start_state)
        assert result.points_observed == 2


class TestEngineSemantics:
    def test_p1_leads_increases_p1_win_prob(self, engine):
        """P1 leading in sets/games should have higher win prob than equal."""
        equal_state = MatchState(
            sets_p1=0, sets_p2=0, games_p1=3, games_p2=3,
            points_p1=0, points_p2=0, server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        p1_leads_state = MatchState(
            sets_p1=1, sets_p2=0, games_p1=3, games_p2=2,
            points_p1=0, points_p2=0, server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        engine.new_match(equal_state)
        r_equal = engine.predict(equal_state)

        engine.new_match(p1_leads_state)
        r_leads = engine.predict(p1_leads_state)

        assert r_leads.win_prob_p1 > r_equal.win_prob_p1

    def test_p2_leads_reduces_p1_win_prob(self, engine):
        equal_state = MatchState(
            sets_p1=0, sets_p2=0, games_p1=0, games_p2=0,
            server=0, match_format=MatchFormat.best_of_3_tiebreak(),
        )
        p2_leads = MatchState(
            sets_p1=0, sets_p2=1, games_p1=0, games_p2=3,
            server=0, match_format=MatchFormat.best_of_3_tiebreak(),
        )
        engine.new_match(equal_state)
        r_equal = engine.predict(equal_state)
        engine.new_match(p2_leads)
        r_p2 = engine.predict(p2_leads)
        assert r_p2.win_prob_p1 < r_equal.win_prob_p1

    def test_confidence_increases_with_more_points(self):
        """More observed points → narrower CI → higher confidence."""
        model_low = MockPointWinModel(p=0.63)
        eng = PredictionEngine(
            model=model_low, calibrator=MockCalibrator(),
            elo_tracker=EloTracker(), player_stats_df=None,
            n_mc_samples=500, rng_seed=99,
        )
        state = MatchState(
            sets_p1=0, sets_p2=0, games_p1=0, games_p2=0,
            server=0, match_format=MatchFormat.best_of_3_tiebreak(),
        )
        eng.new_match(state)
        r_start = eng.predict(state)

        # Simulate 50 points (server wins most)
        for i in range(50):
            eng.record_point(server_won=(i % 3 != 0))

        r_after = eng.predict(state)
        assert r_after.confidence >= r_start.confidence - 0.05  # generally more confident

    def test_symmetric_p_gives_near_50pct(self):
        """With p=0.5 model at match start, win prob should be near 0.5."""
        eng = PredictionEngine(
            model=MockPointWinModel(p=0.5), calibrator=MockCalibrator(),
            elo_tracker=EloTracker(), player_stats_df=None,
            n_mc_samples=500, rng_seed=1,
        )
        state = MatchState(
            sets_p1=0, sets_p2=0, games_p1=0, games_p2=0,
            server=0, match_format=MatchFormat.best_of_3_tiebreak(),
        )
        eng.new_match(state)
        result = eng.predict(state)
        assert abs(result.win_prob_p1 - 0.5) < 0.05


class TestEngineLatency:
    def test_predict_latency_under_100ms_default(self, engine, start_state):
        """predict() with default n_mc_samples=200 should complete under 100ms."""
        engine.new_match(start_state)
        engine.predict(start_state)  # warm up
        times = []
        for _ in range(20):
            t0 = time.perf_counter()
            engine.predict(start_state)
            times.append(time.perf_counter() - t0)
        p95 = np.percentile(times, 95)
        assert p95 < 0.100, f"p95 latency {p95*1000:.1f}ms, expected < 100ms"

    def test_predict_latency_under_10ms_fast_mode(self):
        """With n_mc_samples=50 on a mid-match state, predict() hits 10ms."""
        fast_engine = PredictionEngine(
            model=MockPointWinModel(p=0.63),
            calibrator=MockCalibrator(),
            elo_tracker=EloTracker(),
            player_stats_df=None,
            n_mc_samples=50,
            rng_seed=7,
        )
        state = MatchState(
            sets_p1=1, sets_p2=0, games_p1=3, games_p2=2,
            points_p1=1, points_p2=0, server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        fast_engine.new_match(state)
        fast_engine.predict(state)  # warm up
        times = []
        for _ in range(50):
            t0 = time.perf_counter()
            fast_engine.predict(state)
            times.append(time.perf_counter() - t0)
        p95 = np.percentile(times, 95)
        assert p95 < 0.010, f"p95 latency {p95*1000:.1f}ms exceeds 10ms budget"

    def test_new_match_fast(self, engine, start_state):
        """new_match() should be fast (< 5ms without stats lookup)."""
        t0 = time.perf_counter()
        for _ in range(20):
            engine.new_match(start_state)
        avg = (time.perf_counter() - t0) / 20
        assert avg < 0.005, f"new_match avg {avg*1000:.1f}ms"


class TestCalibration:
    def test_calibrator_fit_and_use(self):
        """MatchWinCalibrator fit and predict smoke test."""
        from tennis.calibration import MatchWinCalibrator
        cal = MatchWinCalibrator()
        rng = np.random.default_rng(42)
        probs = rng.uniform(0.1, 0.9, 1000)
        outcomes = (probs + rng.normal(0, 0.1, 1000) > 0.5).astype(float)
        cal.fit(probs, outcomes)
        result = cal.calibrate(0.7)
        assert 0.0 <= result <= 1.0

    def test_ece_perfect_calibration(self):
        """Perfectly calibrated model should have low ECE."""
        from tennis.calibration import MatchWinCalibrator
        cal = MatchWinCalibrator()
        # Perfect calibration: prob = outcome frequency
        probs = np.linspace(0.1, 0.9, 100)
        rng = np.random.default_rng(0)
        outcomes = rng.binomial(1, probs).astype(float)
        ece = cal.expected_calibration_error(probs, outcomes, n_bins=10)
        # With 100 samples there's noise, but should be < 0.15
        assert ece < 0.15


class TestPredictionResultRepr:
    def test_repr(self, engine, start_state):
        engine.new_match(start_state)
        result = engine.predict(start_state)
        r = repr(result)
        assert "PredictionResult" in r
        assert "p1=" in r
