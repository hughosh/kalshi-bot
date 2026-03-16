"""Tests for the analytical Markov chain engine.

Validates key formulas against known theoretical results and symmetry properties.
"""
import pytest
import numpy as np

from tennis.markov import (
    game_win_prob,
    game_win_prob_from_score,
    tiebreak_win_prob,
    tiebreak_win_prob_from_score,
    set_win_prob,
    set_win_prob_from_score,
    match_win_prob,
    match_win_prob_batch,
)
from tennis.types import MatchFormat, MatchState


# ---------------------------------------------------------------------------
# game_win_prob
# ---------------------------------------------------------------------------

class TestGameWinProb:
    def test_symmetry(self):
        """g(0.5) == 0.5 by symmetry."""
        assert abs(game_win_prob(0.5) - 0.5) < 1e-10

    def test_monotone(self):
        """Higher p → higher game win probability."""
        probs = [game_win_prob(p) for p in np.linspace(0.1, 0.9, 20)]
        for i in range(len(probs) - 1):
            assert probs[i] < probs[i + 1]

    def test_known_value_p06(self):
        """p=0.6 → game win prob ≈ 0.7357 (Newton & Keller benchmark)."""
        g = game_win_prob(0.6)
        assert abs(g - 0.7357) < 0.001, f"Got {g}"

    def test_extremes(self):
        """Near-deterministic probabilities."""
        assert game_win_prob(0.99) > 0.999
        assert game_win_prob(0.01) < 0.001

    def test_range(self):
        for p in np.linspace(0.01, 0.99, 50):
            g = game_win_prob(p)
            assert 0.0 <= g <= 1.0

    def test_complement(self):
        """g(p) + g(1-p) == 1 (if one player wins, the other loses)."""
        for p in [0.3, 0.5, 0.6, 0.7]:
            assert abs(game_win_prob(p) + game_win_prob(1 - p) - 1.0) < 1e-10


class TestGameWinProbFromScore:
    def test_already_won(self):
        assert game_win_prob_from_score(0.6, 4, 0) == 1.0
        assert game_win_prob_from_score(0.6, 4, 1) == 1.0
        assert game_win_prob_from_score(0.6, 4, 2) == 1.0

    def test_already_lost(self):
        assert game_win_prob_from_score(0.6, 0, 4) == 0.0
        assert game_win_prob_from_score(0.6, 1, 4) == 0.0

    def test_start_matches_full(self):
        """From 0-0, should equal game_win_prob(p)."""
        for p in [0.4, 0.5, 0.6, 0.7]:
            assert abs(game_win_prob_from_score(p, 0, 0) - game_win_prob(p)) < 1e-10

    def test_server_advantage(self):
        """From 40-0 (3,0), server should be heavily favoured."""
        assert game_win_prob_from_score(0.6, 3, 0) > 0.95

    def test_returner_advantage(self):
        """From 0-40 (0,3), returner should be heavily favoured."""
        assert game_win_prob_from_score(0.6, 0, 3) < 0.15

    def test_deuce(self):
        """At deuce (3,3) with p=0.5 → 0.5."""
        assert abs(game_win_prob_from_score(0.5, 3, 3) - 0.5) < 1e-10

    def test_deuce_p06(self):
        """At deuce with p=0.6: p²/(p²+(1-p)²) = 0.36/0.52 ≈ 0.6923."""
        expected = 0.6**2 / (0.6**2 + 0.4**2)
        assert abs(game_win_prob_from_score(0.6, 3, 3) - expected) < 1e-10

    def test_advantage_server(self):
        """Server advantage (4,3): win = p."""
        p = 0.65
        assert abs(game_win_prob_from_score(p, 4, 3) - p) < 1e-10

    def test_advantage_returner(self):
        """Returner advantage (3,4): win prob should be low."""
        p = 0.6
        # P(win) = p * P(win from deuce)
        pd = p**2 / (p**2 + (1 - p)**2)
        expected = p * pd
        assert abs(game_win_prob_from_score(p, 3, 4) - expected) < 1e-10


# ---------------------------------------------------------------------------
# tiebreak_win_prob
# ---------------------------------------------------------------------------

class TestTiebreakWinProb:
    def test_symmetry(self):
        """p=0.5 → 0.5 by symmetry."""
        assert abs(tiebreak_win_prob(0.5) - 0.5) < 1e-10

    def test_range(self):
        for p in np.linspace(0.3, 0.7, 20):
            tb = tiebreak_win_prob(p)
            assert 0.0 <= tb <= 1.0

    def test_symmetric_gives_half(self):
        """With p_even=1-p_odd (both players identical), win prob ≈ 0.5."""
        for p in [0.4, 0.5, 0.6, 0.65]:
            tb = tiebreak_win_prob(p)
            assert abs(tb - 0.5) < 0.05, f"p={p} gave {tb}"

    def test_monotone_asymmetric(self):
        """With fixed p_even, higher p_odd → higher tiebreak win probability."""
        probs = [tiebreak_win_prob(p_odd=p, p_even=0.37) for p in np.linspace(0.3, 0.8, 20)]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]

    def test_from_score_at_start_equals_full(self):
        for p in [0.5, 0.6, 0.55]:
            assert abs(
                tiebreak_win_prob_from_score(p, 0, 0) - tiebreak_win_prob(p)
            ) < 1e-9

    def test_won(self):
        assert tiebreak_win_prob_from_score(0.6, 7, 0) == 1.0
        assert tiebreak_win_prob_from_score(0.6, 7, 5) == 1.0

    def test_lost(self):
        assert tiebreak_win_prob_from_score(0.6, 0, 7) == 0.0

    def test_nearly_won(self):
        """6-0 in tiebreak: server needs 1 more, strong favourite."""
        assert tiebreak_win_prob_from_score(0.6, 6, 0) > 0.9


# ---------------------------------------------------------------------------
# set_win_prob
# ---------------------------------------------------------------------------

class TestSetWinProb:
    def test_symmetry(self):
        """With p=0.5, set win prob should be 0.5."""
        fmt = MatchFormat.best_of_3_tiebreak()
        assert abs(set_win_prob(game_win_prob(0.5), match_format=fmt) - 0.5) < 1e-9

    def test_range(self):
        fmt = MatchFormat.best_of_3_tiebreak()
        for p in np.linspace(0.3, 0.8, 20):
            g = game_win_prob(p)
            s = set_win_prob(g, match_format=fmt)
            assert 0.0 <= s <= 1.0

    def test_from_score_start_equals_full(self):
        fmt = MatchFormat.best_of_3_tiebreak()
        for p in [0.5, 0.6]:
            g = game_win_prob(p)
            s_full = set_win_prob(g, match_format=fmt)
            s_score = set_win_prob_from_score(g, 0, 0, fmt, is_final_set=False)
            assert abs(s_full - s_score) < 1e-9


# ---------------------------------------------------------------------------
# match_win_prob
# ---------------------------------------------------------------------------

class TestMatchWinProb:
    def _fresh_state(self, fmt=None):
        if fmt is None:
            fmt = MatchFormat.best_of_3_tiebreak()
        return MatchState(
            sets_p1=0, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=0,
            is_tiebreak=False,
            match_format=fmt,
        )

    def test_symmetry_start_of_match(self):
        """At match start with p=0.5 and server=p1, both players have equal
        fundamental ability so match win should be 0.5."""
        state = self._fresh_state()
        result = match_win_prob(state, 0.5)
        assert abs(result - 0.5) < 1e-8

    def test_range_various_p(self):
        state = self._fresh_state()
        for p in np.linspace(0.4, 0.7, 20):
            prob = match_win_prob(state, p)
            assert 0.0 <= prob <= 1.0

    def test_monotone_in_p(self):
        """Higher p → higher match win prob for server (player 1) when p1 leads."""
        state = MatchState(
            sets_p1=1, sets_p2=0,
            games_p1=3, games_p2=2,
            points_p1=2, points_p2=1,
            server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        probs = [match_win_prob(state, p) for p in np.linspace(0.4, 0.8, 20)]
        for i in range(len(probs) - 1):
            assert probs[i] <= probs[i + 1]

    def test_match_all_but_won(self):
        """p1 has won 2 sets in best-of-3 — should be ~1.0."""
        state = MatchState(
            sets_p1=2, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        # After 2 sets won, match should be over — but match_win_prob
        # from this state: p1 has already won, so _match_dp returns 1.0
        prob = match_win_prob(state, 0.6)
        assert prob >= 0.999

    def test_p1_far_behind(self):
        """p1 down 0-2 in sets (best-of-3) → should be ~0."""
        state = MatchState(
            sets_p1=0, sets_p2=2,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        prob = match_win_prob(state, 0.6)
        assert prob < 0.001

    def test_best_of_5_symmetry(self):
        state = MatchState(
            sets_p1=0, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=0,
            match_format=MatchFormat.best_of_5_tiebreak(),
        )
        prob = match_win_prob(state, 0.5)
        assert abs(prob - 0.5) < 1e-8

    def test_p2_serving_symmetry(self):
        """With p=0.5 and p2 serving at start, should still be 0.5."""
        state = MatchState(
            sets_p1=0, sets_p2=0,
            games_p1=0, games_p2=0,
            points_p1=0, points_p2=0,
            server=1,  # p2 serving
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        prob = match_win_prob(state, 0.5)
        assert abs(prob - 0.5) < 1e-8

    def test_batch_matches_scalar(self):
        state = self._fresh_state()
        p_array = np.array([0.5, 0.55, 0.6, 0.65])
        batch = match_win_prob_batch(state, p_array)
        for i, p in enumerate(p_array):
            scalar = match_win_prob(state, p)
            assert abs(batch[i] - scalar) < 1e-10

    def test_tiebreak_state(self):
        """In a tiebreak at 6-6, p=0.5 → 0.5."""
        state = MatchState(
            sets_p1=0, sets_p2=0,
            games_p1=6, games_p2=6,
            points_p1=0, points_p2=0,
            server=0,
            is_tiebreak=True,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        prob = match_win_prob(state, 0.5)
        # Should be roughly 0.5 (small asymmetry due to service rotation)
        assert abs(prob - 0.5) < 0.05


# ---------------------------------------------------------------------------
# Performance test
# ---------------------------------------------------------------------------

class TestInferenceSpeed:
    def test_match_win_prob_latency(self):
        """Single call to match_win_prob should complete in < 5ms."""
        import time
        state = MatchState(
            sets_p1=1, sets_p2=0,
            games_p1=3, games_p2=2,
            points_p1=1, points_p2=0,
            server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        # Warm up
        match_win_prob(state, 0.6)
        # Time 100 calls
        t0 = time.perf_counter()
        for _ in range(100):
            match_win_prob(state, 0.6)
        elapsed = (time.perf_counter() - t0) / 100
        assert elapsed < 0.005, f"Too slow: {elapsed*1000:.2f}ms per call"

    def test_batch_500_latency(self):
        """500 MC samples should complete in < 50ms total."""
        import time
        state = MatchState(
            sets_p1=1, sets_p2=0,
            games_p1=3, games_p2=2,
            points_p1=1, points_p2=0,
            server=0,
            match_format=MatchFormat.best_of_3_tiebreak(),
        )
        p_samples = np.random.default_rng(42).beta(12, 8, 500)
        t0 = time.perf_counter()
        match_win_prob_batch(state, p_samples)
        elapsed = time.perf_counter() - t0
        assert elapsed < 0.05, f"Batch too slow: {elapsed*1000:.1f}ms"
